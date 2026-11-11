// Small UI tweaks for Chainlit, loaded via custom_js in .chainlit/config.toml.
//  1. A "×" close button on every toast.
//  2. After deleting chats in the left pane: when no chat is left, open a fresh chat, so the screen does not
//     keep showing messages of a chat that no longer exists (Chainlit only clears the screen when the deleted
//     chat is the one it thinks is open; a chat missing from the list, e.g. after the DB was reset, stays).
//  3. An always-visible "Reading list" tab on the right edge to reopen the reading-list panel.
(function () {
  // ------------------------------------------------------------------ 1. toast close button
  // The toast is hidden, not removed: removing the node would break the toast library's React state,
  // and it cleans the hidden toast up itself once its timer ends.
  const MARK = "data-close-added";

  // Add one close button to a toast element.
  function addCloseButton(toast) {
    if (toast.hasAttribute(MARK)) return;
    toast.setAttribute(MARK, "1");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "×";
    btn.title = "Close";
    btn.setAttribute("aria-label", "Close notification");
    btn.style.cssText =
      "position:absolute;top:4px;right:6px;border:none;background:transparent;cursor:pointer;" +
      "font-size:16px;line-height:1;opacity:.6;color:inherit;padding:2px 4px;z-index:1;";
    btn.addEventListener("mouseenter", () => (btn.style.opacity = "1"));
    btn.addEventListener("mouseleave", () => (btn.style.opacity = ".6"));
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      toast.style.display = "none";
      toast.setAttribute("aria-hidden", "true");
    });
    toast.appendChild(btn);
  }

  // Scan the page for toasts that have no close button yet.
  function scanToasts() {
    document.querySelectorAll("[data-sonner-toast]").forEach(addCloseButton);
  }

  new MutationObserver(scanToasts).observe(document.documentElement, { childList: true, subtree: true });
  scanToasts();

  // ------------------------------------------------------------------ 2. fresh chat when all chats are deleted
  // Number of chats listed in the left pane (links to /thread/<id>).
  function listedChats() {
    return document.querySelectorAll('#thread-history a[href^="/thread/"]').length;
  }

  // True if the chat area shows any message.
  function screenHasMessages() {
    return document.querySelectorAll('[data-step-type], .step').length > 0;
  }

  // After a successful chat delete: if the list is now empty but messages are still on screen, start fresh.
  function afterDelete() {
    setTimeout(() => {
      if (listedChats() === 0 && screenHasMessages()) {
        window.location.assign("/");
      }
    }, 800);   // let Chainlit update the list and run its own clear first
  }

  // ------------------------------------------------------------------ 3. always-visible "Reading list" tab
  // A tab on the right edge that toggles the reading-list panel: opens it by clicking one of the app's own
  // "Reading list" buttons (ids start with 'open-list-'), closes it with the panel's own close button.
  // Chainlit marks an open panel with #side-view-title (its header holds the close button).
  function panelOpen() {
    return !!document.getElementById("side-view-title");
  }

  function openReadingList() {
    const buttons = document.querySelectorAll('button[id^="open-list-"]');
    if (buttons.length) {
      buttons[buttons.length - 1].click();
    } else {
      tab.title = "No reading-list button yet: start a new chat or run a search first";
    }
  }

  function closeReadingList() {
    const close = document.querySelector("#side-view-title button");
    if (close) {
      close.click();
    } else {   // small screens: the panel is a sheet without that button; Escape closes it
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    }
  }

  function toggleReadingList() {
    if (panelOpen()) closeReadingList();
    else openReadingList();
  }

  // A paper was selected/unselected on a chat card (window message from app.push_reading_list): if the
  // panel is closed, open it; if it is open, the panel updates itself in place (no reload).
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.type !== "reading_list" || data.redelivery) return;   // ignore our own re-sends
    // Save the new list into the panel's store first: when Chainlit re-creates the panel (it does while an
    // action runs), the new panel starts from this list, so nothing has to be redrawn from the server.
    const store = window.__readingListStore || {};
    window.__readingListStore = { ...store, papers: data.papers || [], pinned: data.pinned !== false };
    if (!panelOpen()) {
      openReadingList();
      return;
    }
    // Panel open: it updates itself in place and confirms (window.__readingListAck). If the message arrived
    // while the panel was being re-created, re-send it to the page a few times (no server call, no slide-in).
    let tries = 0;
    const redeliver = () => {
      if (window.__readingListAck === data.seq || !panelOpen() || tries++ >= 6) return;
      window.postMessage({ ...data, redelivery: true }, "*");
      setTimeout(redeliver, 250);
    };
    setTimeout(redeliver, 250);
  });

  // Show the right label for the panel's current state.
  function updateTabLabel() {
    const open = panelOpen();
    const label = open ? "✕ Hide reading list" : "📚 Reading list";
    if (tab.textContent !== label) {
      tab.textContent = label;
      tab.title = open ? "Hide the reading list" : "Show the reading list";
      tab.setAttribute("aria-label", tab.title);
      tab.setAttribute("aria-expanded", String(open));
    }
  }

  const tab = document.createElement("button");
  tab.type = "button";
  tab.id = "reading-list-tab";
  tab.textContent = "📚 Reading list";
  tab.title = "Show the reading list";
  tab.setAttribute("aria-label", "Show the reading list");
  // Theme pink: Chainlit's --primary colour (HSL numbers), with the same pink as fallback.
  tab.style.cssText =
    "position:fixed;right:0;top:50%;transform:translateY(-50%);writing-mode:vertical-rl;" +
    "z-index:2147483000;padding:12px 6px;border:none;border-radius:8px 0 0 8px;" +
    "background:hsl(var(--primary, 340 92% 52%));color:hsl(var(--primary-foreground, 0 0% 100%));" +
    "cursor:pointer;font:600 12px system-ui,sans-serif;opacity:.9;box-shadow:0 1px 6px rgba(0,0,0,.25);";
  tab.addEventListener("mouseenter", () => (tab.style.opacity = "1"));
  tab.addEventListener("mouseleave", () => (tab.style.opacity = ".9"));
  tab.addEventListener("click", toggleReadingList);

  // Add the tab once the page body exists, and put it back if the page ever removes it.
  function mountTab() {
    if (document.body && !document.getElementById("reading-list-tab")) document.body.appendChild(tab);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountTab);
  } else {
    mountTab();
  }
  new MutationObserver(() => {
    mountTab();
    updateTabLabel();   // follows the panel opening/closing, whatever opened or closed it
  }).observe(document.documentElement, { childList: true, subtree: true });
  console.log("[custom.js] reading-list tab ready");

  // Watch the browser's requests for chat deletions (DELETE /project/thread).
  const originalFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const response = await originalFetch(input, init);
    try {
      const url = typeof input === "string" ? input : input.url;
      const method = ((init && init.method) || (input && input.method) || "GET").toUpperCase();
      if (method === "DELETE" && /\/project\/thread\/?(\?|$)/.test(url) && response.ok) {
        afterDelete();
      }
    } catch (e) {
      /* never break the app's own requests */
    }
    return response;
  };
})();

// 5. "Newest first" checkbox in the top bar. Saved per user on the server (/prefs/newest-first); when
//    ticked, the papers of the next search are shown newest first, otherwise in relevance order.
(function () {
  const ID = "newest-first-toggle";
  let value = null;   // null until the saved value has been loaded

  // The right-hand button group of Chainlit's top bar: <div id="header"> holds [left group, centre spinner,
  // spacer, right group "flex items-center gap-1"].
  function rightGroup() {
    const header = document.getElementById("header");
    if (!header) return null;
    return [...header.children].reverse().find((el) => el.classList.contains("gap-1")) || null;
  }

  function render() {
    if (location.pathname.startsWith("/login") || location.pathname.startsWith("/signup")) return;
    if (value === null) return;
    const right = rightGroup();
    const existing = document.getElementById(ID);
    if (existing) {
      if (right && existing.parentElement !== right) right.prepend(existing);   // keep it on the right
      return;
    }
    if (!right) return;   // top bar not rendered yet: the observer calls again
    const label = document.createElement("label");
    label.id = ID;
    label.title = "Show the papers of your next search newest first (otherwise by relevance)";
    label.style.cssText =
      "display:inline-flex;align-items:center;gap:6px;font-size:14px;cursor:pointer;user-select:none;" +
      "white-space:nowrap;margin:0 8px;";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = value;
    box.style.cssText = "width:16px;height:16px;cursor:pointer;accent-color:currentColor;";
    box.addEventListener("change", async () => {
      const wanted = box.checked;
      try {
        const r = await fetch("/prefs/newest-first", {
          method: "POST", credentials: "same-origin",
          headers: { "Content-Type": "application/json" }, body: JSON.stringify({ value: wanted }),
        });
        if (!r.ok) throw new Error(r.status);
        value = wanted;
      } catch (e) {
        box.checked = !wanted;   // not saved: show the real state
      }
    });
    label.appendChild(box);
    label.appendChild(document.createTextNode("Newest first"));
    right.prepend(label);   // in front of the buttons on the right of the top bar
  }

  async function load() {
    if (location.pathname.startsWith("/login") || location.pathname.startsWith("/signup")) return;
    try {
      const r = await fetch("/prefs/newest-first", { credentials: "same-origin" });
      if (r.ok) { value = !!(await r.json()).value; render(); }
    } catch (e) { /* not logged in yet */ }
  }

  new MutationObserver(() => { if (value === null) return; render(); })
    .observe(document.documentElement, { childList: true, subtree: true });
  // load once now and again after a login (the page changes from /login to / without a reload)
  load();
  let lastPath = location.pathname;
  setInterval(() => {
    if (location.pathname !== lastPath) { lastPath = location.pathname; if (value === null) load(); }
  }, 1000);
})();

// 4. Login page: a "Create an account" link to /signup, and a note after a successful sign-up.
(function () {
  function addSignupLink() {
    if (!location.pathname.startsWith("/login")) return;
    const pw = document.querySelector('form input[type="password"]');
    const form = pw && pw.closest("form");
    if (!form || form.querySelector("[data-signup-link]")) return;
    const box = document.createElement("div");
    box.setAttribute("data-signup-link", "1");
    box.style.cssText = "text-align:center;font-size:14px;margin-top:4px;";
    if (new URLSearchParams(location.search).get("signup") === "ok") {
      const note = document.createElement("div");
      note.textContent = "Account created. You can log in now.";
      note.style.cssText = "color:#16a34a;margin-bottom:6px;";
      box.appendChild(note);
    }
    box.appendChild(document.createTextNode("No account? "));
    const a = document.createElement("a");
    a.href = "/signup";
    a.textContent = "Create an account";
    a.style.cssText = "text-decoration:underline;";
    box.appendChild(a);
    form.appendChild(box);
  }
  new MutationObserver(addSignupLink).observe(document.documentElement, { childList: true, subtree: true });
  addSignupLink();
})();
