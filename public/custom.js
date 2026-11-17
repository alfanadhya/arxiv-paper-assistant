// Small UI tweaks for Chainlit, loaded via custom_js in .chainlit/config.toml.
//  1. A "×" close button on every toast.
//  2. After deleting chats in the left pane: when no chat is left, open a fresh chat, so the screen does not
//     keep showing messages of a chat that no longer exists (Chainlit only clears the screen when the deleted
//     chat is the one it thinks is open; a chat missing from the list, e.g. after the DB was reset, stays).
//  3. An always-visible "Reading list" tab on the right edge to reopen the reading-list panel (the in-chat
//     "Reading list" buttons are hidden; the tab clicks them).

// 0. Light theme only (DESIGN.md): Chainlit keeps the user's last choice in localStorage; reset it before the
//    app reads it (this script runs before Chainlit's own, see the <script> order in the page).
try { localStorage.setItem("vite-ui-theme", "light"); } catch (e) { /* storage blocked: default is light */ }

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
    const label = open ? "✕ Hide reading list" : "Reading list";
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
  tab.textContent = "Reading list";
  tab.title = "Show the reading list";
  tab.setAttribute("aria-label", "Show the reading list");
  // The primary action blue (#0000ff, Chainlit's --primary from public/theme.json), flat, label font.
  tab.style.cssText =
    "position:fixed;right:0;top:50%;transform:translateY(-50%);writing-mode:vertical-rl;" +
    "z-index:2147483000;padding:14px 8px;border:none;border-radius:8px 0 0 8px;" +
    "background:hsl(var(--primary, 240 100% 50%));color:hsl(var(--primary-foreground, 0 0% 100%));" +
    "cursor:pointer;font:500 12px 'Space Grotesk','Inter',system-ui,sans-serif;letter-spacing:.073em;" +
    "text-transform:uppercase;opacity:.9;";
  tab.addEventListener("mouseenter", () => (tab.style.opacity = "1"));
  tab.addEventListener("mouseleave", () => (tab.style.opacity = ".9"));
  tab.addEventListener("click", toggleReadingList);

  // The tab is the only way in: the "Reading list" buttons under the greeting and the summaries are hidden.
  // They stay in the page (display:none), because openReadingList() above clicks them.
  const hideButtons = document.createElement("style");
  hideButtons.textContent = 'button[id^="open-list-"] { display: none !important; }';
  document.head.appendChild(hideButtons);

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

// 6. Banner in the middle of the top bar: a fixed usage hint (no mode information), and the border of the
//    message box, which moves to the next of three colours with every message sent (round robin), so each new
//    input is visibly a new turn. Python sends {type: "mode", mode} (app.show_mode) for each message (the mode
//    name is ignored); mode null = a new or reopened chat: grey border until its first message.
//    Styled for the light theme (public/theme.css): pale grey pill with a hairline border, label font, no
//    shadow; the message box border itself is drawn by theme.css from --mode-color.
(function () {
  const ID = "mode-banner";
  const HINT = "Ask anything · or start with search= or title=";
  const COLORS = ["#2563eb", "#ca8a04", "#7c3aed"];   // blue, amber, purple
  let turn = -1;   // index into COLORS of the current message; -1 = no message yet in this chat

  function render() {
    if (location.pathname.startsWith("/login") || location.pathname.startsWith("/signup")) return;
    const header = document.getElementById("header");
    if (!header) return;   // top bar not rendered yet: the observer calls again
    let banner = document.getElementById(ID);
    if (!banner) {
      banner = document.createElement("div");
      banner.id = ID;
      banner.style.cssText =
        "position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);padding:5px 16px;" +
        "border-radius:9999px;font:500 14px var(--f-label, system-ui, sans-serif);white-space:nowrap;" +
        "pointer-events:none;max-width:60vw;overflow:hidden;text-overflow:ellipsis;" +
        "background:var(--c-ash, #f2f2f2);color:var(--c-slate, #717886);border:1px solid var(--c-line, #e0e3e8);";
      if (getComputedStyle(header).position === "static") header.style.position = "relative";
      header.appendChild(banner);
    }
    banner.title = HINT;   // full text on hover when the top bar is too narrow
    if (banner.textContent !== HINT) banner.textContent = HINT;
    // Message box border (public/theme.css): the turn's colour; unset = theme.css's grey (black on focus)
    const color = turn < 0 ? "" : COLORS[turn];
    if (document.documentElement.style.getPropertyValue("--mode-color") !== color) {
      if (color) document.documentElement.style.setProperty("--mode-color", color);
      else document.documentElement.style.removeProperty("--mode-color");
    }
  }

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.type !== "mode") return;
    turn = data.mode ? (turn + 1) % COLORS.length : -1;   // a message: next colour; new/reopened chat: reset
    render();
  });
  new MutationObserver(render).observe(document.documentElement, { childList: true, subtree: true });
  render();
})();

// 10. Loading lines: the assistant's progress messages ("Thinking...", "Searching Qdrant...", "Loading ...")
//     end with "..." and are removed or replaced when the work is done. They get data-loading, which
//     public/theme.css draws as typing dots plus a text shimmer.
(function () {
  function mark() {
    document.querySelectorAll(".ai-message .message-content").forEach((el) => {
      const text = (el.textContent || "").trim();
      const loading = text.length > 0 && text.length < 300 && /(\.\.\.|…)$/.test(text) &&
                      !el.querySelector(".home, table, h1, h2, h3");
      if (loading && !el.hasAttribute("data-loading")) el.setAttribute("data-loading", "");
      else if (!loading && el.hasAttribute("data-loading")) el.removeAttribute("data-loading");
    });
    // No avatar on the greeting (Home screen, or the old text greeting in older chats) and on the start-up
    // message right after it ("Checking the models..." -> "Ready."). theme.css hides it, keeping the indent.
    let afterGreeting = false;
    document.querySelectorAll(".ai-message").forEach((msg) => {
      const text = (msg.textContent || "").trim();
      const greeting = !!msg.querySelector(".home") || text.startsWith("Hi! Ask in your own words");
      const plain = greeting || (afterGreeting && /^(Checking the models|Loading|Ready\.|Setup failed)/.test(text));
      if (plain !== msg.hasAttribute("data-no-avatar")) msg.toggleAttribute("data-no-avatar", plain);
      afterGreeting = greeting;
    });
  }
  new MutationObserver(mark).observe(document.documentElement,
                                     { childList: true, subtree: true, characterData: true });
  mark();
})();

// 7. Green "Best match:" heading in the result summary (app.summary_md writes it as **Best match:**; messages
//    are Markdown without HTML, so the colour is set here), and a green title on the best match's card.
(function () {
  const GREEN = "#098551";   // DESIGN.md "Wireframe Green"
  // The best match's card title: its arXiv link ends in '#best' (app.paper_card)
  const style = document.createElement("style");
  style.textContent = `a[href$="#best"] { color: ${GREEN} !important; }`;
  document.head.appendChild(style);
  function colour() {
    document.querySelectorAll("strong:not([data-best-match])").forEach((el) => {
      if (el.textContent.trim() === "Best match:") {
        el.setAttribute("data-best-match", "1");
        el.style.color = GREEN;
      }
    });
  }
  new MutationObserver(colour).observe(document.documentElement, { childList: true, subtree: true });
  colour();
})();

// 8. Resizable left pane (the chat list). Chainlit gives it a fixed width through the CSS variable
//    --sidebar-width, which the pane, the space it reserves and so the chat area next to it all use. A thin
//    handle on the pane's right edge changes that variable while dragged; everything inside follows (chat
//    titles are cut with "..." by Chainlit, so a wider pane shows more of them). The width is remembered in
//    this browser; a double-click on the handle goes back to Chainlit's width.
(function () {
  const KEY = "left-pane-width";
  const MIN = 180, MAX = 640;
  let original = null;   // Chainlit's own value, for the double-click reset
  let dragging = false;

  // The element that defines --sidebar-width (Chainlit's sidebar wrapper).
  function wrapper() {
    return document.querySelector('[style*="--sidebar-width"]');
  }

  // The visible pane: the outermost box inside the wrapper that holds the chat list.
  function pane() {
    const list = document.getElementById("thread-history");
    if (!list) return null;
    return list.closest('[data-sidebar="sidebar"]') || list.closest('[data-side="left"]') || list.parentElement;
  }

  function saved() {
    try {
      const n = parseInt(localStorage.getItem(KEY) || "", 10);
      return n >= MIN && n <= MAX ? n : null;
    } catch (e) { return null; }
  }

  function apply(px) {
    const w = wrapper();
    if (!w) return;
    if (original === null) original = w.style.getPropertyValue("--sidebar-width");
    const value = px + "px";
    if (w.style.getPropertyValue("--sidebar-width") !== value) w.style.setProperty("--sidebar-width", value);
  }

  const handle = document.createElement("div");
  handle.id = "left-pane-resizer";
  handle.title = "Drag to resize the chat list (double-click to reset)";
  handle.style.cssText =
    "position:fixed;top:0;bottom:0;width:8px;cursor:col-resize;z-index:2147482000;display:none;" +
    "background:transparent;transition:background .15s;touch-action:none;";
  handle.addEventListener("mouseenter", () => (handle.style.background = "hsl(var(--primary, 340 92% 52%) / .35)"));
  handle.addEventListener("mouseleave", () => { if (!dragging) handle.style.background = "transparent"; });

  // No width animation and no text selection while dragging (the pane animates its width when it opens)
  const style = document.createElement("style");
  style.textContent =
    "html[data-resizing-left-pane] [style*='--sidebar-width'], " +
    "html[data-resizing-left-pane] [style*='--sidebar-width'] * { transition: none !important; } " +
    "html[data-resizing-left-pane] { user-select: none !important; cursor: col-resize !important; }";
  document.head.appendChild(style);

  // Put the handle on the pane's right edge; hide it when the pane is closed or not on screen.
  function place() {
    if (!handle.isConnected && document.body) document.body.appendChild(handle);
    const width = saved();
    if (width !== null && !dragging) apply(width);   // also after Chainlit re-renders the wrapper
    const p = pane();
    const box = p && p.getBoundingClientRect();
    if (!box || box.width < 40 || box.right <= 0 || window.innerWidth < 768) {
      handle.style.display = "none";
      return;
    }
    handle.style.display = "block";
    handle.style.left = Math.round(box.right - 4) + "px";
  }

  handle.addEventListener("pointerdown", (e) => {
    const p = pane();
    if (!p) return;
    e.preventDefault();
    dragging = true;
    handle.setPointerCapture(e.pointerId);
    document.documentElement.setAttribute("data-resizing-left-pane", "1");
    const left = p.getBoundingClientRect().left;
    const move = (ev) => {
      const px = Math.max(MIN, Math.min(MAX, Math.round(ev.clientX - left)));
      apply(px);
      handle.style.left = Math.round(left + px - 4) + "px";
      handle.dataset.width = String(px);
    };
    const stop = () => {
      dragging = false;
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", stop);
      handle.removeEventListener("pointercancel", stop);
      document.documentElement.removeAttribute("data-resizing-left-pane");
      handle.style.background = "transparent";
      try { if (handle.dataset.width) localStorage.setItem(KEY, handle.dataset.width); } catch (err) { /* private mode */ }
      place();
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
  });

  handle.addEventListener("dblclick", () => {
    try { localStorage.removeItem(KEY); } catch (err) { /* private mode */ }
    delete handle.dataset.width;
    const w = wrapper();
    if (w && original) w.style.setProperty("--sidebar-width", original);
    setTimeout(place, 300);
  });

  let timer = null;
  function later() {   // the pane slides open/closed in about 0.2 s: place the handle again after that
    place();
    clearTimeout(timer);
    timer = setTimeout(place, 300);
  }
  new MutationObserver(() => { if (!dragging) later(); })
    .observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-state"] });
  window.addEventListener("resize", later);
  later();
})();

// 9. "⚙ Settings" button in the top bar. Chainlit only shows a small gear inside the message box for its chat
//    settings (the "Reranker cutoff" slider, threeofus_main.send_settings); this button opens the same panel
//    by clicking that gear (id 'chat-settings-open-modal').
(function () {
  const ID = "settings-button";

  // The gear in the message box is hidden (the top-bar button replaces it); it stays in the page so that
  // the click below still reaches it.
  const style = document.createElement("style");
  style.textContent = "#chat-settings-open-modal { display: none !important; }";
  document.head.appendChild(style);

  function rightGroup() {
    const header = document.getElementById("header");
    if (!header) return null;
    return [...header.children].reverse().find((el) => el.classList.contains("gap-1")) || null;
  }

  function render() {
    if (location.pathname.startsWith("/login") || location.pathname.startsWith("/signup")) return;
    const right = rightGroup();
    if (!right) return;   // top bar not rendered yet: the observer calls again
    let btn = document.getElementById(ID);
    if (!btn) {
      btn = document.createElement("button");
      btn.id = ID;
      btn.type = "button";
      btn.textContent = "⚙ Settings";
      btn.style.cssText =
        "display:inline-flex;align-items:center;font-size:14px;cursor:pointer;white-space:nowrap;" +
        "margin:0 8px;padding:4px 10px;border:1px solid hsl(var(--border, 0 0% 80%));border-radius:6px;" +
        "background:transparent;color:inherit;";
      btn.addEventListener("click", () => {
        const gear = document.getElementById("chat-settings-open-modal");
        if (gear) gear.click();
        else btn.title = "Settings are not loaded yet: start a new chat or reload the page";
      });
    }
    if (btn.parentElement !== right) right.prepend(btn);
    btn.title = "Open the settings (reranker cutoff)";
  }

  new MutationObserver(render).observe(document.documentElement, { childList: true, subtree: true });
  render();
})();

// 4. Login page: a "Create an account" link to /signup, and a note after a successful sign-up.
// The link is only shown when the server allows sign-up (ALLOW_SIGNUP, asked once via /signup-enabled).
(function () {
  let signupEnabled = null;   // null = not known yet
  fetch("/signup-enabled")
    .then((r) => r.json())
    .then((j) => { signupEnabled = !!j.enabled; addSignupLink(); })
    .catch(() => { signupEnabled = true; addSignupLink(); });

  function addSignupLink() {
    if (!location.pathname.startsWith("/login") || signupEnabled !== true) return;
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

// 9. Chat list as one plain list: Chainlit groups the chats under date headings ("Today", "Yesterday",
//    "Previous 7 days", ...). The headings are hidden and the groups packed together, so the left pane shows
//    a single list, newest first (Chainlit already orders the groups and the chats in them by date).
(function () {
  const style = document.createElement("style");
  style.id = "flat-chat-list";
  style.textContent =
    "#thread-history [data-sidebar='group-label'] { display: none !important; } " +
    "#thread-history [data-sidebar='group'] + [data-sidebar='group'] { padding-top: 0 !important; } " +
    "#thread-history [data-sidebar='group']:not(:last-child) { padding-bottom: 0 !important; }";
  (document.head || document.documentElement).appendChild(style);
})();
