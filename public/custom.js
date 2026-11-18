// Small UI tweaks for Chainlit, loaded via custom_js in .chainlit/config.toml.
//  1. A "×" close button on every toast.
//  2. After deleting chats in the left pane: when no chat is left, open a fresh chat, so the screen does not
//     keep showing messages of a chat that no longer exists (Chainlit only clears the screen when the deleted
//     chat is the one it thinks is open; a chat missing from the list, e.g. after the DB was reset, stays).
//  3. A "Reading list" button in the top bar, with the number of papers in the list, to open and close the
//     reading-list panel (the in-chat "Reading list" buttons are hidden; the button clicks them). Not shown
//     on the login and sign-up pages.

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

  // ------------------------------------------------------------------ 3. "Reading list" button in the top bar
  // Toggles the reading-list panel: opens it by clicking one of the app's own "Reading list" buttons (ids
  // start with 'open-list-'), closes it with the panel's own close button. A badge shows how many papers are
  // in the list (window messages 'reading_list' and 'reading_list_count' from app.py) and pulses when a paper
  // is added, so selecting a paper visibly lands somewhere.
  // Chainlit marks an open panel with #side-view-title (its header holds the close button).
  function panelOpen() {
    return !!document.getElementById("side-view-title");
  }

  function openReadingList() {
    const buttons = document.querySelectorAll('button[id^="open-list-"]');
    if (buttons.length) {
      buttons[buttons.length - 1].click();
      // In a brand-new chat this click is its first interaction: the server creates the chat and Chainlit moves
      // the page from "/" to "/thread/<id>", which closes the panel it just opened. Open it once more then.
      const before = location.pathname;
      setTimeout(() => {
        if (location.pathname !== before && !panelOpen()) {
          const again = document.querySelectorAll('button[id^="open-list-"]');
          if (again.length) again[again.length - 1].click();
        }
      }, 800);
    } else {
      tab.title = "The reading list is not ready yet: start a new chat or run a search first";
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
    window.__readingListStore = { ...store, papers: data.papers || [], pinned: data.pinned !== false,
                                  version: data.version !== undefined ? data.version : store.version };
    setCount((data.papers || []).length);
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

  // Count of papers in the list; null = not known yet (no message since the page loaded)
  let count = null;
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (data && data.type === "reading_list_count") setCount(data.count);
    // A chat started or was reopened (app.show_mode(None)): the panel's stored copy belonged to the chat
    // before, so it is dropped; the panel takes this chat's list from the server when it next opens.
    if (data && data.type === "mode" && !data.mode) delete window.__readingListStore;
  });

  function setCount(n) {
    n = Math.max(0, Number(n) || 0);
    const grew = count !== null && n > count;
    count = n;
    updateTab();
    if (grew && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      tab.classList.remove("rl-pulse");
      void tab.offsetWidth;   // restart the animation
      tab.classList.add("rl-pulse");
    }
  }

  // Badge, pressed state and accessible name for the panel's current state and the current count.
  function updateTab() {
    const open = panelOpen();
    const n = count || 0;
    if (tab.dataset.open !== String(open)) {
      tab.dataset.open = String(open);
      tab.setAttribute("aria-expanded", String(open));
    }
    if (tab.dataset.count !== String(n)) {
      tab.dataset.count = String(n);
      badge.textContent = String(n);
    }
    const title = (open ? "Hide the reading list" : "Show the reading list") +
                  ` (${n} paper${n === 1 ? "" : "s"})`;
    if (tab.title !== title) {
      tab.title = title;
      tab.setAttribute("aria-label", title);
    }
  }

  const tab = document.createElement("button");
  tab.type = "button";
  tab.id = "reading-list-button";
  // Bookmark icon (lucide "bookmark"), label, count badge; styles in the <style> below
  tab.innerHTML =
    '<svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 ' +
    '2 0 0 1 2 2v16z"/></svg><span class="rl-label">Reading list</span><span class="rl-badge">0</span>';
  const badge = tab.querySelector(".rl-badge");
  tab.dataset.count = "0";
  tab.addEventListener("click", toggleReadingList);

  // Same frame as the Settings button next to it; the badge turns primary blue once the list has papers,
  // and the button is shaded while the panel is open.
  const tabStyle = document.createElement("style");
  tabStyle.textContent = `
    #reading-list-button { display:inline-flex; align-items:center; gap:6px; margin:0 8px 0 0;
      padding:4px 6px 4px 10px; font-size:14px; font-weight:500; white-space:nowrap; cursor:pointer;
      color:#fff; background:hsl(var(--primary, 240 100% 50%)); border:1px solid hsl(var(--primary, 240 100% 50%));
      border-radius:6px; transition:background .15s, border-color .15s; }
    #reading-list-button:hover { background:#0000cc; border-color:#0000cc; }
    #reading-list-button:focus-visible { outline:2px solid hsl(var(--primary, 240 100% 50%)); outline-offset:2px; }
    #reading-list-button[data-open="true"] { background:#000099; border-color:#000099; }
    #reading-list-button .rl-badge { min-width:20px; height:20px; padding:0 6px; border-radius:9999px;
      display:inline-flex; align-items:center; justify-content:center; font:600 12px var(--f-label, system-ui);
      background: rgba(255,255,255,.25); color:#fff; }
    #reading-list-button:not([data-count="0"]) .rl-badge { background:#fff; color: hsl(var(--primary, 240 100% 50%)); }
    #reading-list-button.rl-pulse .rl-badge { animation: rl-pulse .6s ease-out; }
    @keyframes rl-pulse { 0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,255,255,.8); }
      40% { transform: scale(1.3); } 100% { transform: scale(1); box-shadow: 0 0 0 8px rgba(255,255,255,0); } }
    @media (max-width: 640px) { #reading-list-button .rl-label { display:none; } }`;
  document.head.appendChild(tabStyle);

  // The button is the only way in: the "Reading list" buttons under the greeting and the summaries are hidden.
  // They stay in the page (display:none), because openReadingList() above clicks them.
  const hideButtons = document.createElement("style");
  hideButtons.textContent = 'button[id^="open-list-"] { display: none !important; }';
  document.head.appendChild(hideButtons);

  // Put the button in the top bar's right-hand group. The page's own controls there always come in one order,
  // whichever script added them last: Newest first, Settings, Reading list, then Chainlit's buttons. Never on
  // the login or sign-up page: taken off the page there if it is still in it.
  const ORDER = ["newest-first-toggle", "settings-button", "reading-list-button"];
  function mountTab() {
    const authPage = location.pathname.startsWith("/login") || location.pathname.startsWith("/signup");
    if (authPage) {
      if (tab.isConnected) tab.remove();
      return;
    }
    const header = document.getElementById("header");
    const right = header && [...header.children].reverse().find((el) => el.classList.contains("gap-1"));
    if (!right) return;   // top bar not rendered yet: the observer calls again
    if (tab.parentElement !== right) right.prepend(tab);
    const items = ORDER.map((id) => document.getElementById(id)).filter((el) => el && el.parentElement === right);
    items.forEach((el, i) => {
      if (right.children[i] !== el) right.insertBefore(el, right.children[i]);
    });
  }
  new MutationObserver(() => {
    mountTab();
    updateTab();   // follows the panel opening/closing, whatever opened or closed it
  }).observe(document.documentElement, { childList: true, subtree: true });
  mountTab();
  updateTab();
  console.log("[custom.js] reading-list button ready");

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
    // full text, and a short one for phones (public/theme.css "top bar on small screens")
    const full = document.createElement("span");
    full.className = "hdr-label";
    full.textContent = "Newest first";
    const short = document.createElement("span");
    short.className = "hdr-short";
    short.textContent = "Newest";
    label.append(full, short);
    right.prepend(label);   // in front of the buttons on the right of the top bar
  }

  let loading = false;
  let lastTry = 0;   // at most one try every 3 s, whatever the server answers
  async function load() {
    if (location.pathname.startsWith("/login") || location.pathname.startsWith("/signup")) return;
    if (loading || Date.now() - lastTry < 3000) return;
    loading = true;
    lastTry = Date.now();
    try {
      const r = await fetch("/prefs/newest-first", { credentials: "same-origin" });
      if (r.ok) { value = !!(await r.json()).value; render(); }
    } catch (e) { /* not logged in yet */ }
    loading = false;
  }

  // The value is asked again as soon as the top bar is on screen while it is still unknown: Chainlit can show
  // its login form at "/" itself (session over), and after signing in there the address does not change, so
  // the first answer (401) would otherwise leave the box out for good.
  new MutationObserver(() => {
    if (value === null) { if (rightGroup()) load(); return; }
    render();
  }).observe(document.documentElement, { childList: true, subtree: true });
  // load once now and again after a login (the page changes from /login to / without a reload)
  load();
  let lastPath = location.pathname;
  setInterval(() => {
    if (location.pathname !== lastPath) { lastPath = location.pathname; if (value === null) load(); }
    else if (value === null && rightGroup()) load();   // signed in at the same address (see above)
  }, 1000);
})();

// 6. Border of the message box: it moves to the next of three colours with every message sent (round robin),
//    so each new input is visibly a new turn. Python sends {type: "mode", mode} (app.show_mode) for each message
//    (the mode name is ignored); mode null = a new or reopened chat: grey border until its first message. The
//    border itself is drawn by public/theme.css from --mode-color. (The usage-hint pill that used to sit in the
//    top bar is gone: the Home screen shows the ways to ask.)
(function () {
  const COLORS = ["#2563eb", "#ca8a04", "#7c3aed"];   // blue, amber, purple
  let turn = -1;   // index into COLORS of the current message; -1 = no message yet in this chat

  function render() {
    // unset = theme.css's grey (black on focus)
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
      // the start-up status panel (Status.jsx), or the plain-text check line of older chats
      const plain = greeting || !!msg.querySelector(".startup-status") ||
                    !!msg.querySelector('img[alt="Avatar for diagnostics"]') ||   // diagnostics panels
                    (afterGreeting && /^(Checking the models|Loading|Ready\.|Setup failed)/.test(text));
      if (plain !== msg.hasAttribute("data-no-avatar")) msg.toggleAttribute("data-no-avatar", plain);
      afterGreeting = greeting;
    });
  }
  new MutationObserver(mark).observe(document.documentElement,
                                     { childList: true, subtree: true, characterData: true });
  mark();
})();

// 7. Paper number chips: each card title ends with a chip ("Paper 2", "Survey S1"; app.paper_card, drawn by
//    public/theme.css, which also adds "Best match" / "Title match" to the best one instead of colouring the
//    title). Here the chip gets a tooltip saying what the number is for.
(function () {
  function hint() {
    document.querySelectorAll(".message-content h3 > code:not([title])").forEach((el) => {
      const m = /^(?:Paper|Survey) (S?\d+)$/.exec(el.textContent.trim());
      if (!m) return;
      el.title = `Use “${m[1]}” to ask about this paper, e.g. “compare ${m[1]} and ` +
                 `${m[1] === "1" ? "2" : "1"}” or “summarize ${m[1]}”`;
    });
  }
  new MutationObserver(hint).observe(document.documentElement, { childList: true, subtree: true });
  hint();
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
      btn.innerHTML = '⚙<span class="hdr-label">&nbsp;Settings</span>';   // gear only on phones (theme.css)
      btn.setAttribute("aria-label", "Settings");
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

// 11. Login page look: the app's own brand instead of Chainlit's.
//     - Top left: a horizontal lockup, the ring logo of the Home screen (public/elements/Home.jsx, static here)
//       + "ARXIV PAPER ASSISTANT", in place of Chainlit's logo image.
//     - Right-hand panel: a GradientASCII animation (public/gradient-ascii.js, loaded only on this page) in the
//       logo's blues, in place of Chainlit's big pink logo, with what the app does on top of it.
//     - Small screens (no right-hand panel): the same one-line tagline under the form title.
//     Styles: public/theme.css ("login page").
(function () {
  const PITCH_TITLE = "Find computer science papers on arXiv";
  const PITCH_LEAD = "Search arXiv papers from 2007 to October 2026 in plain words, save the ones you need and " +
                     "export them to Zotero or Mendeley.";
  // The Home screen's ring (Home.jsx Logo), without the spin; ids prefixed so they never clash
  const RING =
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 200 200" aria-hidden="true">' +
    '<g clip-path="url(#brand_clip)"><mask id="brand_mask" style="mask-type:alpha" width="200" height="200" x="0" ' +
    'y="0" maskUnits="userSpaceOnUse"><path fill="#fff" fill-rule="evenodd" clip-rule="evenodd" d="M100 150c27.614 ' +
    '0 50-22.386 50-50s-22.386-50-50-50-50 22.386-50 50 22.386 50 50 50zm0 50c55.228 0 100-44.772 100-100S155.228 ' +
    '0 100 0 0 44.772 0 100s44.772 100 100 100z"/></mask><g mask="url(#brand_mask)"><path fill="#fff" ' +
    'd="M200 0H0v200h200V0z"/><path fill="#0066FF" fill-opacity="0.33" d="M200 0H0v200h200V0z"/>' +
    '<g filter="url(#brand_blur)"><path fill="#0066FF" d="M110 32H18v68h92V32z"/><path fill="#0044FF" ' +
    'd="M188-24H15v98h173v-98z"/><path fill="#0099FF" d="M175 70H5v156h170V70z"/><path fill="#00CCFF" ' +
    'd="M230 51H100v103h130V51z"/></g></g></g><defs><filter id="brand_blur" width="385" height="410" x="-75" ' +
    'y="-104" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" ' +
    'result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/>' +
    '<feGaussianBlur result="blur" stdDeviation="40"/></filter><clipPath id="brand_clip"><path fill="#fff" ' +
    'd="M0 0H200V200H0z"/></clipPath></defs></svg>';

  let stopArt = null;   // stops the animation (set once it runs)
  let scriptRequested = false;

  // The login page: /login, or Chainlit's login form shown at another address (session over) - a password
  // field with no top bar
  function onLogin() {
    return location.pathname.startsWith("/login") ||
           (!!document.querySelector('form input[type="password"]') && !document.getElementById("header"));
  }

  function brand() {
    const logo = document.querySelector("img.logo");
    if (!logo || document.getElementById("brand-lockup")) return;
    const lockup = document.createElement("div");
    lockup.id = "brand-lockup";
    lockup.innerHTML = RING + '<span class="brand-name">arXiv Paper Assistant</span>';
    lockup.setAttribute("aria-label", "arXiv Paper Assistant");
    lockup.setAttribute("role", "img");
    logo.after(lockup);
  }

  function tagline() {
    const title = document.querySelector("form h1");
    if (!title || document.getElementById("login-tagline")) return;
    const p = document.createElement("p");
    p.id = "login-tagline";
    p.textContent = PITCH_TITLE + ". " + PITCH_LEAD;
    title.after(p);
  }

  function art() {
    const old = document.querySelector('img[src*="/favicon"]');
    const panel = old && old.parentElement;
    if (!panel || panel.querySelector("#login-art")) return;
    panel.id = "login-panel";
    const box = document.createElement("div");
    box.id = "login-art";
    const pitch = document.createElement("div");
    pitch.id = "login-pitch";
    pitch.innerHTML = "<h2></h2><p></p>";   // the name is already in the top-left lockup
    pitch.querySelector("h2").textContent = PITCH_TITLE;
    pitch.querySelector("p").textContent = PITCH_LEAD;
    panel.append(box, pitch);
    const start = () => {
      if (stopArt || !box.isConnected || typeof window.mountGradientASCII !== "function") return;
      stopArt = window.mountGradientASCII(box, {
        lightMode: true, shape: "horizon", colors: ["#0044FF", "#f6f8ff", "#00CCFF", "#0099FF"],
        cellSize: 16, density: 0.8, glyphOpacity: 0.8, glow: 0.15, gradientOpacity: 0.7, speed: 0.8,
      }) || (() => {});
    };
    if (typeof window.mountGradientASCII === "function") {
      start();
    } else if (!scriptRequested) {
      scriptRequested = true;
      const s = document.createElement("script");
      s.src = "/public/gradient-ascii.js?v=1";
      s.onload = start;
      document.head.appendChild(s);
    }
  }

  function render() {
    if (!onLogin()) {
      if (stopArt) { stopArt(); stopArt = null; }   // left the page (signed in): stop drawing
      if (document.documentElement.hasAttribute("data-auth-page")) {
        document.documentElement.removeAttribute("data-auth-page");
      }
      return;
    }
    document.documentElement.setAttribute("data-auth-page", "");
    brand();
    tagline();
    art();
  }
  new MutationObserver(render).observe(document.documentElement, { childList: true, subtree: true });
  render();
})();

// 12. While the start-up check runs (public/elements/Status.jsx, window message 'startup_status'), the message
//     box stays usable for typing but nothing is sent: the placeholder says so, Enter is held back and the
//     send button explains why. Everything goes back to normal when the check ends (ready, limited or failed).
(function () {
  const WAIT_TEXT = "Getting ready… you can start typing";
  let waiting = false;
  let normalPlaceholder = null;   // Chainlit's own placeholder, restored afterwards

  function apply() {
    const box = document.getElementById("chat-input");
    if (box) {
      if (normalPlaceholder === null && box.placeholder !== WAIT_TEXT) normalPlaceholder = box.placeholder;
      const want = waiting ? WAIT_TEXT : normalPlaceholder;
      if (want !== null && box.placeholder !== want) box.placeholder = want;
    }
    const send = document.getElementById("chat-submit");
    if (send) {
      const title = waiting ? "Available in a moment, when the check is done" : "";
      if ((send.getAttribute("title") || "") !== title) {
        if (title) send.setAttribute("title", title);
        else send.removeAttribute("title");
      }
    }
  }

  window.addEventListener("message", (event) => {
    const d = event.data;
    if (!d || d.type !== "startup_status" || !d.data) return;
    const now = d.data.state === "checking";
    if (now !== waiting) {
      waiting = now;
      apply();
    }
  });

  // Enter would send the message: hold it back while waiting (Shift+Enter still makes a new line)
  document.addEventListener("keydown", (event) => {
    if (!waiting || event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    if (!event.target || event.target.id !== "chat-input") return;
    event.preventDefault();
    event.stopPropagation();
  }, true);

  new MutationObserver(() => { if (waiting) apply(); })
    .observe(document.documentElement, { childList: true, subtree: true });
})();

// 13. Account menu at the bottom of the left pane, in place of Chainlit's avatar button in the top bar: the
//     user's initial and name; a click opens a small menu with "Readme" (Chainlit's readme dialog, its top-bar
//     button is hidden and clicked from here) and "Log out" (Chainlit's POST /logout, then the login page).
//     Styles: public/theme.css ("account menu").
(function () {
  const hide = document.createElement("style");
  hide.textContent = "#user-nav-button, #readme-button { display: none !important; }";
  document.head.appendChild(hide);

  // The signed-in user's name (GET /user). Asked when the pane is first shown after signing in: the script is
  // loaded on the login page already, where there is no user yet.
  let name = "";
  let asking = false;
  function askName() {
    if (name || asking) return;
    asking = true;
    fetch("/user", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((u) => { name = (u && (u.display_name || u.identifier)) || ""; update(); })
      .catch(() => {})
      .finally(() => { asking = false; });
  }

  const ICON_BOOK = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 ' +
    '4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>';
  const ICON_OUT = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 17 ' +
    '5-5-5-5"/><path d="M21 12H9"/><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/></svg>';
  const ICON_UP = '<svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7 15 5-5 ' +
    '5 5"/></svg>';

  const footer = document.createElement("div");
  footer.id = "account-footer";
  footer.innerHTML =
    '<div id="account-menu" role="menu" aria-label="Account" hidden>' +
    '<div class="am-who">Signed in as <b class="am-name-menu"></b></div>' +
    '<button type="button" role="menuitem" data-act="readme">' + ICON_BOOK + "<span>Readme</span></button>" +
    '<button type="button" role="menuitem" data-act="logout">' + ICON_OUT + "<span>Log out</span></button>" +
    "</div>" +
    '<button type="button" id="account-button" aria-haspopup="menu" aria-expanded="false">' +
    '<span class="am-avatar" aria-hidden="true"></span><span class="am-name"></span>' + ICON_UP + "</button>";
  const menu = footer.querySelector("#account-menu");
  const button = footer.querySelector("#account-button");

  function update() {
    const shown = name || "Account";
    footer.querySelector(".am-avatar").textContent = (name || "?").charAt(0).toUpperCase();
    footer.querySelector(".am-name").textContent = shown;
    footer.querySelector(".am-name-menu").textContent = shown;
    button.setAttribute("aria-label", `Account: ${shown}`);
  }
  update();

  function setOpen(open) {
    menu.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    if (open) {
      const first = menu.querySelector("[role=menuitem]");
      if (first) first.focus();
    }
  }
  button.addEventListener("click", () => setOpen(menu.hidden));
  menu.addEventListener("click", (event) => {
    const item = event.target.closest("[data-act]");
    if (!item) return;
    setOpen(false);
    if (item.dataset.act === "readme") {
      const readme = document.getElementById("readme-button");
      if (readme) readme.click();
    } else if (item.dataset.act === "logout") {
      fetch("/logout", { method: "POST", credentials: "include" })
        .catch(() => {})
        .finally(() => window.location.assign("/login"));
    }
  });
  // Close on Escape (focus back on the button), on a click elsewhere; arrow keys move between the items
  footer.addEventListener("keydown", (event) => {
    if (menu.hidden) return;
    const items = [...menu.querySelectorAll("[role=menuitem]")];
    const i = items.indexOf(document.activeElement);
    if (event.key === "Escape") { setOpen(false); button.focus(); }
    else if (event.key === "ArrowDown") { event.preventDefault(); items[(i + 1) % items.length].focus(); }
    else if (event.key === "ArrowUp") { event.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
  });
  document.addEventListener("pointerdown", (event) => {
    if (!menu.hidden && !footer.contains(event.target)) setOpen(false);
  });

  // Keep the footer at the bottom of the left pane (also in the phone sheet); never on login / sign-up
  function mount() {
    if (location.pathname.startsWith("/login") || location.pathname.startsWith("/signup")) {
      if (footer.isConnected) footer.remove();
      return;
    }
    const pane = document.querySelector('[data-sidebar="sidebar"]');
    if (pane && footer.parentElement !== pane) pane.appendChild(footer);
    if (pane) askName();
  }
  new MutationObserver(mount).observe(document.documentElement, { childList: true, subtree: true });
  mount();
})();
