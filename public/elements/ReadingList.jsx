// The Reading List panel shown in Chainlit's right sidebar.
// `props.papers` and `props.pinned` come from Python (app.py -> refresh_sidebar).
// `callAction` runs a Python @cl.action_callback with the same name.
//
// Reorder / remove are applied locally first (no sidebar reload), then sent to Python
// with callAction("reorder") / callAction("remove_paper") so the session stays in sync.
// Pin (default on): pinned = panel stays open and refreshes when papers are added;
// unpinned = panel closes right away (like its close arrow) and again whenever a new question is asked
// (reopen with the "Reading list" button).
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Trash2, Download, GripVertical, ChevronUp, ChevronDown, Pin, PinOff } from "lucide-react";

// Chainlit re-evaluates this file (remounting the component) whenever the chat UI re-renders, e.g. while
// any action runs, which wipes React state. So the panel keeps its state on `window`, keyed by the
// server's props.version: same version -> keep the local state; new version -> take the server's list.
function loadStore() {
  const s = window.__readingListStore;
  if (s && s.version === props.version) return s;
  const fresh = { version: props.version, papers: props.papers || [], pinned: props.pinned !== false };
  window.__readingListStore = fresh;
  return fresh;
}

function saveStore(patch) {
  window.__readingListStore = { ...loadStore(), ...patch };
}

export default function ReadingList() {
  const initial = loadStore();
  const [papers, setPapers] = useState(initial.papers);
  const [pinned, setPinned] = useState(initial.pinned);   // default: pinned
  const [dragIndex, setDragIndex] = useState(null);   // the card being dragged
  const [overIndex, setOverIndex] = useState(null);   // the card it's hovering over

  // New server version (panel opened/redrawn): take the server's list.
  useEffect(() => {
    const s = loadStore();
    setPapers(s.papers);
    setPinned(s.pinned);
  }, [props.version]);

  // Paper selected/unselected on a chat card: Python sends the new list as a window message
  // (app.push_reading_list), so the open panel updates in place without being redrawn.
  useEffect(() => {
    function onMessage(event) {
      const data = event.data;
      if (!data || data.type !== "reading_list") return;
      setPapers(data.papers || []);
      setPinned(data.pinned !== false);
      saveStore({ papers: data.papers || [], pinned: data.pinned !== false });
      window.__readingListAck = data.seq;   // tells public/custom.js the update was applied in place
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  function move(from, to) {
    if (from === null || from === to || to < 0 || to >= papers.length) return;
    const next = [...papers];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    setPapers(next);                                             // instant
    saveStore({ papers: next });                                 // survives remounts
    callAction({ name: "reorder", payload: { order: next.map((p) => p.arxiv_id) } });
  }

  function remove(arxiv_id) {
    const next = papers.filter((p) => p.arxiv_id !== arxiv_id);
    setPapers(next);
    saveStore({ papers: next });
    callAction({ name: "remove_paper", payload: { arxiv_id } });
  }

  // Close the panel exactly like its own close arrow (the button in Chainlit's #side-view-title header);
  // small screens show the panel as a sheet without that button, Escape closes it there.
  function closePanel() {
    const close = document.querySelector("#side-view-title button");
    if (close) close.click();
    else document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  }

  // Unpinning also closes the panel at once (after the new state is saved); reopen it with the
  // "Reading list" tab/button, it then stays unpinned until pinned again.
  function togglePin() {
    const next = !pinned;
    setPinned(next);
    saveStore({ pinned: next });
    const saved = callAction({ name: "pin_list", payload: { pinned: next } });
    if (!next) Promise.resolve(saved).finally(closePanel);
  }

  return (
    <div className="flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {pinned ? "Pinned: stays open" : "Unpinned: closes on the next question"}
        </p>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={togglePin}
                title={pinned ? "Unpin panel" : "Pin panel"}>
          {pinned ? <Pin className="h-4 w-4 text-primary" /> : <PinOff className="h-4 w-4" />}
        </Button>
      </div>

      {papers.length === 0 && (
        <p className="text-sm text-muted-foreground">
          Empty. Click "☐ Select" under a paper in the chat.
        </p>
      )}
      {papers.length > 1 && (
        <p className="text-xs text-muted-foreground">Drag ⋮⋮ or use ↑ ↓ to change the order. Export keeps this order.</p>
      )}

      {papers.map((p, i) => (
        <Card
          key={p.arxiv_id}
          draggable
          onDragStart={() => setDragIndex(i)}
          onDragOver={(e) => { e.preventDefault(); setOverIndex(i); }}
          onDragLeave={() => setOverIndex(null)}
          onDrop={(e) => { e.preventDefault(); move(dragIndex, i); setDragIndex(null); setOverIndex(null); }}
          onDragEnd={() => { setDragIndex(null); setOverIndex(null); }}
          className={
            "p-3 flex flex-row gap-2 items-start transition-all " +
            (dragIndex === i ? "opacity-40 " : "") +
            (overIndex === i && dragIndex !== i ? "ring-2 ring-primary " : "")
          }
        >
          <div className="cursor-grab pt-1 text-muted-foreground" title="Drag to reorder">
            <GripVertical className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <a href={p.abs_url} target="_blank" className="text-sm font-medium hover:underline">
              {i + 1}. {p.title}
            </a>
            <p className="text-xs text-muted-foreground mt-1">
              {p.arxiv_id} · {p.year} · {p.authors[0]}{p.authors.length > 1 ? " et al." : ""}
            </p>
          </div>
          <div className="flex flex-col">
            <Button variant="ghost" size="icon" className="h-6 w-6" title="Move up"
                    disabled={i === 0} onClick={() => move(i, i - 1)}>
              <ChevronUp className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6" title="Move down"
                    disabled={i === papers.length - 1} onClick={() => move(i, i + 1)}>
              <ChevronDown className="h-4 w-4" />
            </Button>
          </div>
          <Button variant="ghost" size="icon" title="Remove" onClick={() => remove(p.arxiv_id)}>
            <Trash2 className="h-4 w-4 text-red-600" />
          </Button>
        </Card>
      ))}

      <Button
        className="mt-auto"
        disabled={papers.length === 0}
        onClick={() => callAction({ name: "export", payload: {} })}
      >
        <Download className="h-4 w-4 mr-2" />
        Export and download {papers.length} paper{papers.length === 1 ? "" : "s"}
      </Button>
    </div>
  );
}
