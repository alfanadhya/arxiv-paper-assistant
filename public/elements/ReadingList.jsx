// The Reading List panel shown in Chainlit's right sidebar.
// `props.papers` comes from Python (app.py -> refresh_sidebar).
// `callAction` runs a Python @cl.action_callback with the same name.
//
// Rearranging works two ways:
//   - drag a card by its handle (⋮⋮) and drop it on another card (mouse)
//   - the ↑ / ↓ buttons (works on phones too, where dragging doesn't)
// Either way, the new order is sent to Python with callAction("reorder").
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Trash2, Download, GripVertical, ChevronUp, ChevronDown } from "lucide-react";

export default function ReadingList() {
  const papers = props.papers || [];
  const [dragIndex, setDragIndex] = useState(null);   // the card being dragged
  const [overIndex, setOverIndex] = useState(null);   // the card it's hovering over

  function move(from, to) {
    if (from === to || to < 0 || to >= papers.length) return;
    const order = papers.map((p) => p.arxiv_id);
    const [moved] = order.splice(from, 1);
    order.splice(to, 0, moved);
    callAction({ name: "reorder", payload: { order } });
  }

  return (
    <div className="flex flex-col gap-3 h-full">
      {papers.length === 0 && (
        <p className="text-sm text-muted-foreground">
          Empty. Click "Add to reading list" under a paper in the chat.
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
          <Button
            variant="ghost"
            size="icon"
            title="Remove"
            onClick={() => callAction({ name: "remove_paper", payload: { arxiv_id: p.arxiv_id } })}
          >
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
