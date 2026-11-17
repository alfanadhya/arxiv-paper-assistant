// Collapsible diagnostic digest shown under each answer.
// `props` comes from Python (chatbot_diagnostics.Diagnostics.to_dict() -> app.send_diagnostics_panel):
//   { original_prompt, rewritten_prompt, modified_query, mode, messages: [str], ranks: [CandidateRank] }
// mode "agent": ranks are grouped by search (pool = "#1", "#2", ...; search = the agent's query).
// Live panel (props.live): sent empty when a message arrives; Python then posts the buffer as it fills
//   (app.watch_diagnostics -> window message {type: "diagnostics_live", panel_id, data}) and the panel shows
//   that data instead of its props. props.replaces (finished panel) = id of the live panel it takes over from.
import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Stethoscope } from "lucide-react";

// Colour per tag so fallbacks and failures stand out.
const TAG_STYLES = {
  "[mode]": "text-slate-500",
  "[analysis]": "text-blue-600",
  "[validation]": "text-amber-600",
  "[rerank]": "text-indigo-600",
  "[ranking]": "text-purple-600",
  "[memory]": "text-teal-600",
  "[selection]": "text-slate-500",
  "[context]": "text-cyan-700",
  "[agent]": "text-violet-600",
  "[scope]": "text-amber-600",
  "[query]": "text-slate-500",
};

function tagClass(message) {
  if (/FAILED|FALLBACK|failed|WARNING/.test(message)) return "text-red-600";
  const tag = Object.keys(TAG_STYLES).find((t) => message.startsWith(t));
  return tag ? TAG_STYLES[tag] : "text-muted-foreground";
}

// Rank cell: value or a dash when that stage was not reached.
const cell = (v) => (v === null || v === undefined ? "–" : v);

// Chainlit remounts custom elements whenever the chat re-renders (status updates, button clicks, sidebar
// refresh), which would reset useState and collapse an open panel. So the open/closed state lives on
// `window`, keyed by props.panel_id (one per diagnostics message).
const openPanels = (window.__diagnosticsOpen = window.__diagnosticsOpen || {});

// Latest data of each live panel, kept on `window` for the same reason. One page-wide listener stores what
// Python posts and tells the mounted panel (if any) to redraw.
const liveData = (window.__diagnosticsLive = window.__diagnosticsLive || {});
if (!window.__diagnosticsLiveListener) {
  window.__diagnosticsLiveListener = true;
  window.addEventListener("message", (event) => {
    const d = event.data;
    if (!d || d.type !== "diagnostics_live" || !d.panel_id) return;
    liveData[d.panel_id] = d.data;
    window.dispatchEvent(new CustomEvent("diagnostics-live", { detail: d.panel_id }));
  });
}

export default function Diagnostics() {
  const panelId = props.panel_id;
  const isLive = Boolean(props.live);
  // A live panel starts open (it is there to be watched). The finished panel that replaces it opens if the
  // live one was still open; any other panel (e.g. in a reopened chat) starts closed.
  if (isLive && panelId && !(panelId in openPanels)) openPanels[panelId] = true;
  if (!isLive && panelId && !(panelId in openPanels) && props.replaces && props.replaces in openPanels) {
    openPanels[panelId] = openPanels[props.replaces];
  }
  const [open, setOpenState] = useState(Boolean(panelId && openPanels[panelId]));
  const setOpen = (value) => {
    if (panelId) openPanels[panelId] = value;
    setOpenState(value);
  };
  const [, redraw] = useState(0);
  useEffect(() => {
    if (!isLive) return undefined;
    const onLive = (event) => { if (event.detail === panelId) redraw((n) => n + 1); };
    window.addEventListener("diagnostics-live", onLive);
    return () => window.removeEventListener("diagnostics-live", onLive);
  }, [panelId, isLive]);
  const data = (isLive && liveData[panelId]) || props;
  const messages = data.messages || [];
  const ranks = data.ranks || [];
  const isChat = data.mode === "chat";
  const isAgent = data.mode === "agent";   // tool-calling answer: one rank-trace block per search
  const searches = isAgent ? new Set(ranks.map((r) => r.pool)).size : 0;
  const issues = messages.filter((m) => /FAILED|FALLBACK|failed|Dropped|ADJUSTED/.test(m)).length;
  const norm = (s) => (s || "").trim().toLowerCase().split(/\s+/).join(" ");
  const queryChanged = Boolean(data.rewritten_prompt) ||
    (data.modified_query && norm(data.modified_query) !== norm(data.original_prompt));
  const moved = ranks.filter((r) => r.rerank_rank && r.rerank_rank !== r.initial_rank).length;
  const summary = [
    isChat ? "chat (no search)" :
      isAgent ? (searches ? `agent: ${searches} search${searches === 1 ? "" : "es"} · ${data.modified_query}` :
                 "agent (no search)") :
      data.modified_query ? `query: ${data.modified_query}${queryChanged ? " (changed)" : ""}` : "no query",
    messages.length ? `${messages.length} note${messages.length === 1 ? "" : "s"}` : "no notes",
    issues ? `${issues} adjustment${issues === 1 ? "" : "s"}` : null,
    ranks.length ? `${moved} rank change${moved === 1 ? "" : "s"}` : null,
  ].filter(Boolean).join(" · ");

  return (
    <div className="rounded-md border text-sm">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-muted"
      >
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        <Stethoscope className="h-4 w-4" />
        <span className="font-medium">{isLive ? "Diagnostics (running…)" : "Diagnostics"}</span>
        <span className="text-xs text-muted-foreground truncate">{summary}</span>
      </button>

      {open && (
        <div className="border-t px-3 py-2 space-y-3">
          <table className="text-xs">
            <tbody>
              <tr><td className="pr-3 text-muted-foreground">Mode</td><td>{isChat ? "chat with the LLM" : isAgent ? "agent (the LLM searches with tools)" : "database search"}</td></tr>
              <tr><td className="pr-3 text-muted-foreground">Original prompt</td><td>{data.original_prompt}</td></tr>
              {data.rewritten_prompt && (
                <tr><td className="pr-3 text-muted-foreground">Follow-up rewrite</td><td>{data.rewritten_prompt}</td></tr>
              )}
              {!isChat && (
                <tr>
                  <td className="pr-3 text-muted-foreground">{isAgent ? "Agent queries" : "Modified query"}</td>
                  <td className="font-medium">{data.modified_query || "(none)"}</td>
                </tr>
              )}
            </tbody>
          </table>

          {messages.length === 0 ? (
            <p className="text-xs text-muted-foreground">No notes: no fallbacks, adjustments or dropped values.</p>
          ) : (
            <ul className="space-y-1 text-xs font-mono">
              {messages.map((m, i) => (
                <li key={i} className={tagClass(m)}>{m}</li>
              ))}
            </ul>
          )}

          {ranks.length > 0 && (
            <div className="overflow-x-auto">
              <p className="text-xs font-medium mb-1">{isAgent
                ? "Rank trace per search (Qdrant → reranker, best of query / question → returned to the model → cited)"
                : "Rank trace (Qdrant → reranker → LLM → shown)"}</p>
              <table className="text-xs w-full">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="text-left pr-2" title={isAgent ? "n-th search of the agent (hover: its query)" : undefined}>{isAgent ? "Search" : "Pool"}</th>
                    <th className="text-right pr-2" title="Position in the pool in Qdrant's hybrid (RRF) order">Qdrant</th>
                    <th className="text-right pr-2" title="Position after the cross-encoder reranker">Rerank</th>
                    <th className="text-right pr-2" title="Places moved by the reranker: ↑ up, ↓ down, = unchanged">Change</th>
                    <th className="text-right pr-2" title="Cross-encoder score">Score</th>
                    <th className="text-right pr-2" title={isAgent ? "Position among the papers returned to the model (score at or above the cutoff)" : "Label the LLM saw"}>{isAgent ? "Returned" : "Sent"}</th>
                    {!isAgent && <th className="text-right pr-2" title="Order of the LLM's (or fallback's) pick">LLM</th>}
                    <th className="text-right pr-2" title={isAgent ? "Card number if the answer cites it" : "Position in the displayed list"}>{isAgent ? "Cited" : "Shown"}</th>
                    <th className="text-right pr-2" title="Publication year">Year</th>
                    {ranks.some((r) => r.title_sim !== null && r.title_sim !== undefined) && (
                      <th className="text-right pr-2" title="Title similarity to the typed title (title lookups)">Title sim</th>
                    )}
                    <th className="text-left pr-2" title="arXiv id">arXiv</th>
                    <th className="text-left pr-2" title="First three authors">Authors</th>
                    <th className="text-left">Title</th>
                  </tr>
                </thead>
                <tbody>
                  {ranks.map((r) => (
                    <tr key={r.chunk_id} className={r.final_rank ? "font-medium" : "text-muted-foreground"}>
                      <td className="pr-2" title={isAgent ? r.search : undefined}>{r.pool}</td>
                      <td className="text-right pr-2" title={`${r.search} search #${r.search_rank}, RRF ${r.rrf_score}`}>{r.initial_rank}</td>
                      <td className="text-right pr-2">{cell(r.rerank_rank)}</td>
                      <td className="text-right pr-2">{!r.rerank_rank || !r.initial_rank ? "–" :
                        r.initial_rank === r.rerank_rank ? "=" :
                        r.initial_rank > r.rerank_rank ? `↑${r.initial_rank - r.rerank_rank}` :
                        `↓${r.rerank_rank - r.initial_rank}`}</td>
                      <td className="text-right pr-2">{r.rerank_score === null || r.rerank_score === undefined ? "–" : r.rerank_score.toFixed(2)}</td>
                      <td className="text-right pr-2" title={r.picked_by === "skipped" ? "LLM ranking skipped" : undefined}>{r.llm_label ? (isAgent ? r.llm_label : `[${r.llm_label}]`) : "–"}</td>
                      {!isAgent && <td className="text-right pr-2" title={r.picked_by === "skipped" ? "LLM ranking skipped" : undefined}>{r.llm_rank ? `${r.llm_rank}${r.picked_by === "fallback" ? "*" : ""}` : "–"}</td>}
                      <td className="text-right pr-2">{isAgent && r.final_rank ? `[${r.final_rank}]` : cell(r.final_rank)}</td>
                      <td className="text-right pr-2">{cell(r.year)}</td>
                      {ranks.some((x) => x.title_sim !== null && x.title_sim !== undefined) && (
                        <td className="text-right pr-2">{r.title_sim === null || r.title_sim === undefined ? "–" : r.title_sim.toFixed(2)}</td>
                      )}
                      <td className="pr-2 whitespace-nowrap">{cell(r.arxiv_id)}</td>
                      <td className="pr-2 truncate max-w-[14rem]" title={r.authors || undefined}>{r.authors || "–"}</td>
                      <td className="truncate max-w-[18rem]" title={r.title}>{r.title}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {ranks.some((r) => r.picked_by === "fallback") && (
                <p className="text-xs text-muted-foreground mt-1">* picked by the fallback (LLM unavailable)</p>
              )}
              {ranks.some((r) => r.picked_by === "skipped") && (
                <p className="text-xs text-muted-foreground mt-1">Surveys: – = LLM ranking skipped (reranker top)</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
