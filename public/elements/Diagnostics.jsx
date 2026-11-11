// Collapsible diagnostic digest shown under each answer.
// `props` comes from Python (chatbot_diagnostics.Diagnostics.to_dict() -> app.send_diagnostics_panel):
//   { original_prompt, rewritten_prompt, modified_query, mode, messages: [str], ranks: [CandidateRank] }
import { useState } from "react";
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
};

function tagClass(message) {
  if (/FAILED|FALLBACK|failed|WARNING/.test(message)) return "text-red-600";
  const tag = Object.keys(TAG_STYLES).find((t) => message.startsWith(t));
  return tag ? TAG_STYLES[tag] : "text-muted-foreground";
}

// Rank cell: value or a dash when that stage was not reached.
const cell = (v) => (v === null || v === undefined ? "–" : v);

export default function Diagnostics() {
  const [open, setOpen] = useState(false);
  const messages = props.messages || [];
  const ranks = props.ranks || [];
  const isChat = props.mode === "chat";
  const issues = messages.filter((m) => /FAILED|FALLBACK|failed|Dropped|ADJUSTED/.test(m)).length;
  const norm = (s) => (s || "").trim().toLowerCase().split(/\s+/).join(" ");
  const queryChanged = Boolean(props.rewritten_prompt) ||
    (props.modified_query && norm(props.modified_query) !== norm(props.original_prompt));
  const moved = ranks.filter((r) => r.rerank_rank && r.rerank_rank !== r.initial_rank).length;
  const summary = [
    isChat ? "chat (no search)" :
      props.modified_query ? `query: ${props.modified_query}${queryChanged ? " (changed)" : ""}` : "no query",
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
        <span className="font-medium">Diagnostics</span>
        <span className="text-xs text-muted-foreground truncate">{summary}</span>
      </button>

      {open && (
        <div className="border-t px-3 py-2 space-y-3">
          <table className="text-xs">
            <tbody>
              <tr><td className="pr-3 text-muted-foreground">Mode</td><td>{isChat ? "chat with the LLM" : "database search"}</td></tr>
              <tr><td className="pr-3 text-muted-foreground">Original prompt</td><td>{props.original_prompt}</td></tr>
              {props.rewritten_prompt && (
                <tr><td className="pr-3 text-muted-foreground">Follow-up rewrite</td><td>{props.rewritten_prompt}</td></tr>
              )}
              {!isChat && (
                <tr>
                  <td className="pr-3 text-muted-foreground">Modified query</td>
                  <td className="font-medium">{props.modified_query || "(none)"}</td>
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
              <p className="text-xs font-medium mb-1">Rank trace (Qdrant → reranker → LLM → shown)</p>
              <table className="text-xs w-full">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="text-left pr-2">Pool</th>
                    <th className="text-right pr-2" title="Position in the pool in Qdrant's hybrid (RRF) order">Qdrant</th>
                    <th className="text-right pr-2" title="Position after the cross-encoder reranker">Rerank</th>
                    <th className="text-right pr-2" title="Cross-encoder score">Score</th>
                    <th className="text-right pr-2" title="Label the LLM saw">Sent</th>
                    <th className="text-right pr-2" title="Order of the LLM's (or fallback's) pick">LLM</th>
                    <th className="text-right pr-2" title="Position in the displayed list">Shown</th>
                    <th className="text-right pr-2" title="Publication year">Year</th>
                    {ranks.some((r) => r.title_sim !== null && r.title_sim !== undefined) && (
                      <th className="text-right pr-2" title="Title similarity to the typed title (title lookups)">Title sim</th>
                    )}
                    <th className="text-left">Title</th>
                  </tr>
                </thead>
                <tbody>
                  {ranks.map((r) => (
                    <tr key={r.chunk_id} className={r.final_rank ? "font-medium" : "text-muted-foreground"}>
                      <td className="pr-2">{r.pool}</td>
                      <td className="text-right pr-2" title={`${r.search} search #${r.search_rank}, RRF ${r.rrf_score}`}>{r.initial_rank}</td>
                      <td className="text-right pr-2">{cell(r.rerank_rank)}</td>
                      <td className="text-right pr-2">{r.rerank_score === null || r.rerank_score === undefined ? "–" : r.rerank_score.toFixed(2)}</td>
                      <td className="text-right pr-2">{r.llm_label ? `[${r.llm_label}]` : "–"}</td>
                      <td className="text-right pr-2">{r.llm_rank ? `${r.llm_rank}${r.picked_by === "fallback" ? "*" : ""}` : "–"}</td>
                      <td className="text-right pr-2">{cell(r.final_rank)}</td>
                      <td className="text-right pr-2">{cell(r.year)}</td>
                      {ranks.some((x) => x.title_sim !== null && x.title_sim !== undefined) && (
                        <td className="text-right pr-2">{r.title_sim === null || r.title_sim === undefined ? "–" : r.title_sim.toFixed(2)}</td>
                      )}
                      <td className="truncate max-w-[18rem]" title={r.title}>{r.title}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {ranks.some((r) => r.picked_by === "fallback") && (
                <p className="text-xs text-muted-foreground mt-1">* picked by the fallback (LLM unavailable)</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
