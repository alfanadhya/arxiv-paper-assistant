// Start-up status panel: the check that runs when a chat starts (threeofus_main.health_check via
// app.send_status / app.finish_status). Styles: public/theme.css ("start-up status").
// props (and the live data, same shape): { panel_id, state: "checking" | "ok" | "limited" | "failed",
//   steps: [{ key, label, state: "pending" | "running" | "done" | "warn" | "fail", detail, tech }],
//   note (a loading line), error, seq (grows with every update) }
// While checking: a live checklist, so the wait shows progress. Then one line: "Ready" with the details folded
// away, or "Ready with limits" / "Setup failed" with the details open, because then the user should read them.
import { useEffect, useState } from "react";
import { Check, AlertTriangle, X, Loader2, Circle, ChevronRight, RotateCw } from "lucide-react";

// Chainlit remounts custom elements on re-renders, so the live data and the open/closed state live on
// `window`, keyed by panel_id (as in Diagnostics.jsx). Python posts updates as window messages.
const liveData = (window.__startupStatus = window.__startupStatus || {});
const openState = (window.__startupStatusOpen = window.__startupStatusOpen || {});
if (!window.__startupStatusListener) {
  window.__startupStatusListener = true;
  window.addEventListener("message", (event) => {
    const d = event.data;
    if (!d || d.type !== "startup_status" || !d.panel_id) return;
    const old = liveData[d.panel_id];
    if (old && (old.seq || 0) > (d.data.seq || 0)) return;   // an older update arriving late
    liveData[d.panel_id] = d.data;
    window.dispatchEvent(new CustomEvent("startup-status", { detail: d.panel_id }));
  });
}

const STEP_TEXT = { pending: "waiting", running: "in progress", done: "ready", warn: "limited", fail: "failed" };

function StepIcon({ state }) {
  if (state === "done") return <Check className="ss-icon ss-done" aria-hidden="true" />;
  if (state === "warn") return <AlertTriangle className="ss-icon ss-warn" aria-hidden="true" />;
  if (state === "fail") return <X className="ss-icon ss-fail" aria-hidden="true" />;
  if (state === "running") return <Loader2 className="ss-icon ss-spin" aria-hidden="true" />;
  return <Circle className="ss-icon ss-pending" aria-hidden="true" />;
}

function Steps({ steps, showTech }) {
  return (
    <ul className="ss-steps">
      {steps.map((s) => (
        <li key={s.key} data-state={s.state}>
          <StepIcon state={s.state} />
          <span className="ss-step-text">
            <span className="ss-label">{s.label}</span>
            <span className="sr-only">: {STEP_TEXT[s.state] || s.state}</span>
            {s.detail ? <span className="ss-detail"> · {s.detail}</span> : null}
            {showTech && s.tech ? <span className="ss-tech">{s.tech}</span> : null}
          </span>
        </li>
      ))}
    </ul>
  );
}

export default function Status() {
  const panelId = props.panel_id;
  const [, redraw] = useState(0);
  useEffect(() => {
    const onLive = (event) => { if (event.detail === panelId) redraw((n) => n + 1); };
    window.addEventListener("startup-status", onLive);
    return () => window.removeEventListener("startup-status", onLive);
  }, [panelId]);

  const live = liveData[panelId];
  const data = live && (live.seq || 0) >= (props.seq || 0) ? live : props;
  const steps = data.steps || [];
  const state = data.state || "checking";

  // Open/closed (finished panel only): starts open when something needs attention, then follows clicks.
  const attention = state !== "ok";
  const [openLocal, setOpenLocal] = useState(null);
  const open = panelId && panelId in openState ? openState[panelId] : (openLocal === null ? attention : openLocal);
  const toggle = (event) => {
    event.preventDefault();
    if (panelId) openState[panelId] = !open;
    setOpenLocal(!open);
  };

  if (state === "checking") {
    return (
      <div className="startup-status" data-state="checking" role="status" aria-live="polite">
        <div className="ss-head">
          <Loader2 className="ss-icon ss-spin" aria-hidden="true" />
          <span className="ss-title">Getting ready</span>
          <span className="ss-sub">usually a few seconds; up to a minute after a server restart</span>
        </div>
        <Steps steps={steps} showTech={false} />
        {data.note ? <p className="ss-note">{data.note}</p> : null}
        <p className="ss-hint">Meanwhile, pick an example above or start typing your question.</p>
      </div>
    );
  }

  // Finished: one line, details folded away unless something needs attention.
  const warn = steps.find((s) => s.state === "warn");
  const title = state === "ok" ? "Ready" : state === "limited" ? "Ready with limits" : "Setup failed";
  const sub = state === "ok" ? "All systems working"
    : state === "limited" ? `${warn ? warn.label + ": " : ""}${warn && warn.detail ? warn.detail : "see details"}`
    : data.error || "see details";
  const HeadIcon = state === "ok" ? Check : state === "limited" ? AlertTriangle : X;

  return (
    <div className="startup-status" data-state={state} role={state === "failed" ? "alert" : "status"}>
      <details open={open}>
        <summary onClick={toggle} aria-expanded={open}>
          <HeadIcon className={`ss-icon ss-${state === "ok" ? "done" : state === "limited" ? "warn" : "fail"}`}
                    aria-hidden="true" />
          <span className="ss-title">{title}</span>
          <span className="ss-sub">{sub}</span>
          <span className="ss-more">
            {open ? "Hide details" : "Details"}
            <ChevronRight className="ss-chevron" aria-hidden="true" />
          </span>
        </summary>
        <Steps steps={steps} showTech={true} />
        {state === "failed" ? (
          <button type="button" className="ss-retry" onClick={() => window.location.reload()}>
            <RotateCw className="ss-icon" aria-hidden="true" /> Try again
          </button>
        ) : null}
      </details>
    </div>
  );
}
