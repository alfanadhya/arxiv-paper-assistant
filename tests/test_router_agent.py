"""Checks for the agent and the search=/title= keys (threeofus_main.run_agent / run_turn).

Usage, from the project folder:
    python -X utf8 tests/test_router_agent.py offline   # no LLM, no Qdrant: key handling and helpers
    python -X utf8 tests/test_router_agent.py agent     # live Ollama + Qdrant: one tool-calling answer
    python -X utf8 tests/test_router_agent.py turn      # live: a no-key message, a thanks, keyed search m07
    python -X utf8 tests/test_router_agent.py followup  # live: search, then questions about the shown papers
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import threeofus_main as m

part = sys.argv[1] if len(sys.argv) > 1 else "offline"
fails = []

def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond:
        fails.append(name)

if part == "offline":
    # every message without search= / title= goes to the agent (no LLM here: run_agent is replaced)
    saved_agent, saved_query, saved_title = m.run_agent, m.run_query, m.run_title_query
    calls = []

    def fake_agent(q, *a, **k):
        calls.append(("agent", q))
        return m.SearchResponse(prompt=q, mode="agent", answer="(agent)")

    def fake_pipeline(name):
        def run(q, *a, **k):
            calls.append((name, q))
            return m.SearchResponse(prompt=q, mode=name)
        return run

    m.run_agent, m.run_query, m.run_title_query = fake_agent, fake_pipeline("search"), fake_pipeline("title")
    try:
        cases = [
            ("how are GNNs used for drug discovery?", "agent", "how are GNNs used for drug discovery?"),
            ("analyze= compare 1 and 2", "agent", "compare 1 and 2"),
            ("search= transformers title= attention", "agent", None),   # two keys: just text for the agent
            ("hello", "agent", "hello"),
            ("search= graph neural networks", "search", "graph neural networks"),
            ("title= Attention is all you need", "title", "Attention is all you need"),
        ]
        for msg, want, text in cases:
            calls.clear()
            r = m.run_turn(msg, log=lambda s: None)
            check(f"'{msg}' -> {want}", r.mode == want and len(calls) == 1 and calls[0][0] == want
                  and (text is None or calls[0][1] == text), calls)
        calls.clear()
        bad = m.run_turn("search= GNNs year=19x", log=lambda s: None)
        check("search= with a bad year is rejected", bool(bad.errors) and not calls, bad.errors)
    finally:
        m.run_agent, m.run_query, m.run_title_query = saved_agent, saved_query, saved_title
    # author names stripped from the search topic (author= is the filter)
    strip_cases = [
        ("what has Iryna Gurevych published on misleading charts?", ["Iryna Gurevych"], "misleading charts"),
        ("misleading charts by Gurevych", ["Iryna Gurevych"], "misleading charts"),
        ("Iryna Gurevych's work on fact checking", ["Iryna Gurevych"], "work on fact checking"),
        ("papers by Thomas Kipf on graph convolutional networks", ["Kipf, Thomas"], "graph convolutional networks"),
        ("Turing test for language models", ["Alan Turing"], "Turing test for language models"),
        ("Iryna Gurevych", ["Iryna Gurevych"], "Iryna Gurevych"),   # nothing left -> kept
        ("graph neural networks", ["Kipf; Welling"], "graph neural networks"),
    ]
    for text, authors, want in strip_cases:
        got, _ = m.strip_author_names(text, authors)
        check(f"strip '{text}'", got == want, got)
    from retriever_hybrid_rrf import Evidence
    hit = Evidence(0, 0.0, "p", "2101.00001", "t", "u", {"authors": ["Gurevych, Boris", "Smith, Ann"]}, "t\n\na")
    check("last-name matches", m.last_name_matches([hit], ["Iryna Gurevych"]) == ["Boris Gurevych"],
          m.last_name_matches([hit], ["Iryna Gurevych"]))
    print("offline done")

elif part == "agent":
    t = time.time()
    r = m.run_agent("how are graph neural networks used for drug discovery?", log=lambda s: print("   log:", s, flush=True))
    print(f"--- {time.time() - t:.1f}s, errors={r.errors}, warnings={r.warnings}")
    print("\n".join(r.diagnostics.messages))
    print("ANSWER:", r.answer)
    check("agent answered", bool(r.answer) and not r.errors)
    check("agent searched", any("search_papers" in s for s in r.diagnostics.messages))
    check("agent cited papers", len(r.papers) > 0, f"{len(r.papers)} cited")
    check("cards match labels", r.top_papers == list(range(1, len(r.papers) + 1)))
    print(m.format_response(r))

elif part == "turn":
    t = time.time()
    r = m.run_turn("what has Iryna Gurevych published on misleading claims", log=lambda s: None)
    print(f"--- {time.time() - t:.1f}s mode={r.mode} errors={r.errors}")
    print("\n".join(l for l in r.diagnostics.messages if l.startswith("[agent]")))
    check("no key -> agent", r.mode == "agent" and not r.errors, r.errors)
    check("agent searched by author",
          any("search_papers" in l and "gurevych" in l.lower() for l in r.diagnostics.messages))
    check("agent cited papers", bool(r.papers), [h.title[:50] for h in r.papers])
    r2 = m.run_turn("thanks!", log=lambda s: None)
    check("short reply to thanks", r2.mode == "agent" and bool(r2.answer), r2.answer[:120])
    # eval case m07 exactly as run_eval --mode-key search sends it: the author name must not reach the target
    r3 = m.run_turn("search= what has Iryna Gurevych published on misleading charts? author=Iryna Gurevych",
                    log=lambda s: None)
    target = r3.analysis.corrected_query if r3.analysis else ""
    print("\n".join(l for l in r3.diagnostics.messages if "Removed the author" in l or "Spell check" in l))
    check("m07 target has no author name", "gurev" not in target.lower() and "chart" in target.lower(), target)
    check("m07 got papers", bool(r3.top_papers), [r3.papers[n - 1].title[:50] for n in r3.top_papers])

elif part == "followup":
    # search, then follow-ups about the shown papers (one needs their FULL author lists)
    import app as gui
    r = m.run_turn("search= misleading claims author=Iryna Gurevych", log=lambda s: None)
    labelled = gui.picked_hits(r)
    shown = [{**gui.to_paper(h, label), "best": False} for label, h in labelled]
    known = {p["arxiv_id"]: p for p in shown}
    for p in shown[:3]:
        print(f"  [{p['label']}] {len(p['authors'])} authors: {', '.join(p['authors'])}")
    hist = [{"role": "user", "content": "search= misleading claims author=Iryna Gurevych"},
            {"role": "assistant", "content": m.assistant_turn_text(r)}]
    t = time.time()
    a = m.run_turn("list all authors of 1 2 and 3", log=lambda s: None, history=hist, shown=shown, known=known)
    print(f"--- {time.time() - t:.1f}s mode={a.mode} errors={a.errors}")
    print("\n".join(l for l in a.diagnostics.messages if l.startswith("[agent]")))
    print("ANSWER:", a.answer)
    check("answered by the agent", a.mode == "agent" and not a.errors, a.errors)
    for p in shown[:3]:
        last_names = [n.split()[-1] for n in p["authors"]]
        missing = [n for n in last_names if n.lower() not in a.answer.lower()]
        check(f"all {len(last_names)} authors of [{p['label']}] listed", not missing, f"missing {missing}")
    # no label at all: answered about the shown papers
    n = m.run_turn("which of these papers is the newest?", log=lambda s: None, history=hist, shown=shown, known=known)
    print("NO-LABEL ANSWER:", n.mode, n.errors, n.answer[:300])
    check("no-label follow-up answered", n.mode == "agent" and bool(n.answer) and not n.errors)

print("FAILED:", fails if fails else "none")
