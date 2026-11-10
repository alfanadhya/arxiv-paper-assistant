"""RAG over arXiv papers: query analysis -> hybrid Qdrant search -> local Ollama model picks the best papers.

CLI:  python chat_ollama.py "your prompt"
GUI:  response = run_query(prompt, log=...); then read the SearchResponse fields (or format_response()).
"""

import functools
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from retriever_hybrid_rrf import MIN_YEAR, MetadataIntent, build_retriever

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "sebun_bge768_hybrid_full"   # must match the retriever's model
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))   # Ollama's default context silently truncates
SHOW_REASONING = os.getenv("SHOW_REASONING", "0") == "1"

K_RETRIEVE = 20          # candidates fetched from Qdrant
K_FINAL = 6              # research papers passed to the model
K_SURVEYS = 5            # surveys passed to the model (normal query)
MAX_SURVEYS = 3          # surveys the model may pick (normal query)
K_SURVEYS_FOCUS = 10     # surveys passed to the model (survey query)
MAX_SURVEYS_FOCUS = 6    # surveys the model may pick (survey query)
MAX_PAPERS = 3           # research papers the model may pick
CURRENT_YEAR = date.today().year

DEFAULT_PROMPT = "Generative adversarial networks for cyber security reserach"

# arXiv category codes -> names, e.g. {"cs.CR": "Cryptography and Security"}
CATEGORIES: dict[str, str] = json.loads(
    (Path(__file__).resolve().parent / "csrelated.json").read_text(encoding="utf-8"))

SURVEY_WORD_RE = re.compile(r"\bsurveys?\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
TRAILING_BY_RE = re.compile(r"\bby\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)\s*$")
CATEGORY_RE = re.compile(r"\bcategory\s*:\s*([^,;\n]+)", re.IGNORECASE)
# Leading request phrases ("find me papers about X"); only when followed by on/about/for/...
FILLER_RE = re.compile(
    r"^(?:please\s+)?(?:(?:can|could)\s+you\s+)?"
    r"(?:(?:find|show|give|get|search\s+for|look\s+for|list|recommend|suggest)\s+(?:me\s+)?)?"
    r"(?:(?:some|the|any|recent)\s+)*"
    r"(?:papers?|articles?|research|studies|publications?|work)\s+"
    r"(?:on|about|regarding|for|related\s+to)\s+",
    re.IGNORECASE,
)

LogFn = Callable[[str], None]


# --------------------------------------------------------------------------- schemas
# Cleaned topic + metadata used as Qdrant filters.
class QueryAnalysis(BaseModel):
    """Cleaned topic + metadata used as Qdrant filters."""
    corrected_query: str = Field(
        description="Only the research topic, spelling fixed. No author names, years or filler words.")
    authors: list[str] = Field(default_factory=list,
        description="Author names mentioned in the text, e.g. after 'by'. Empty list if none.")
    year: int | None = Field(default=None, description="Exact publication year, e.g. 'in 2021'. Null if none.")
    year_from: int | None = Field(default=None, description="Start year for 'after/since 2019'. Null if none.")
    year_to: int | None = Field(default=None, description="End year for 'before/until 2020'. Null if none.")
    category: str | None = Field(default=None, description="Always null.")


# Validated analysis; errors skip the search, warnings don't.
class QueryCheck(BaseModel):
    """Validated analysis; errors skip the search, warnings don't."""
    analysis: QueryAnalysis
    errors: list[str] = []
    warnings: list[str] = []


# Structured answer the model must return.
class PaperSearchResult(BaseModel):
    """Structured answer the model must return."""
    criteria: list[str] = Field(
        description="2-4 short criteria used to judge how well a paper matches the query")
    top_papers: list[int] = Field(default_factory=list,
        description="Numbers of the 3 best matching research papers [1], [2], ..., best first, e.g. [4, 1, 7]. "
                    "Empty list if only survey papers are requested.")
    top_surveys: list[int] = Field(default_factory=list,
        description="Numbers of relevant survey papers [S1], [S2], ..., best first, e.g. [2, 1], "
                    "no more than the limit given in the instructions. Empty list if none are relevant.")
    why_best: str = Field(description="One sentence explaining why your first pick is the best match")


# Everything a GUI needs to display one query's result.
@dataclass
class SearchResponse:
    prompt: str
    analysis: QueryAnalysis | None = None
    survey_focus: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    papers: list = field(default_factory=list)          # retrieved research-paper hits
    surveys: list = field(default_factory=list)         # retrieved survey hits
    top_papers: list[int] = field(default_factory=list) # 1-based picks into papers, newest first
    top_surveys: list[int] = field(default_factory=list)# 1-based picks into surveys, newest first
    best: tuple[str, object] | None = None              # (label like "3" / "S2", hit)
    criteria: list[str] = field(default_factory=list)
    why_best: str = ""
    reasoning: str = ""

    # True when the query produced results without errors.
    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------- prompts
ANALYZE_SYSTEM_PROMPT = """You analyze search queries for a database of computer science research papers.
1. Fix spelling mistakes in the research topic. Keep technical terms and acronyms (e.g. GAN, BERT) unchanged.
2. Remove filler words that are not part of the topic, such as "find me papers about", "research on",
   "I want", "please", or a trailing "research"/"papers". Do not add new words.
3. Extract author names. Authors usually come after the word "by", often at the end.
   Copy names exactly as written. If no author is mentioned, return an empty list.
4. Extract years: "in 2021" -> year; "after/since 2019" -> year_from; "before/until 2020" -> year_to.
   Only use years written in the text. Otherwise leave them null.
5. Always return category as null.
6. corrected_query must contain only the topic, without author names, years or filler words.

Examples:
"find me papers about GAN for malwre detection by Ian Goodfellow 2020"
-> corrected_query: "GAN for malware detection", authors: ["Ian Goodfellow"], year: 2020
"transformr models for code generation research since 2021"
-> corrected_query: "transformer models for code generation", authors: [], year_from: 2021
"graph neural netwrks"
-> corrected_query: "graph neural networks", authors: [], all years null"""

TASK_NORMAL = (
    "Then apply them: list the numbers of the three best matching research papers, best first, "
    "and the numbers of up to 3 relevant survey papers, best first (only the number, e.g. 2 for [S2]). "
    "Do not explain each paper; only give one sentence on why the first research paper is the best match.")
TASK_SURVEYS_ONLY = (
    "The user wants survey papers only: leave top_papers empty. "
    "Apply the criteria and list the numbers of up to 6 relevant survey papers, best first "
    "(only the number, e.g. 2 for [S2]). "
    "Do not explain each paper; only give one sentence on why the first survey is the best match.")

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are given research papers numbered [1], [2], ... and survey papers numbered [S1], [S2], ... "
     "(title and abstract each). "
     "First state the criteria you use to judge how well a paper matches the user's query. "
     "{task} "
     "Only use papers from the context.\n\n"
     "Research papers:\n{papers}\n\n"
     "Survey papers:\n{surveys}"),
    ("human", "{query}"),
])


# --------------------------------------------------------------------------- query analysis
# Map a category code, full name or unique name part to its code, or None.
def match_category(value: str) -> str | None:
    v = " ".join(value.lower().split())
    for code, name in CATEGORIES.items():
        if v == code.lower() or v == name.lower():
            return code
    matches = [code for code, name in CATEGORIES.items()
               if v in [p.strip() for p in re.split(r",|\band\b|\(|\)", name.lower()) if p.strip()]]
    return matches[0] if len(matches) == 1 else None


# Extract an explicit 'category: ...'; returns (code, remaining prompt, warning).
def parse_category(text: str) -> tuple[str | None, str, str | None]:
    m = CATEGORY_RE.search(text)
    if not m:
        return None, text, None
    value = m.group(1).strip()
    first = value.split()[0] if value.split() else ""
    code = match_category(first) if first.lower().startswith("cs.") else None
    if code:  # "category: cs.CR GANs" -> only the code is the category
        rest = text[:m.start()] + value[len(first):] + text[m.end():]
    else:
        code = match_category(value)
        rest = text[:m.start()] + text[m.end():]
    rest = " ".join(rest.replace(" ,", ",").split()).strip(" ,;")
    if code is None:
        return None, rest, f"Unknown category '{value}' ignored (use a code from csrelated.json, e.g. cs.CR)."
    return code, rest, None


# Remove a leading request phrase like 'find me papers about'.
def strip_filler(topic: str) -> str:
    cleaned = FILLER_RE.sub("", topic.strip(), count=1)
    return " ".join(cleaned.split()) or topic.strip()


# Regex-only analysis used when the model call fails.
def regex_fallback(text: str) -> QueryAnalysis:
    authors, topic = [], text
    m = TRAILING_BY_RE.search(topic)
    if m:
        authors = [m.group(1)]
        topic = topic[:m.start()]
    years = YEAR_RE.findall(topic)
    year = int(years[0]) if len(years) == 1 else None
    topic = re.sub(r"\b(?:in|from|since|after|before|until)?\s*\b(?:19|20)\d{2}\b", " ", topic)
    return QueryAnalysis(corrected_query=strip_filler(topic) or text, authors=authors, year=year)


# Drop invented values and check years/topic make sense for an arXiv search.
def validate(text: str, a: QueryAnalysis) -> QueryCheck:
    errors, warnings = [], []

    lowered = text.lower()
    a.authors = [n.strip() for n in a.authors if n.strip() and n.strip().lower() in lowered]
    for name in ("year", "year_from", "year_to"):
        value = getattr(a, name)
        if value is not None and str(value) not in text:
            setattr(a, name, None)
    if a.year is not None:
        a.year_from = a.year_to = None

    if a.year is not None:
        if a.year > CURRENT_YEAR:
            errors.append(f"The year {a.year} hasn't happened yet (current year: {CURRENT_YEAR}).")
        elif a.year < MIN_YEAR:
            errors.append(f"arXiv started in {MIN_YEAR}, so there are no papers from {a.year}.")
    if a.year_from is not None and a.year_from > CURRENT_YEAR:
        errors.append(f"'Since {a.year_from}' is in the future (current year: {CURRENT_YEAR}).")
    if a.year_to is not None and a.year_to < MIN_YEAR:
        errors.append(f"arXiv started in {MIN_YEAR}, so there are no papers before {a.year_to}.")
    if a.year_to is not None and a.year_to > CURRENT_YEAR:
        warnings.append(f"'Until {a.year_to}' is in the future; searching up to {CURRENT_YEAR} instead.")
        a.year_to = None
    if a.year_from is not None and a.year_from < MIN_YEAR:
        warnings.append(f"arXiv started in {MIN_YEAR}; 'since {a.year_from}' has no effect.")
        a.year_from = None
    if a.year_from is not None and a.year_to is not None and a.year_from > a.year_to:
        warnings.append(f"Start year {a.year_from} is after end year {a.year_to}; swapped them.")
        a.year_from, a.year_to = a.year_to, a.year_from

    a.category = None

    topic = strip_filler(a.corrected_query.strip().strip('"'))
    if not topic or len(topic) > len(text) * 1.5:
        topic = regex_fallback(text).corrected_query
    a.corrected_query = topic
    if not topic.strip():
        errors.append("The prompt has no research topic to search for.")

    return QueryCheck(analysis=a, errors=errors, warnings=warnings)


# --------------------------------------------------------------------------- models / chains
# Build (once) the analysis and answer chains; both share one Ollama model and num_ctx.
@functools.lru_cache(maxsize=1)
def get_chains():
    if not OLLAMA_MODEL:
        raise RuntimeError("Set OLLAMA_MODEL in the .env file (run 'ollama list' to see your models).")
    analyzer = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0, reasoning=False,
                          num_ctx=NUM_CTX)
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, reasoning=True, num_ctx=NUM_CTX,
                     temperature=0.6, seed=42)
    analyze_prompt = ChatPromptTemplate.from_messages([("system", ANALYZE_SYSTEM_PROMPT), ("human", "{query}")])
    analyze_chain = (analyze_prompt | analyzer.with_structured_output(QueryAnalysis)).with_fallbacks(
        [RunnableLambda(lambda inputs: regex_fallback(inputs["query"]))])
    answer_chain = ANSWER_PROMPT | llm.with_structured_output(PaperSearchResult, include_raw=True)
    return analyze_chain, answer_chain


# Parse category, run the analyzer and validate; returns (check, prompt without category).
def analyze_query(prompt: str) -> tuple[QueryCheck, str]:
    analyze_chain, _ = get_chains()
    category, rest, category_warning = parse_category(prompt)
    check = validate(rest, analyze_chain.invoke({"query": rest}))
    check.analysis.category = category
    if category_warning:
        check.warnings.append(category_warning)
    return check, rest


# --------------------------------------------------------------------------- search helpers
# True if the hit's title contains 'survey'.
def is_survey(hit) -> bool:
    return "survey" in hit.title.lower()


# Publication year from metadata; 0 if missing.
def year_of(hit) -> int:
    year = hit.metadata.get("year")
    return year if isinstance(year, int) else 0


# Year as text, or 'year unknown'.
def year_label(hit) -> str:
    return str(year_of(hit)) if year_of(hit) else "year unknown"


# Keep in-range, unique model picks, at most limit.
def valid_picks(numbers: list[int], items: list, limit: int) -> list[int]:
    picks = []
    for n in numbers:
        if 1 <= n <= len(items) and n not in picks:
            picks.append(n)
    return picks[:limit]


# Human-readable summary of the active filters.
def describe_filters(a: QueryAnalysis) -> str:
    parts = []
    if a.category:
        parts.append(f"category={a.category} ({CATEGORIES[a.category]})")
    if a.year is not None:
        parts.append(f"year={a.year}")
    if a.year_from is not None or a.year_to is not None:
        parts.append(f"years={a.year_from or MIN_YEAR}-{a.year_to or CURRENT_YEAR}")
    return ", ".join(parts) or "none"


# Convert the analysis into retriever filters (exact year = from == to).
def build_filters(a: QueryAnalysis) -> MetadataIntent:
    year_from = a.year if a.year is not None else a.year_from
    year_to = a.year if a.year is not None else a.year_to
    return MetadataIntent(category=a.category, year_from=year_from, year_to=year_to)


# Topic with the word 'survey(s)' removed.
def topic_without_survey(topic: str) -> str:
    return " ".join(SURVEY_WORD_RE.sub(" ", topic).split()) or topic


# Run the main and survey searches; returns (research papers, surveys).
def search_papers(a: QueryAnalysis, survey_focus: bool, log: LogFn) -> tuple[list, list]:
    filters = build_filters(a)
    log("Loading models and connecting to Qdrant...")
    retriever = build_retriever(qdrant_url=QDRANT_URL, collection_name=COLLECTION)
    try:
        log("Searching Qdrant...")
        main_hits = retriever.retrieve(a.corrected_query, K_RETRIEVE, filters)
        log("Searching Qdrant for surveys...")
        survey_hits = retriever.retrieve(f"survey of {topic_without_survey(a.corrected_query)}",
                                         K_RETRIEVE, filters)
    finally:
        retriever.close()

    papers = [] if survey_focus else [h for h in main_hits if not is_survey(h)][:K_FINAL]
    surveys, seen = [], set()
    for h in main_hits + survey_hits:
        if is_survey(h) and h.point_id not in seen:
            seen.add(h.point_id)
            surveys.append(h)
    return papers, surveys[:K_SURVEYS_FOCUS if survey_focus else K_SURVEYS]


# Format hits as '[n] title\\nabstract' blocks for the prompt.
def numbered(hits: list, prefix: str = "") -> str:
    return "\n\n".join(f"[{prefix}{i}] {h.title}\n{h.text}" for i, h in enumerate(hits, start=1)) or "(none)"


# Ask the model to pick the best papers and fill the response in place.
def rank_papers(response: SearchResponse, topic: str) -> None:
    _, answer_chain = get_chains()
    max_surveys = MAX_SURVEYS_FOCUS if response.survey_focus else MAX_SURVEYS
    try:
        output = answer_chain.invoke({
            "papers": numbered(response.papers), "surveys": numbered(response.surveys, "S"),
            "query": topic, "task": TASK_SURVEYS_ONLY if response.survey_focus else TASK_NORMAL})
    except Exception as exc:
        response.errors.append(f"Ollama request failed: {exc}. Check that Ollama is running and "
                               f"'{OLLAMA_MODEL}' is pulled (ollama pull {OLLAMA_MODEL}).")
        return

    response.reasoning = output["raw"].additional_kwargs.get("reasoning_content", "").strip()
    result: PaperSearchResult | None = output["parsed"]
    if result is None:
        response.errors.append(f"Model did not return valid output: {output['parsing_error']}")
        return

    paper_picks = valid_picks(result.top_papers, response.papers, MAX_PAPERS)
    survey_picks = valid_picks(result.top_surveys, response.surveys, max_surveys)

    if response.survey_focus and survey_picks:
        response.best = (f"S{survey_picks[0]}", response.surveys[survey_picks[0] - 1])
    elif not response.survey_focus and paper_picks:
        response.best = (str(paper_picks[0]), response.papers[paper_picks[0] - 1])

    # Model chooses which papers; display order is newest first
    paper_picks.sort(key=lambda n: year_of(response.papers[n - 1]), reverse=True)
    survey_picks.sort(key=lambda n: year_of(response.surveys[n - 1]), reverse=True)

    response.criteria = result.criteria
    response.why_best = result.why_best
    response.top_papers = paper_picks
    response.top_surveys = survey_picks


# --------------------------------------------------------------------------- pipeline
# Full pipeline for one prompt; GUI entry point (no prints/exits beyond log).
def run_query(prompt: str, log: LogFn = lambda msg: print(msg, flush=True)) -> SearchResponse:
    response = SearchResponse(prompt=prompt)
    try:
        log("Analyzing query...")
        check, rest = analyze_query(prompt)
    except Exception as exc:
        response.errors.append(f"{type(exc).__name__}: {exc}")
        return response

    a = response.analysis = check.analysis
    response.warnings, response.errors = check.warnings, check.errors
    log(f"Search topic: {a.corrected_query} | Filters: {describe_filters(a)}")
    if response.errors:
        return response

    response.survey_focus = bool(SURVEY_WORD_RE.search(rest))
    try:
        response.papers, response.surveys = search_papers(a, response.survey_focus, log)
    except Exception as exc:
        response.errors.append(f"Qdrant search failed: {type(exc).__name__}: {exc}")
        return response

    if not response.surveys and (response.survey_focus or not response.papers):
        kind = "survey papers" if response.survey_focus else "papers"
        response.errors.append(f"No {kind} found for '{a.corrected_query}' with filters: {describe_filters(a)}. "
                               "Try a broader topic, or remove the year/category from your prompt.")
        return response

    log(f"Asking Ollama ({OLLAMA_MODEL})...")
    rank_papers(response, a.corrected_query)
    return response


# One picked paper as '- [label] (year) title\\n   url'.
def format_hit(label: str, hit) -> str:
    return f"- [{label}] ({year_label(hit)}) {hit.title}\n   {hit.url}"


# Render a SearchResponse as plain text.
def format_response(r: SearchResponse, show_reasoning: bool = SHOW_REASONING) -> str:
    lines = []
    if r.analysis and r.analysis.authors:
        lines.append(f"Authors mentioned (not filtered yet): {', '.join(r.analysis.authors)}")
    lines += [f"Warning: {w}" for w in r.warnings]
    if r.errors:
        lines.append("\nCan't complete this query:")
        lines += [f"  - {e}" for e in r.errors]
        return "\n".join(lines)

    if show_reasoning and r.reasoning:
        lines += ["\nModel reasoning:", r.reasoning]
    lines.append("\nEvaluation criteria:")
    lines += [f"  - {c}" for c in r.criteria]

    if not r.survey_focus:
        lines.append("\nThree best matches (newest first):")
        lines += [format_hit(str(n), r.papers[n - 1]) for n in r.top_papers] or ["  (none)"]
    lines.append("\nSurvey papers (newest first):")
    lines += [format_hit(f"S{n}", r.surveys[n - 1]) for n in r.top_surveys] or ["  (no relevant surveys found)"]

    if r.best:
        label, hit = r.best
        lines += [f"\nBest match: [{label}] {hit.title}", f"Why: {r.why_best}"]

    lines.append("\nAll retrieved papers:")
    lines += [f"[{i}] {h.title} (rrf={h.vector_score:.4f}) {h.url}" for i, h in enumerate(r.papers, start=1)]
    lines += [f"[S{i}] {h.title} (rrf={h.vector_score:.4f}) {h.url}" for i, h in enumerate(r.surveys, start=1)]
    return "\n".join(lines)


# CLI entry point: run the prompt (argument or default) and print the result.
def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    response = run_query(" ".join(args) if args else DEFAULT_PROMPT)
    print(format_response(response))
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
