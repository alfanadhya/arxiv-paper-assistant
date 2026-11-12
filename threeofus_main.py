"""RAG over arXiv papers: query analysis -> hybrid Qdrant search -> local Ollama model picks the best papers.

CLI:  python threeofus_main.py "your prompt"     |  python threeofus_main.py --chat  (with follow-ups)
GUI:  chainlit run threeofus_main.py -w   (handlers here, GUI helpers in app.py)
"""

import functools
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

from export import DEFAULT_SELECTION_FILE, selection_file, write_chunks_jsonl
from chatbot_diagnostics import (Diagnostics, format_diagnostics, llm_config,
                                 start_context_tracking, stop_context_tracking)
from prompt_cleaning import (MODE_EXAMPLES, SEARCH_EXAMPLE, ParsedMessage, collapse_repeats,
                             length_problem, parse_message, remove_symbols)   # input checks (steps 1-4)
from retriever_hybrid_rrf import MODELS_CACHE_DIR, Evidence, MetadataIntent, build_retriever, split_payload

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "sebun_bge768_hybrid_full"   # must match the retriever's model
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))   # same for every chain, so Ollama never reloads the model
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")   # keep qwen loaded between questions ("-1" = forever)
SHOW_REASONING = os.getenv("SHOW_REASONING", "0") == "1"

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")   # cross-encoder reranker

K_RETRIEVE = 20          # candidates fetched from Qdrant (per search)
K_FINAL = 6              # (kept) old fixed number of research papers passed to the model
K_SURVEYS = 5            # (kept) old fixed number of surveys passed to the model (normal query)
K_SURVEYS_FOCUS = 10     # (kept) old fixed number of surveys passed to the model (survey query)
SURVEYS_SHOWN = 5        # surveys shown: the reranker's top this many (no LLM ranking for surveys)
MAX_PAPERS = 3           # research papers the model may pick
EXTRA_CANDIDATES = 1     # (kept) old fixed extra; now chosen per query by fit_candidates()
MAX_EXTRA_CANDIDATES = 2 # the LLM sees up to N+2 reranked candidates per pool (N = number it may pick) ...
MIN_EXTRA_CANDIDATES = 1 # ... and at least N+1 when the context is too tight for N+2
RANKING_OUTPUT_RESERVE = int(os.getenv("RANKING_OUTPUT_RESERVE", "3000"))   # tokens kept free for ranking output
CHARS_PER_TOKEN = 3.5    # rough, conservative chars/token estimate for the context budget
MAX_HISTORY_TURNS = 10   # question/answer pairs kept for follow-up questions

# Messages without search= / title= are answered by the agent (run_agent): the LLM calls the search tools
# itself (search, look up a paper, read the papers shown). AGENT=0 in .env sends them to the normal search.
AGENT_ENABLED = os.getenv("AGENT", "1").strip().lower() not in ("0", "false", "no", "off")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "4"))      # tool rounds before the model must answer
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096")) # generated-token cap per step (thinking + answer)
AGENT_DEFAULT_K = 5        # papers one search_papers call returns unless the model asks for more
AGENT_MAX_K = 8            # most papers one call may return
AGENT_RERANK_POOL = 40     # Qdrant hits reranked per call
AGENT_ABSTRACT_CHARS = 700 # abstract length the model sees per paper in search results (get_paper: full)
TITLE_EXACT = 0.95       # title similarity treated as an exact match; below it the user is told
                         # "exact title match not found; best available match"
TITLE_CLOSE = 0.80       # (kept) old cut-off for switching to the reranker
TITLE_BLEND_MIN = 0.60   # from here up to TITLE_EXACT: blend title similarity (mostly) with the reranker;
                         # below it the reranker counts most
TITLE_SIM_WEIGHT = 0.7   # weight of the dominant signal in the blend (other signal gets 1 - this)
K_TITLE = 100            # candidates checked for a title (many titles share phrases like "... is all you need")
K_TITLE_RERANK = 30      # (kept) old reranker pool size
TITLE_POOL_BY_SIM = 10   # rerank pool = these best by title similarity ...
TITLE_POOL_BY_QDRANT = 20   # ... plus Qdrant's top this many (deduplicated, at most 30)
TITLE_AUTHOR_BONUS = 0.03   # soft boosts: a title lookup never filters out a paper by author or year
TITLE_YEAR_BONUS = 0.02     # (arXiv year = first version, often a year before the venue year the user knows)

DB_FIRST_YEAR = 2007                 # oldest paper in the Qdrant collection
DB_LAST_YEAR = 2026                  # newest paper in the Qdrant collection
DB_COVERAGE = "2007 to Oct 2026"     # human-readable coverage for messages

DEFAULT_PROMPT = "search= Generative adversarial networks for cyber security reserach"

# arXiv category codes -> names, e.g. {"cs.CR": "Cryptography and Security"}
CATEGORIES: dict[str, str] = json.loads(
    (Path(__file__).resolve().parent / "csrelated.json").read_text(encoding="utf-8"))

SURVEY_WORD_RE = re.compile(r"\bsurveys?\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
TRAILING_BY_RE = re.compile(r"\bby\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)\s*$")
CATEGORY_RE = re.compile(r"\bcategory\s*:\s*([^,;\n]+)", re.IGNORECASE)
# Authors are metadata ONLY when given with author= (parse_message); 'by ...' is ordinary topic/title text.
AUTHOR_CUE_RE = re.compile(r"\b(?:by|authors?|authored|written)\b", re.IGNORECASE)   # authors need a cue
# Words that refer back to earlier turns; only such prompts are rewritten with the conversation.
FOLLOWUP_CUE_RE = re.compile(
    r"\b(?:it|its|this|that|these|those|them|they|their|one|ones|same|similar|more|other|another|else|"
    r"previous|above|earlier|last|first|second|third|fourth|fifth|sixth|instead|also|too|again|"
    r"what about|how about|like (?:it|that|this|those|these|the)|\[?s?\d\]?)\b",
    re.IGNORECASE)
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
TokenFn = Callable[[str], None]   # receives reply text piece by piece while the model is still writing


# --------------------------------------------------------------------------- schemas
# What the spell-check LLM returns (LangChain PydanticOutputParser): the search target for Qdrant and the
# unclear flag. The model sees the request WITH its year=/author= keys but does not return metadata: the
# parser already extracted it (asking the small model for it made it loop and return an empty topic).
class TopicCheck(BaseModel):
    """Search target for the paper database and whether the request is understandable."""
    # required (no default): a default of "" in the JSON schema made the model return an empty search_query
    search_query: str = Field(
        description="Never empty: what to search the paper database for, i.e. the research topic "
                    "(or paper title) with spelling fixed, WITHOUT the year=/author= keys, their values and "
                    "the words linking to them.")
    unclear: bool = Field(
        description="true only if the request is not understandable even after spelling correction "
                    "(random letters, keyboard mashing). Known acronyms are clear.")
    unclear_reason: str = Field(default="", description="If unclear is true: a few words on why, else ''.")

    @field_validator("unclear", mode="before")
    @classmethod
    def _coerce_unclear(cls, v):
        return v.strip().lower() in ("true", "yes", "1") if isinstance(v, str) else bool(v)

    @field_validator("search_query", "unclear_reason", mode="before")
    @classmethod
    def _coerce_text(cls, v):
        return "" if v is None else str(v)


# Cleaned topic + metadata used as Qdrant filters (built by code: topic from TopicCheck, metadata from the keys).
class QueryAnalysis(BaseModel):
    """Cleaned topic + metadata used as Qdrant filters."""
    corrected_query: str = Field(default="",
        description="REQUIRED, never empty: the research topic with spelling fixed. "
                    "No author names, years or filler words.")
    authors: list[str] = Field(default_factory=list,
        description="Always []. Authors are supplied separately by the application.")
    year: int | None = Field(default=None,
        description="Optional. Exact year only if written, e.g. 'in 2021'; otherwise null.")
    year_from: int | None = Field(default=None,
        description="Optional. Start year only if written, e.g. 'since 2019'; otherwise null.")
    year_to: int | None = Field(default=None,
        description="Optional. End year only if written, e.g. 'before 2020'; otherwise null.")
    category: str | None = Field(default=None, description="Always null.")
    unclear: bool = Field(default=False,
        description="true only if the query is not an understandable research topic even after spelling "
                    "correction (random letters, keyboard mashing, meaningless text). Known acronyms are clear.")
    unclear_reason: str = Field(default="",
        description="If unclear is true: a few words on why. Otherwise an empty string.")

    # Accept 'true'/'false'/'yes'/'no'/None for the unclear flag.
    @field_validator("unclear", mode="before")
    @classmethod
    def _coerce_unclear(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return bool(v)

    # Accept None for the unclear reason.
    @field_validator("unclear_reason", mode="before")
    @classmethod
    def _coerce_reason(cls, v):
        return "" if v is None else str(v)

    # Accept None or a single string for authors (models sometimes return these).
    @field_validator("authors", mode="before")
    @classmethod
    def _coerce_authors(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [p.strip() for p in re.split(r",|;|\band\b", v) if p.strip()]
        return [str(x) for x in v if x is not None and str(x).strip()]

    # Turn '', 'null', 'none', 0 or non-numeric years into None instead of failing validation.
    @field_validator("year", "year_from", "year_to", mode="before")
    @classmethod
    def _coerce_year(cls, v):
        if v is None or isinstance(v, bool):
            return None
        m = YEAR_RE.search(str(v))
        return int(m.group(1)) if m else None

    # Accept a null topic (treated as empty; validate() then falls back).
    @field_validator("corrected_query", mode="before")
    @classmethod
    def _coerce_topic(cls, v):
        return "" if v is None else str(v)


# Structured answer of the follow-up rewrite step.
class RewriteResult(BaseModel):
    """Standalone search query for the user's last message."""
    standalone_query: str = Field(default="",
        description="The user's last message as a standalone search query, one short line, no explanation. "
                    "Unchanged if it is already standalone.")


# Validated analysis; errors skip the search, warnings don't.
class QueryCheck(BaseModel):
    """Validated analysis; errors skip the search, warnings don't."""
    analysis: QueryAnalysis
    errors: list[str] = []
    warnings: list[str] = []
    diagnostics: list[str] = []


# Structured answer the model must return for ONE pool (research papers or surveys: separate calls).
class PoolRanking(BaseModel):
    """Structured answer the model must return."""
    criteria: list[str] = Field(
        description="2-4 short criteria used to judge how well a paper matches the query")
    picks: list[int] = Field(default_factory=list,
        description="Numbers of the best matching papers, best first, e.g. [4, 1, 7] (only the number, "
                    "e.g. 2 for [S2]), no more than the limit given in the instructions. "
                    "Empty list if none are relevant.")
    why_best: str = Field(default="", description="One sentence explaining why your first pick is the best match")


# Everything a GUI needs to display one query's result.
@dataclass
class SearchResponse:
    prompt: str
    mode: str = "search"                                # "search", "title" or "agent" (tool-calling answer)
    answer: str = ""                                    # chat mode: the LLM's reply
    search_prompt: str = ""                             # prompt after follow-up rewrite (= prompt if none)
    ranking_fallback: bool = False                      # True if the model failed and retrieval order was used
    analysis: QueryAnalysis | None = None
    survey_focus: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    papers: list = field(default_factory=list)          # reranked research papers sent to the LLM (N+1)
    surveys: list = field(default_factory=list)         # reranker's top SURVEYS_SHOWN surveys (not sent to the LLM)
    newest_first: bool = False                          # GUI "Newest first" box: sort the shown papers by year
    top_papers: list[int] = field(default_factory=list) # 1-based picks into papers (newest first if newest_first)
    top_surveys: list[int] = field(default_factory=list)# 1-based positions into surveys (newest first if newest_first)
    best: tuple[str, object] | None = None              # (label like "3" / "S2", hit)
    active_mode: str | None = None                      # mode remembered for the chat after this turn (banner)
    request: str = ""                                   # the user's request with its year=/author= keys (LLM context)
    filters: str = "none"                               # metadata filters Qdrant used (describe_filters), LLM context
    memory_note: str = ""                               # agent: papers cited, kept in the conversation memory only
    memory_papers: list[str] = field(default_factory=list)   # agent: arXiv ids of the papers cited
    # agent: one entry per search_papers call (filters, reranked ids + scores, ids returned to the model), for eval
    agent_searches: list[dict] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    why_best: str = ""
    reasoning: str = ""
    diagnostics: Diagnostics = field(default_factory=Diagnostics)  # queries per stage + internal details

    # True when the query produced results without errors.
    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------- prompts
ANALYZE_SYSTEM_PROMPT = """You turn a user's request into the search target for a database of computer science research papers.
You get two lines:
- Full request: the whole request as the user wrote it, for context. It may contain metadata keys:
  year=2021 (exactly 2021), year=2019-2022 (2019 to 2022), year=2019- (hyphen after: from 2019 on),
  year=-2020 (hyphen before: up to 2020), author=Kipf; Welling (one or more authors).
  The application has already read and applied these keys; they are always valid and never your concern.
- Topic text: the part of the request before the keys. Build search_query from this line only.

1. search_query (never empty): the research topic from the Topic text.
   - Fix misspelled words, including technical terms and model names
     (e.g. "transfomers" -> "transformers", "nueral" -> "neural", "BRET" -> "BERT").
   - Keep correctly spelled terms, acronyms and numbers exactly as written (e.g. GAN, BERT, LSTM, GPT-4).
   - Drop a trailing word that only linked to a key ("after", "since", "before", "from", "in", "by").
   - Drop request words such as "find me papers about", "research on", "papers on", "I want", "please".
   - Never copy keys, years or author names from the Full request into search_query.
   - Do not add new words and do not rephrase. A single word is a valid topic.
2. unclear: true ONLY if the Topic text is not an understandable research topic even after spelling
   correction, e.g. random letters ("fsg", "asdf qwe") or keyboard mashing. Then give a short unclear_reason
   and copy the Topic text as search_query. Known acronyms are clear (GAN, BERT, RAG, LLM, CNN, SLAM, NLP).
   Typos, the keys and missing years or authors are never a reason for unclear.

Examples:
Full request: transfomers / Topic text: transfomers
-> search_query: "transformers", unclear: false
Full request: transforemrs papaers after year=2026- / Topic text: transforemrs papaers after
-> search_query: "transformers", unclear: false
Full request: find me papers about GAN for malwre detection by author=Goodfellow / Topic text: find me papers about GAN for malwre detection by
-> search_query: "GAN for malware detection", unclear: false
Full request: graph nueral netwrks for drug discovery year=2019-2022 / Topic text: graph nueral netwrks for drug discovery
-> search_query: "graph neural networks for drug discovery", unclear: false
Full request: fsg / Topic text: fsg
-> search_query: "fsg", unclear: true, unclear_reason: "not a recognizable research topic"
Full request: RAG year=2023 / Topic text: RAG
-> search_query: "RAG", unclear: false"""

ANALYZE_TITLE_SYSTEM_PROMPT = """You turn a user's request into the exact paper TITLE to look up in a database of computer science papers.
You get two lines:
- Full request: the whole request, for context. It may contain metadata keys year= (year=2017, year=2016-2018,
  year=2016-, year=-2018) and author= (author=Vaswani); the application has already applied them.
- Topic text: the title part before the keys. Build search_query from this line only.

1. search_query (never empty): the title from the Topic text with spelling mistakes fixed and NOTHING else
   changed. Keep every word, the word order, colons, hyphens, numbers and acronyms. Do not remove words such as
   "a survey", "towards", "on"; do not shorten, paraphrase or add words. Never copy keys from the Full request.
2. unclear: true only if the title is not understandable even after spelling correction (random letters,
   keyboard mashing). Real titles, even unusual ones, are clear.

Examples:
Full request: atention is all you need year=2017 author=Vaswani / Topic text: atention is all you need
-> search_query: "Attention is all you need", unclear: false
Full request: Malware resistant data protection in hyper-connected networks: a survey / Topic text: Malware resistant data protection in hyper-connected networks: a survey
-> search_query: "Malware resistant data protection in hyper-connected networks: a survey", unclear: false"""

ANALYZE_RETRY_NOTE = ("\n\nIMPORTANT: your previous answer had an empty search_query. search_query is required: "
                      "return the research topic from the request with spelling fixed, even if it is a single word.")

# Ranking runs as one LLM call per pool, so the model only ever compares like with like.
RANK_POOLS = {
    "paper": {"kind": "research papers", "labels": "[1], [2], ...", "example": "[2]", "prefix": "",
              "limit": lambda n: f"{n} best matching research papers"},
    "survey": {"kind": "survey papers", "labels": "[S1], [S2], ...", "example": "[S2]", "prefix": "S",
               "limit": lambda n: f"up to {n} relevant survey papers"},
}

REWRITE_SYSTEM_PROMPT = (
    "You turn the user's last message into a standalone search query for computer science research papers. "
    "Use the conversation only to resolve references such as 'the second one', 'more like that' or 'those authors'. "
    "Keep any years, author names and 'category: ...' parts written in the last message. "
    "If the last message is already a standalone topic, return it unchanged. "
    "Return only the query, on one line, without quotes or explanations.")

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

RANK_POOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are given {kind} numbered {labels} (title and abstract each). "
     "First state the criteria you use to judge how well a paper matches the user's query. "
     "Then apply them: list the numbers of the {limit}, best first (only the number, e.g. 2 for {example}). "
     "Do not explain each paper; only give one sentence on why your first pick is the best match. "
     "Only use papers from the context. Judge by the search target. The filters were applied in the database "
     "search, so every candidate already matches them; the user's request is given for context.\n\n"
     "Candidates:\n{candidates}"),
    ("human", "Search target: {query}\nFilters used in the database search: {filters}\nUser's request: {request}"),
])


# Prompt inputs for ranking one pool (pool = "paper" or "survey"; limit = how many it may pick;
# request = the user's request with its keys, kept in context next to the search target; filters = the
# metadata filters Qdrant actually used, after validation, e.g. "years=2019-2026, authors=Kipf").
def rank_inputs(pool: str, hits: list, topic: str, limit: int, request: str = "", filters: str = "none") -> dict:
    p = RANK_POOLS[pool]
    return {"kind": p["kind"], "labels": p["labels"], "example": p["example"], "limit": p["limit"](limit),
            "candidates": numbered(hits, p["prefix"]), "query": topic, "request": request or topic,
            "filters": filters}


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


# Regex-only analysis used when the model call fails (its year is ignored: metadata comes from year=/author=).
def regex_fallback(text: str) -> QueryAnalysis:
    topic = text
    years = YEAR_RE.findall(topic)
    year = int(years[0]) if len(years) == 1 else None
    topic = re.sub(r"\b(?:in|from|since|after|before|until)?\s*\b(?:19|20)\d{2}\b", " ", topic)
    return QueryAnalysis(corrected_query=strip_filler(topic) or text, authors=[], year=year)


# Drop invented values and check years/topic against the database coverage; details go to diagnostics.
# trusted_meta=True: authors/years came from the user's author=/year= keys, so they are not checked
# against the prompt text (only the range checks against the database coverage apply).
def validate(text: str, a: QueryAnalysis, keep_wording: bool = False, trusted_meta: bool = False) -> QueryCheck:
    errors, warnings, diag = [], [], []
    cover = f"database covers {DB_COVERAGE}"

    # Error: search is skipped; logged with the reason
    def fail(user_msg: str, reason: str) -> None:
        errors.append(user_msg)
        diag.append(f"[validation] FAILED: {reason} ({cover}) -> search skipped.")

    # Warning: value adjusted, search continues; logged with the reason
    def adjust(user_msg: str, reason: str) -> None:
        warnings.append(user_msg)
        diag.append(f"[validation] ADJUSTED: {reason} ({cover}).")

    lowered = text.lower()
    has_cue = bool(AUTHOR_CUE_RE.search(text))
    topic_lower = (a.corrected_query or "").lower()
    kept = []
    for n in (n.strip() for n in a.authors):
        if not n:
            continue
        if trusted_meta:
            kept.append(n)
        elif n.lower() not in lowered:
            diag.append(f"[validation] Dropped author '{n}': not written in the prompt.")
        elif not has_cue:
            diag.append(f"[validation] Dropped author '{n}': no 'by'/'author' in the prompt (metadata is optional).")
        elif n.lower() in topic_lower:
            diag.append(f"[validation] Dropped author '{n}': it is part of the topic.")
        elif n not in kept:
            kept.append(n)
    a.authors = kept
    for name in ("year", "year_from", "year_to"):
        value = getattr(a, name)
        if not trusted_meta and value is not None and str(value) not in text:
            diag.append(f"[validation] Dropped {name}={value}: not written in the prompt.")
            setattr(a, name, None)
    if a.year is not None and (a.year_from is not None or a.year_to is not None):
        diag.append(f"[validation] Exact year {a.year} given; ignored range {a.year_from}-{a.year_to}.")
        a.year_from = a.year_to = None

    if a.year is not None and not DB_FIRST_YEAR <= a.year <= DB_LAST_YEAR:
        fail(f"No papers from {a.year}: the database covers {DB_COVERAGE}.",
             f"year={a.year} is outside {DB_FIRST_YEAR}-{DB_LAST_YEAR}")
    if a.year_from is not None and a.year_from > DB_LAST_YEAR:
        fail(f"No papers since {a.year_from}: the database covers {DB_COVERAGE}.",
             f"year_from={a.year_from} is after {DB_LAST_YEAR}")
    if a.year_to is not None and a.year_to < DB_FIRST_YEAR:
        fail(f"No papers before {a.year_to}: the database covers {DB_COVERAGE}.",
             f"year_to={a.year_to} is before {DB_FIRST_YEAR}")
    if a.year_to is not None and a.year_to > DB_LAST_YEAR:
        adjust(f"'Until {a.year_to}' is past the data; searching up to {DB_LAST_YEAR}.",
               f"year_to={a.year_to} is after {DB_LAST_YEAR}; filter removed")
        a.year_to = None
    if a.year_from is not None and a.year_from < DB_FIRST_YEAR:
        adjust(f"'Since {a.year_from}' is before the data; searching from {DB_FIRST_YEAR}.",
               f"year_from={a.year_from} is before {DB_FIRST_YEAR}; filter removed")
        a.year_from = None
    if a.year_from is not None and a.year_to is not None and a.year_from > a.year_to:
        adjust(f"Start year {a.year_from} is after end year {a.year_to}; swapped them.",
               f"year_from={a.year_from} > year_to={a.year_to}; swapped")
        a.year_from, a.year_to = a.year_to, a.year_from

    a.category = None

    topic = a.corrected_query.strip().strip('"')
    if not keep_wording:            # titles keep every word; topics lose request phrases
        topic = strip_filler(topic)
    if not topic or len(topic) > len(text) * 1.5:
        reason = "empty" if not topic else f"too long ({len(topic)} chars vs prompt {len(text)})"
        topic = regex_fallback(text).corrected_query
        diag.append(f"[validation] Model topic was {reason}; used regex topic '{topic}'.")
    a.corrected_query = topic
    if not topic.strip():
        fail("The prompt has no research topic to search for.", "no topic left after cleaning")
    elif a.unclear:
        # The analyzer could not read a research topic in the query: ask the user to write it properly
        errors.append(f"I couldn't understand '{text.strip()}' as a research topic. Please write the search "
                      f"query properly, e.g. {SEARCH_EXAMPLE}.")
        diag.append(f"[validation] FAILED: query judged unclear by the analyzer "
                    f"({a.unclear_reason or 'no reason given'}) -> search skipped.")
    elif topic.lower() != " ".join(text.lower().split()):
        diag.append(f"[analysis] OK, query cleaned: '{text.strip()}' -> '{topic}' (spelling/filler fixed).")

    return QueryCheck(analysis=a, errors=errors, warnings=warnings, diagnostics=diag)


# --------------------------------------------------------------------------- models / chains
# One Ollama chat model; same model, num_ctx and keep_alive everywhere so Ollama keeps it loaded.
def ollama_llm(**options) -> ChatOllama:
    if not OLLAMA_MODEL:
        raise RuntimeError("Set OLLAMA_MODEL in the .env file (run 'ollama list' to see your models).")
    keep_alive = int(OLLAMA_KEEP_ALIVE) if OLLAMA_KEEP_ALIVE.lstrip("-").isdigit() else OLLAMA_KEEP_ALIVE
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, num_ctx=NUM_CTX, keep_alive=keep_alive, **options)


# Spell-check chain (search and title): prompt with the parser's format instructions -> Ollama held to the
# TopicCheck JSON schema -> LangChain PydanticOutputParser (raises OutputParserException on bad output, which
# run_analyzer turns into the regex fallback). The human message is the request WITH its year=/author= keys.
def spell_check_chain(system_prompt: str):
    parser = PydanticOutputParser(pydantic_object=TopicCheck)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{format_instructions}"),
        ("human", "Full request: {request}\nTopic text: {query}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    llm = ollama_llm(temperature=0, reasoning=False, format=TopicCheck.model_json_schema())
    return prompt | llm | parser


# Build (once) the search spell-check chain and the ranking chain; both share one Ollama model and num_ctx.
@functools.lru_cache(maxsize=1)
def get_chains():
    llm = ollama_llm(reasoning=True, temperature=0.6, seed=42)
    rank_chain = RANK_POOL_PROMPT | llm.with_structured_output(PoolRanking, include_raw=True)   # one pool per call
    return spell_check_chain(ANALYZE_SYSTEM_PROMPT), rank_chain


# Build (once) the spell-check chain used for one retry after an empty search target (stronger instruction).
@functools.lru_cache(maxsize=1)
def get_retry_analyze_chain():
    return spell_check_chain(ANALYZE_SYSTEM_PROMPT + ANALYZE_RETRY_NOTE)


# Build (once) the title spell-check chain: spelling fixed, wording kept, keys dropped.
@functools.lru_cache(maxsize=1)
def get_title_chain():
    return spell_check_chain(ANALYZE_TITLE_SYSTEM_PROMPT)


# Build (once) the follow-up rewrite chain; same model and num_ctx as the analyzer, no reasoning.
@functools.lru_cache(maxsize=1)
def get_rewrite_chain():
    rewriter = ollama_llm(temperature=0, reasoning=False)
    return REWRITE_PROMPT | rewriter.with_structured_output(RewriteResult)


# Remove model thinking from a reply: '<think>...</think>' blocks, or everything up to a lone '</think>'
# (the thinking-only qwen3 template opens <think> itself, so leaked thinking has no opening tag).
def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", text, re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)   # unclosed (cut off) thinking
    return text.strip()


_reranker = None
_reranker_lock = threading.Lock()


# Load (once) the cross-encoder reranker; fp16 on GPU to save memory next to Ollama.
def get_reranker(log: LogFn = lambda msg: None):
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
            log(f"Loading the reranker {RERANKER_MODEL} on {device}...")
            try:
                _reranker = CrossEncoder(RERANKER_MODEL, device=device, cache_folder=str(MODELS_CACHE_DIR),
                                         local_files_only=True, model_kwargs=kwargs, max_length=512)
            except Exception:   # not cached yet: download once
                _reranker = CrossEncoder(RERANKER_MODEL, device=device, cache_folder=str(MODELS_CACHE_DIR),
                                         model_kwargs=kwargs, max_length=512)
        return _reranker


# --------------------------------------------------------------------------- conversation memory
# Append one turn to a history list and keep only the last MAX_HISTORY_TURNS pairs.
def add_turn(history: list[dict], role: str, content: str, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    history.append({"role": role, "content": content})
    del history[:-2 * max_turns]
    return history


# Tags saved with a question/answer pair (by code, no model call), read by select_history in later turns:
#   mode   : "search" / "title" / "agent"
#   ok     : False if the turn failed (error, no results, no answer) -> never read again
#   papers : arXiv ids the turn showed (search/title) or cited (agent)
def memory_tags(r: SearchResponse) -> dict:
    if r.mode == "agent":
        papers = list(r.memory_papers)
    elif r.mode in ("search", "title"):
        papers = [r.papers[n - 1].arxiv_id for n in r.top_papers] + [r.surveys[n - 1].arxiv_id for n in r.top_surveys]
    else:
        papers = []
    ok = not r.errors and (bool(r.answer) if r.mode == "agent" else bool(papers))
    return {"mode": r.mode, "ok": ok, "papers": papers}


# Earlier turns to put in the model's context, without their tags (LangChain format). Failed turns are
# skipped. With `paper_ids` (an analyze= call): only turns about at least one of those papers, plus the latest
# search/title result (so the model knows which list the labels come from). Turns saved before tags existed
# count as ok and relevant. Returns (history, [memory] diagnostic line).
def select_history(history: list[dict] | None, paper_ids: set[str] | None = None) -> tuple[list[dict], str]:
    entries = history or []
    pairs = [entries[i:i + 2] for i in range(0, len(entries) - 1, 2)]
    tags_of = lambda pair: pair[-1].get("tags") or {}
    last_result = max((i for i, p in enumerate(pairs) if tags_of(p).get("mode") in ("search", "title")
                       and tags_of(p).get("ok", True)), default=None)
    kept, failed, other = [], 0, 0
    for i, pair in enumerate(pairs):
        tags = tags_of(pair)
        if not tags.get("ok", True):
            failed += 1
            continue
        known = tags.get("papers")
        if paper_ids is not None and i != last_result and known is not None and not set(known) & paper_ids:
            other += 1
            continue
        kept.append(pair)
    note = (f"[memory] Context reads {len(kept)} of {len(pairs)} earlier turn(s)"
            + (f"; skipped {failed} failed" if failed else "")
            + (f"; skipped {other} about other papers" if other else "") + ".")
    return [{"role": e["role"], "content": e["content"]} for pair in kept for e in pair], note


# Assistant turn text for the history: summary line plus numbered picks with arXiv ids.
def assistant_turn_text(r: SearchResponse) -> str:
    if r.mode == "agent":   # the papers cited are remembered here (not shown in the answer)
        body = r.answer[:2000] if r.answer else "No answer: " + " ".join(r.errors)
        return f"{r.memory_note}\n{body}" if r.memory_note else body
    if r.errors:
        return "No results: " + " ".join(r.errors)
    topic = r.analysis.corrected_query if r.analysis else r.search_prompt
    lines = [f"Papers for '{topic}':"]
    lines += [f"[{n}] [{r.papers[n - 1].arxiv_id}] {r.papers[n - 1].title}" for n in r.top_papers]
    lines += [f"[S{n}] [{r.surveys[n - 1].arxiv_id}] {r.surveys[n - 1].title}" for n in r.top_surveys]
    return "\n".join(lines)


# True if the prompt refers back to earlier turns ("the second one", "more like that", "same authors", ...).
def looks_like_followup(prompt: str) -> bool:
    return bool(FOLLOWUP_CUE_RE.search(prompt))


# Strip thinking blocks, labels and quotes a model may wrap around the rewritten query.
def clean_rewrite(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", raw or "", flags=re.DOTALL | re.IGNORECASE)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    query = lines[-1] if lines else ""
    query = re.sub(r"^(?:standalone\s+)?(?:search\s+)?query\s*:\s*", "", query, flags=re.IGNORECASE)
    return query.strip().strip('"').strip("'").strip()


# Rewrite a follow-up into a standalone query using the history; returns (query, diagnostic or None).
def rewrite_with_history(prompt: str, history: list[dict] | None) -> tuple[str, str | None]:
    if not history:
        return prompt, None
    if not looks_like_followup(prompt):
        return prompt, None   # standalone topic: no model call, nothing to report
    try:
        result = get_rewrite_chain().invoke({"history": history, "question": prompt},
                                            config=llm_config("follow-up rewrite"))
        raw = result.standalone_query if result is not None else ""
    except Exception as exc:
        return prompt, f"[memory] Follow-up rewrite failed ({type(exc).__name__}: {exc}); used the prompt as typed."
    query = clean_rewrite(raw)
    if not query or len(query) > 3 * len(prompt) + 200:
        preview = " ".join((raw or "").split())[:80]
        return prompt, (f"[memory] Follow-up rewrite unusable ({len(query)} chars, model returned: '{preview}...'); "
                        "used the prompt as typed.")
    if query == prompt:
        return prompt, None
    return query, f"[memory] Rewrote follow-up '{prompt}' -> '{query}' using {len(history) // 2} earlier turn(s)."


# Run the analyzer (one retry if the topic is empty, regex fallback on failure); returns (analysis, notes).
# title_mode=True uses the title prompt: spelling fixed, wording kept exactly.
# request = what the model sees: the text WITH its year=/author= keys (metadata is extracted by the parser but
# kept in the request); text = the topic/title alone, used by the regex fallback.
def run_analyzer(text: str, title_mode: bool = False, request: str | None = None) -> tuple[QueryAnalysis, list[str]]:
    if not text.strip():
        return regex_fallback(text), []
    request = request or text
    analyze_chain = get_title_chain() if title_mode else get_chains()[0]
    notes = []
    try:
        inputs = {"request": request, "query": text}
        check: TopicCheck = analyze_chain.invoke(inputs, config=llm_config("analysis"))
        if not check.search_query.strip():
            notes.append("[analysis] Model returned an empty search target; retrying once.")
            retry = get_retry_analyze_chain().invoke(inputs, config=llm_config("analysis retry"))
            if retry.search_query.strip():
                notes.append(f"[analysis] Retry gave search target '{retry.search_query}'.")
                check = retry
            else:
                notes.append("[analysis] Retry search target also empty.")
        notes.append(f"[analysis] Spell check: request '{request}' -> search target '{check.search_query}'.")
    except Exception as exc:   # OutputParserException (bad JSON / schema) or Ollama errors
        notes.append(f"[analysis] Model failed ({type(exc).__name__}: {str(exc)[:200]}); used regex fallback "
                     "(topic as typed, no spelling fix).")
        return regex_fallback(text), notes
    # Metadata stays empty here: analyze_query fills it from the year= / author= keys
    return QueryAnalysis(corrected_query=check.search_query, unclear=check.unclear,
                         unclear_reason=check.unclear_reason), notes


# The author= names (and the question words around them) removed from a search topic before the spell-check
# LLM sees it: the author is already a Qdrant filter, and a name in the topic only adds noise to the embeddings
# and the reranker (eval m07: 'what has Iryna Gurevych published on misleading charts?' became the target
# 'Iryna Gurev: misleading charts'). The full name is removed anywhere; the last name alone only next to a cue
# ('by Gurevych', "Gurevych's") so a topic like 'Turing test' survives author=Alan Turing.
# Returns (topic, diagnostic or None); the text is kept as it was if nothing would be left.
def strip_author_names(text: str, authors: list[str]) -> tuple[str, str | None]:
    original, flags = text, re.IGNORECASE
    for name in authors:
        if "," in name:                                       # 'Gurevych, Iryna' -> 'Iryna Gurevych'
            last, first = name.split(",", 1)
            name = f"{first.strip()} {last.strip()}"
        tokens = name.split()
        if not tokens:
            continue
        full = r"\s+".join(re.escape(t) for t in tokens)
        variants = [(full, False)]
        if len(tokens) > 1 and len(tokens[-1]) >= 3:
            variants.append((re.escape(tokens[-1]), True))    # last name alone: only with a cue
        for pattern, needs_cue in variants:
            named = rf"{pattern}(?:['’]s)?"
            text = re.sub(rf"\bwhat\s+(?:has|have|did|does|do)\s+{named}\s+(?:published|publish|written|write|"
                          rf"wrote|done|worked)\s*(?:on|about|in|regarding)?(?!\w)", " ", text, flags=flags)
            text = re.sub(rf"\b(?:papers?|works?|research|publications?|articles?)\s+(?:by|from|of)\s+{named}(?!\w)",
                          " ", text, flags=flags)
            text = re.sub(rf"\b(?:(?:authored|written)\s+by|by|from)\s+{named}(?!\w)", " ", text, flags=flags)
            text = re.sub(rf"\b{pattern}['’]s(?!\w)", " ", text, flags=flags)
            if not needs_cue:
                text = re.sub(rf"\b{pattern}(?!\w)", " ", text, flags=flags)
    text = " ".join(text.split()).strip(" ,;:?.!-")
    text = re.sub(r"^(?:on|about|regarding|in)\s+", "", text, flags=flags)
    if text == " ".join(original.split()):
        return original, None
    if not re.search(r"[^\W_]", text):
        return original, "[validation] Author names not removed from the topic: nothing would be left."
    return text, f"[validation] Removed the author name(s) from the topic (author= is a filter): '{original}' -> '{text}'."


# Distinct author names on the hits whose last name matched one of `authors` (after the last-name-only retry),
# e.g. ['Boris Gurevych'], so the user sees who was actually matched.
def last_name_matches(hits: list, authors: list[str], limit: int = 5) -> list[str]:
    lasts = set()
    for n in authors:
        c = author_condition(n, use_initials=False)
        if c:
            lasts.update(c[1])
    names: list[str] = []
    for h in hits:
        for raw in hit_authors(h):
            last = raw.split(",", 1)[0] if "," in raw else raw.split()[-1] if raw.split() else ""
            shown = " ".join(p.strip() for p in reversed(raw.split(",", 1))) if "," in raw else raw
            if normalize_author(last) in lasts and shown not in names:
                names.append(shown)
                if len(names) >= limit:
                    return names
    return names


# Run the analyzer on the search=/title= text (spelling fix + clarity check only) and validate; returns
# (check, text without the 'category:' marker). Years and authors come ONLY from the parsed year=/author=
# keys (meta); anything the model guesses is ignored and noted.
def analyze_query(prompt: str, title_mode: bool = False,
                  meta: ParsedMessage | None = None) -> tuple[QueryCheck, str]:
    meta = meta or ParsedMessage(mode="title" if title_mode else "search", text=prompt)
    category, rest, category_warning = parse_category(prompt)
    strip_note = None
    if meta.authors and not title_mode:   # title lookups keep the title exactly as typed
        rest, strip_note = strip_author_names(rest, meta.authors)
    analysis, notes = run_analyzer(rest, title_mode, request=meta.with_meta(rest))
    if strip_note:
        notes.insert(0, strip_note)
    analysis.authors = list(meta.authors)
    analysis.year, analysis.year_from, analysis.year_to = meta.year, meta.year_from, meta.year_to
    check = validate(rest, analysis, keep_wording=title_mode, trusted_meta=True)
    check.diagnostics[:0] = notes
    if meta.authors:
        check.diagnostics.append(f"[validation] Authors from author=: {', '.join(meta.authors)}.")
    if any(v is not None for v in (meta.year, meta.year_from, meta.year_to)):
        check.diagnostics.append(f"[validation] Years from year=: year={meta.year}, from={meta.year_from}, "
                                 f"to={meta.year_to} -> used year={check.analysis.year}, "
                                 f"from={check.analysis.year_from}, to={check.analysis.year_to}.")
    check.analysis.category = category
    if category_warning:
        check.warnings.append(category_warning)
        check.diagnostics.append(f"[validation] ADJUSTED: {category_warning}")
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
        parts.append(f"years={a.year_from or DB_FIRST_YEAR}-{a.year_to or DB_LAST_YEAR}")
    if a.authors:
        parts.append("authors=" + " & ".join(a.authors))
    return ", ".join(parts) or "none"


# Name as stored in metadata.author_last_names / author_keys: 'Müller' -> 'muller', "O'Brien" -> 'obrien'.
# Mirrors normalize_name() in utils/db_generator_hybrid_gpu.py (minus LaTeX handling, not needed for typed input).
def normalize_author(value: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", value or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    return " ".join(s.split()).strip("-")


# One author -> Qdrant condition: ('author_keys', last_initial variants) when a first name/initial is given,
# else ('author_last_names', last-name variants). Variants cover particles: 'van der Maaten' -> 'maaten',
# 'der maaten', 'van der maaten'. use_initials=False gives the looser last-name-only condition.
def author_condition(name: str, use_initials: bool = True) -> tuple[str, tuple[str, ...]] | None:
    if "," in name:                                   # "Chen, Mark" -> "Mark Chen"
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    tokens = normalize_author(name).split()
    if not tokens:
        return None
    if len(tokens) == 1:
        return ("author_last_names", (tokens[0],))
    lasts = tuple(" ".join(tokens[i:]) for i in range(1, len(tokens)))
    if use_initials:
        return ("author_keys", tuple(f"{last}_{tokens[0][0]}" for last in lasts))
    return ("author_last_names", lasts)


# Convert the analysis into retriever filters (exact year = from == to; every named author must match).
def build_filters(a: QueryAnalysis, use_initials: bool = True) -> MetadataIntent:
    year_from = a.year if a.year is not None else a.year_from
    year_to = a.year if a.year is not None else a.year_to
    authors = tuple(c for c in (author_condition(n, use_initials) for n in a.authors) if c)
    return MetadataIntent(category=a.category, year_from=year_from, year_to=year_to, authors=authors)


# Topic with the word 'survey(s)' removed.
def topic_without_survey(topic: str) -> str:
    return " ".join(SURVEY_WORD_RE.sub(" ", topic).split()) or topic


_retriever = None
_retriever_lock = threading.Lock()


# Build (once) the hybrid retriever for our Qdrant collection; setup problems become RuntimeError.
def get_retriever(log: LogFn = lambda msg: None):
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            log(f"Loading the embedding models and connecting to Qdrant ({QDRANT_URL}, '{COLLECTION}')...")
            try:
                _retriever = build_retriever(qdrant_url=QDRANT_URL, collection_name=COLLECTION)
            except SystemExit as exc:   # retriever_hybrid_rrf reports setup errors this way
                raise RuntimeError(str(exc)) from None
        return _retriever


# Load the retriever and Ollama chains up front; returns the number of papers in the collection.
def warmup(log: LogFn = lambda msg: None) -> int | None:
    retriever = get_retriever(log)
    get_chains()
    get_retry_analyze_chain()
    get_rewrite_chain()
    get_reranker(log)
    return retriever.points_count


# Cheap start-of-chat check: load/reconnect only what is missing. Models already in memory are reused;
# Qdrant is pinged (client rebuilt only if the ping fails); qwen is warmed up only if Ollama hasn't got it
# loaded. Returns one status line per component (for the "Ready" message); raises if Qdrant is unreachable.
def health_check(log: LogFn = lambda msg: None) -> list[str]:
    import time
    import httpx
    status = []

    # Embedding models + Qdrant connection (models are loaded once per server process)
    t = time.time()
    loaded_before = _retriever is not None
    retriever = get_retriever(log)
    try:
        info = retriever.client.get_collection(COLLECTION)            # ping, a few ms
        ping = "ok"
    except Exception as exc:                                            # e.g. Qdrant restarted: reconnect
        log(f"Qdrant did not answer ({type(exc).__name__}); reconnecting...")
        from qdrant_client import QdrantClient
        retriever.client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
        info = retriever.client.get_collection(COLLECTION)            # raises if still unreachable
        ping = "reconnected"
    count = getattr(info, "points_count", None) or retriever.points_count
    status.append(f"Qdrant `{COLLECTION}`: {ping}, {count:,} papers ({DB_COVERAGE})" if count else
                  f"Qdrant `{COLLECTION}`: {ping}")
    status.append("Embedding models: " + ("already loaded" if loaded_before else
                                          f"loaded now ({time.time() - t:.1f} s)"))

    # Cross-encoder reranker (loaded once per server process)
    t = time.time()
    rerank_before = _reranker is not None
    get_reranker(log)
    status.append(f"Reranker `{RERANKER_MODEL}`: " + ("already loaded" if rerank_before else
                                                      f"loaded now ({time.time() - t:.1f} s)"))

    # LangChain chains: cheap objects, cached after the first call
    get_chains(), get_retry_analyze_chain(), get_rewrite_chain(), get_title_chain()

    # Ollama: is qwen loaded (with our context size)? If not, an empty request loads it.
    try:
        ps = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=5).json().get("models", [])
        loaded = next((m for m in ps if m.get("name") == OLLAMA_MODEL or m.get("model") == OLLAMA_MODEL), None)
        if loaded and loaded.get("context_length") in (None, NUM_CTX):
            gpu = loaded.get("size_vram", 0) / max(loaded.get("size", 1), 1)
            status.append(f"Ollama `{OLLAMA_MODEL}`: already loaded ({gpu:.0%} GPU, context "
                          f"{loaded.get('context_length') or NUM_CTX:,})")
        else:
            log(f"Loading {OLLAMA_MODEL} into Ollama...")
            t = time.time()
            keep_alive = int(OLLAMA_KEEP_ALIVE) if OLLAMA_KEEP_ALIVE.lstrip("-").isdigit() else OLLAMA_KEEP_ALIVE
            httpx.post(f"{OLLAMA_URL}/api/generate", timeout=300,
                       json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": keep_alive,
                             "options": {"num_ctx": NUM_CTX}}).raise_for_status()
            reason = "context size differed" if loaded else "was not loaded"
            status.append(f"Ollama `{OLLAMA_MODEL}`: loaded now ({time.time() - t:.1f} s; {reason})")
    except Exception as exc:
        status.append(f"Ollama `{OLLAMA_MODEL}`: NOT available ({type(exc).__name__}: {exc}); searches will "
                      "fall back to reranker order")
    return status


# Run the main and survey searches; returns (paper pool, survey pool, origin) in Qdrant order, uncut.
# origin maps chunk id -> (search name, rank in that search) for the rank trace.
def search_papers(a: QueryAnalysis, survey_focus: bool, log: LogFn,
                  use_initials: bool = True) -> tuple[list, list, dict]:
    filters = build_filters(a, use_initials)
    retriever = get_retriever(log)
    with _retriever_lock:   # one search at a time on the shared models
        log("Searching Qdrant...")
        main_hits = retriever.retrieve(a.corrected_query, K_RETRIEVE, filters)
        log("Searching Qdrant for surveys...")
        survey_hits = retriever.retrieve(f"survey of {topic_without_survey(a.corrected_query)}",
                                         K_RETRIEVE, filters)

    origin: dict[str, tuple[str, int]] = {}
    for name, hits in (("topic", main_hits), ("survey", survey_hits)):
        for i, h in enumerate(hits, start=1):
            origin.setdefault(h.point_id, (name, i))

    papers = [] if survey_focus else [h for h in main_hits if not is_survey(h)]
    surveys, seen = [], set()
    for h in main_hits + survey_hits:
        if is_survey(h) and h.point_id not in seen:
            seen.add(h.point_id)
            surveys.append(h)
    return papers, surveys, origin


# Re-order hits with the cross-encoder (query, title + abstract); returns (hits best first, scores).
def rerank_hits(query: str, hits: list) -> tuple[list, list[float]]:
    if len(hits) < 2:
        return list(hits), [0.0] * len(hits)
    reranker = get_reranker()
    with _reranker_lock:
        scores = reranker.predict([(query, h.text) for h in hits], batch_size=32, show_progress_bar=False)
    order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)
    return [hits[i] for i in order], [float(scores[i]) for i in order]


# Rerank one pool, record ranks, and keep the top `keep` for the LLM; falls back to Qdrant order on error.
def rerank_pool(response: SearchResponse, query: str, pool: list, pool_name: str, keep: int) -> list:
    ranked, scores = rerank_full(response, query, pool, pool_name)
    if scores is not None:
        response.diagnostics.note_rerank(pool_name, pool, ranked, scores, keep, RERANKER_MODEL)
    return ranked[:keep]


# Rerank a whole pool; returns (hits best first, scores), or (Qdrant order, None) if the reranker fails.
def rerank_full(response: SearchResponse, query: str, pool: list, pool_name: str) -> tuple[list, list | None]:
    if not pool:
        return [], None
    try:
        return rerank_hits(query, pool)
    except Exception as exc:
        response.diagnostics.append(f"[rerank] FAILED for {pool_name}s ({type(exc).__name__}: {exc}); "
                                    "kept Qdrant order.")
        return list(pool), None


# Rough token count of text (no tokenizer call; ~CHARS_PER_TOKEN characters per token, on the safe side).
def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


# Estimated tokens of the ranking prompt for these candidates (system text + papers + surveys + query).
def ranking_prompt_tokens(pool: str, hits: list, topic: str, limit: int) -> int:
    messages = RANK_POOL_PROMPT.format_messages(**rank_inputs(pool, hits, topic, limit))
    return sum(estimate_tokens(str(m.content)) for m in messages)


# Pick how many reranked candidates of ONE pool its ranking call sees: N+2 if that prompt plus
# RANKING_OUTPUT_RESERVE fits in NUM_CTX, else N+1. Returns (hits to send, note part, fits).
def fit_pool(pool: str, hits: list, n: int, topic: str) -> tuple[list, str, bool]:
    budget = NUM_CTX - RANKING_OUTPUT_RESERVE
    for extra in range(MAX_EXTRA_CANDIDATES, MIN_EXTRA_CANDIDATES - 1, -1):
        sent = hits[:n + extra]
        est = ranking_prompt_tokens(pool, sent, topic, n)
        if est <= budget:
            return sent, f"N+{extra} {pool}s ({len(sent)}, prompt ~{est:,} tokens)", True
    return sent, f"N+{MIN_EXTRA_CANDIDATES} {pool}s ({len(sent)}, prompt ~{est:,} tokens, TOO BIG)", False


# Research papers only (surveys skip the LLM): how many reranked papers the ranking call sees. Returns
# (papers, note).
def fit_candidates(papers: list, n_papers: int, topic: str) -> tuple[list, str]:
    if not papers:
        return [], "[rerank] No research papers to send to the LLM."
    p, p_note, p_fits = fit_pool("paper", papers, n_papers, topic)
    if p_fits:
        return p, (f"[rerank] Sending {p_note} to the LLM; prompt + {RANKING_OUTPUT_RESERVE:,} reserved for its "
                   f"output fits {NUM_CTX:,}.")
    return p, (f"[rerank] WARNING: {p_note}: even N+{MIN_EXTRA_CANDIDATES} + {RANKING_OUTPUT_RESERVE:,} reserved "
               f"> {NUM_CTX:,} tokens; sending anyway.")


# Record the picks in the rank trace: the papers' LLM order ("llm" or "fallback") and the displayed order;
# surveys are marked as skipped by the LLM.
def record_picks(response: SearchResponse, paper_order: list[int], paper_by: str) -> None:
    response.diagnostics.record_picks(response.papers, response.surveys, paper_order,
                                      response.top_papers, response.top_surveys, paper_by)


# Format hits as '[n] title\\nabstract' blocks for the prompt.
# The stored text is 'title\n\nabstract', so only its abstract part follows the '[n] title' line (no repeated
# title); a record with only a title or only an abstract keeps what it has.
def numbered(hits: list, prefix: str = "") -> str:
    return "\n\n".join(f"[{prefix}{i}] {h.title}\n{h.text.split(chr(10) * 2, 1)[-1]}"
                       for i, h in enumerate(hits, start=1)) or "(none)"


# Survey display order (no LLM ranking): newest first if the GUI box asks for it (same year keeps the
# reranker order), else the reranker order. Returns 1-based positions into surveys.
def order_surveys(surveys: list, newest_first: bool) -> list[int]:
    order = list(range(1, len(surveys) + 1))
    return sorted(order, key=lambda n: year_of(surveys[n - 1]), reverse=True) if newest_first else order


# Papers in display order: newest first only if the GUI box asks for it, else the ranking order.
def order_papers(picks: list[int], papers: list, newest_first: bool) -> list[int]:
    return sorted(picks, key=lambda n: year_of(papers[n - 1]), reverse=True) if newest_first else list(picks)


# Result of ranking one pool (research papers or surveys).
@dataclass
class PoolOutcome:
    picks: list[int] = field(default_factory=list)   # 1-based picks into the pool, in the model's order
    picked_by: str = "llm"                            # "llm", or "fallback" (reranker order)
    criteria: list[str] = field(default_factory=list)
    why_best: str = ""
    reasoning: str = ""


# Rank ONE pool with its own LLM call; on failure that pool alone falls back to the reranker order.
def rank_pool(response: SearchResponse, pool: str, hits: list, topic: str, limit: int,
              log: LogFn, found: int) -> PoolOutcome:
    if not hits:
        return PoolOutcome()
    diag, name = response.diagnostics, f"{pool}s"

    def fallback(reason: str) -> PoolOutcome:
        diag.append(f"[ranking] FALLBACK ({name}): {reason} -> top {name} in reranker order.")
        return PoolOutcome(picks=list(range(1, min(limit, len(hits)) + 1)), picked_by="fallback")

    log(f"Asking Ollama ({OLLAMA_MODEL}) to pick the best {name}: top {len(hits)} of {found} found...")
    _, rank_chain = get_chains()
    try:
        output = rank_chain.invoke(rank_inputs(pool, hits, topic, limit, response.request, response.filters),
                                   config=llm_config(f"ranking {name}"))
    except Exception as exc:
        return fallback(f"Ollama request failed ({type(exc).__name__}: {exc}); check that Ollama is running "
                        f"and '{OLLAMA_MODEL}' is pulled (ollama pull {OLLAMA_MODEL})")
    result: PoolRanking | None = output["parsed"]
    if result is None:
        return fallback(f"model did not return valid output: {output['parsing_error']}")

    picks = valid_picks(list(result.picks), hits, limit)
    if picks != list(result.picks):
        diag.append(f"[ranking] {name.capitalize()}: model picks cleaned {list(result.picks)} -> {picks} "
                    "(out of range, duplicate or over limit).")
    if not picks:
        return fallback("model picked none of the candidates")
    prefix = RANK_POOLS[pool]["prefix"]
    diag.append(f"[ranking] {name.capitalize()} (own LLM call): picked "
                f"{', '.join(f'[{prefix}{n}]' for n in picks)} of {len(hits)} candidates"
                + (f"; criteria: {'; '.join(result.criteria)}" if result.criteria else "")
                + (f"; why [{prefix}{picks[0]}]: {result.why_best}" if result.why_best else ""))
    return PoolOutcome(picks=picks, criteria=list(result.criteria), why_best=result.why_best,
                       reasoning=output["raw"].additional_kwargs.get("reasoning_content", "").strip())


# Ask the model to pick the best research papers; surveys are not sent to the LLM (reranker top
# SURVEYS_SHOWN, newest first only with the box). Fills the response in place.
def rank_papers(response: SearchResponse, topic: str, log: LogFn = lambda msg: None,
                found: tuple[int, int] = (0, 0)) -> None:
    p = rank_pool(response, "paper", response.papers, topic, MAX_PAPERS, log, found[0])
    response.top_papers = order_papers(p.picks, response.papers, response.newest_first)
    response.top_surveys = order_surveys(response.surveys, response.newest_first)
    if response.surveys:
        response.diagnostics.append(
            f"[ranking] Surveys: LLM ranking skipped; reranker top {len(response.surveys)} of {found[1]} shown "
            + ("newest first" if response.newest_first else "in reranker order") + ": " + ", ".join(f"[S{n}] {year_of(response.surveys[n - 1]) or '----'}"
                                         for n in response.top_surveys) + ".")

    # Best/criteria/why: the LLM's best paper; for a survey query (no papers) the reranker's top survey
    if response.survey_focus:
        if response.surveys:
            response.best = ("S1", response.surveys[0])
            response.why_best = "Highest reranker score among the surveys (surveys are not ranked by the LLM)."
    else:
        if p.picks:
            response.best = (f"{p.picks[0]}", response.papers[p.picks[0] - 1])
        response.criteria = p.criteria
        response.why_best = p.why_best or (
            "Highest reranker score (the model was unavailable to judge the papers)." if p.picked_by == "fallback"
            else "")
    response.reasoning = p.reasoning
    record_picks(response, p.picks, p.picked_by)

    if response.papers and p.picked_by == "fallback":
        response.ranking_fallback = True
        response.warnings.append("The model could not judge the research papers, so those are the closest by "
                                 "search score.")


# --------------------------------------------------------------------------- pipeline
# Full pipeline for one prompt; GUI entry point (no prints/exits beyond log).
def run_query(prompt: str, log: LogFn = lambda msg: print(msg, flush=True),
              history: list[dict] | None = None, newest_first: bool = False,
              meta: ParsedMessage | None = None) -> SearchResponse:
    response = SearchResponse(prompt=prompt, search_prompt=prompt, newest_first=newest_first)
    response.diagnostics.original_prompt = prompt
    start_context_tracking(response.diagnostics)   # LLM calls below report their token usage
    try:
        if history:
            log("Reading the conversation...")
            response.search_prompt, note = rewrite_with_history(prompt, history)
            if note:
                response.diagnostics.append(note)
            if response.search_prompt != prompt:
                response.diagnostics.rewritten_prompt = response.search_prompt
        # the request as the user gave it (keys included) stays in the LLM context next to the search target
        response.request = meta.with_meta(response.search_prompt) if meta else response.search_prompt
        log("Analyzing query...")
        check, rest = analyze_query(response.search_prompt, meta=meta)
    except Exception as exc:
        response.errors.append(f"{type(exc).__name__}: {exc}")
        return response

    a = response.analysis = check.analysis
    response.warnings, response.errors = check.warnings, check.errors
    response.diagnostics += check.diagnostics
    response.diagnostics.modified_query = a.corrected_query or None
    log(f"Search topic: {a.corrected_query} | Filters: {describe_filters(a)}")
    if response.errors:
        return response

    response.survey_focus = bool(SURVEY_WORD_RE.search(rest))
    if a.authors:
        response.diagnostics.append(
            "[validation] Author filter: " + "; ".join(
                f"{n} -> {c[0]} in {list(c[1])}" for n, c in
                ((n, author_condition(n)) for n in a.authors) if c) + ".")
    try:
        paper_pool, survey_pool, origin = search_papers(a, response.survey_focus, log)
        used_initials = any(author_condition(n) and author_condition(n)[0] == "author_keys" for n in a.authors)
        if not paper_pool and not survey_pool and used_initials:
            # The first-name initial may differ in the stored record: retry with last names only
            response.diagnostics.append("[validation] ADJUSTED: no papers for the author(s) with that first "
                                        "initial; retried with last name only.")
            paper_pool, survey_pool, origin = search_papers(a, response.survey_focus, log, use_initials=False)
            used_initials = False
            matched = last_name_matches(paper_pool + survey_pool, a.authors)
            if matched:   # tell the user who was actually matched: may be a different person with that last name
                response.warnings.append(
                    f"No papers by {' & '.join(a.authors)} with that first name; showing papers by authors with "
                    f"the same last name: {', '.join(matched)}. These may be different people.")
                response.diagnostics.append(f"[validation] Last-name-only matches: {', '.join(matched)}.")
    except Exception as exc:
        response.errors.append(f"Qdrant search failed: {type(exc).__name__}: {exc}")
        return response
    # The filters exactly as Qdrant used them (after validation and the last-name retry): ranking LLM context
    response.filters = describe_filters(a) + (" (authors matched by last name only)"
                                              if a.authors and not used_initials else "")

    if not survey_pool and (response.survey_focus or not paper_pool):
        kind = "survey papers" if response.survey_focus else "papers"
        response.errors.append(f"No {kind} found for '{a.corrected_query}' with filters: {describe_filters(a)}. "
                               "Try a broader topic, or remove the year/category from your prompt.")
        return response

    # Rerank each pool with the cross-encoder; the LLM sees N+2 papers if the context allows, else N+1.
    # Surveys are not sent to the LLM: the reranker's top SURVEYS_SHOWN are shown (newest first only with the box).
    diag = response.diagnostics
    diag.track_pool("paper", paper_pool, origin)
    diag.track_pool("survey", survey_pool, origin)
    log("Reranking candidates...")
    papers_ranked, paper_scores = rerank_full(response, a.corrected_query, paper_pool, "paper")
    surveys_ranked, survey_scores = rerank_full(response, a.corrected_query, survey_pool, "survey")
    response.papers, fit_note = fit_candidates(papers_ranked, MAX_PAPERS, a.corrected_query)
    response.surveys = surveys_ranked[:SURVEYS_SHOWN]
    if paper_scores is not None:
        diag.note_rerank("paper", paper_pool, papers_ranked, paper_scores, len(response.papers), RERANKER_MODEL)
    if survey_scores is not None:
        diag.note_rerank("survey", survey_pool, surveys_ranked, survey_scores, len(response.surveys), RERANKER_MODEL,
                         to_llm=False)
    diag.append(fit_note)
    diag.track_sent(response.papers)

    # One LLM call for the papers; status e.g. "Asking Ollama (qwen3:4b) to pick the best papers: top 5 of 17 found..."
    rank_papers(response, a.corrected_query, log, (len(paper_pool), len(survey_pool)))
    return response


# --------------------------------------------------------------------------- title lookup (title=)
# Title text for comparison: lowercase, LaTeX/punctuation removed, single spaces.
def normalize_title(text: str) -> str:
    text = re.sub(r"\$[^$]*\$|\\[A-Za-z]+|[{}]", " ", text or "")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


# Similarity of two titles in [0, 1]: mean of a character-level ratio (tolerates typos) and a word-level F1
# (penalises different words, e.g. "is all you need" vs "is not what you need").
def title_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    chars = SequenceMatcher(None, na, nb).ratio()
    wa, wb = na.split(), nb.split()
    common = sum(min(wa.count(w), wb.count(w)) for w in set(wa))
    words = 2 * common / (len(wa) + len(wb))
    return (chars + words) / 2


# True if the hit's year is within one year of the requested year (arXiv vs venue year).
def year_matches(hit, a: QueryAnalysis) -> bool:
    y = year_of(hit)
    if not y:
        return False
    if a.year is not None:
        return abs(y - a.year) <= 1
    if a.year_from is not None or a.year_to is not None:
        return (a.year_from or 0) - 1 <= y <= (a.year_to or 9999) + 1
    return False


# True if any requested author's last name appears in the hit's author list.
def author_matches(hit, authors: list[str]) -> bool:
    if not authors:
        return False
    meta = hit.metadata or {}
    stored = [str(x).lower() for x in (meta.get("author_last_names") or [])]
    stored += [normalize_title(str(x).split(",")[0]) for x in (meta.get("authors") or [])]
    wanted = [normalize_title(a).split()[-1] for a in authors if normalize_title(a)]
    return any(w in s.split() or w == s for w in wanted for s in stored)


# Look up one paper by its exact title (spelling fixed first); the result is just the best matching paper.
def run_title_query(title: str, log: LogFn = lambda msg: print(msg, flush=True),
                    meta: ParsedMessage | None = None) -> SearchResponse:
    response = SearchResponse(prompt=title, mode="title", search_prompt=title)
    diag = response.diagnostics
    diag.original_prompt, diag.mode = title, "title"
    start_context_tracking(diag)
    try:
        log("Checking the title...")
        check, rest = analyze_query(title, title_mode=True, meta=meta)
    except Exception as exc:
        response.errors.append(f"{type(exc).__name__}: {exc}")
        return response
    a = response.analysis = check.analysis
    response.warnings, response.errors = check.warnings, check.errors
    diag += check.diagnostics
    diag.modified_query = a.corrected_query or None
    if response.errors:
        return response

    # Candidates from the hybrid search on the title. Years/authors are NOT filters here (only soft bonuses);
    # an explicit 'category: ...' is a filter, dropped again if it leaves nothing.
    try:
        retriever = get_retriever(log)
        filters = MetadataIntent(category=a.category)
        with _retriever_lock:
            log("Looking up the title...")
            hits = retriever.retrieve(a.corrected_query, K_TITLE, filters)
            if not hits and a.category:
                diag.append("[title] No match in the given category; searched again without it.")
                response.warnings.append("No match in the given category, so it was ignored.")
                hits = retriever.retrieve(a.corrected_query, K_TITLE, MetadataIntent())
    except Exception as exc:
        response.errors.append(f"Qdrant search failed: {type(exc).__name__}: {exc}")
        return response
    if not hits:
        response.errors.append(f"No paper found for the title '{a.corrected_query}'.")
        return response

    # Score: title similarity + bge reranker (title + abstract) + small author/year bonus.
    diag.track_pool("paper", hits, {h.point_id: ("topic", i) for i, h in enumerate(hits, start=1)})
    sim_of = {h.point_id: title_similarity(a.corrected_query, h.title) for h in hits}
    bonus_of = {h.point_id: (TITLE_AUTHOR_BONUS if author_matches(h, a.authors) else 0.0)
                + (TITLE_YEAR_BONUS if year_matches(h, a) else 0.0) for h in hits}
    diag.track_title_sims(sim_of)

    # Rerank pool: best by title similarity (the exact title can sit deep in Qdrant's order) + Qdrant's top
    by_sim = sorted(hits, key=lambda h: sim_of[h.point_id], reverse=True)[:TITLE_POOL_BY_SIM]
    pool = list({h.point_id: h for h in by_sim + hits[:TITLE_POOL_BY_QDRANT]}.values())
    log("Reranking title candidates...")
    try:
        reranked, rr = rerank_hits(a.corrected_query, pool)
        rr_of = {h.point_id: min(max(float(s), 0.0), 1.0) for h, s in zip(reranked, rr)}
    except Exception as exc:
        diag.append(f"[rerank] FAILED for the title pool ({type(exc).__name__}: {exc}); title similarity only.")
        rr_of = {h.point_id: 0.0 for h in pool}

    # Exact titles always win (reranker only breaks ties between them); otherwise blend the two scores
    def score(h) -> float:
        sim, rr_score, bonus = sim_of[h.point_id], rr_of[h.point_id], bonus_of[h.point_id]
        if sim >= TITLE_EXACT:
            return 10.0 + sim + bonus + 0.01 * rr_score
        if sim >= TITLE_BLEND_MIN:
            return TITLE_SIM_WEIGHT * sim + (1 - TITLE_SIM_WEIGHT) * rr_score + bonus
        return (1 - TITLE_SIM_WEIGHT) * sim + TITLE_SIM_WEIGHT * rr_score + bonus

    ranked = sorted(pool, key=score, reverse=True)
    diag.track_rerank(ranked, [rr_of[h.point_id] for h in ranked])

    best = ranked[0]
    response.papers = [best]
    response.top_papers = [1]
    response.best = ("1", best)
    sim, rr_best = sim_of[best.point_id], rr_of[best.point_id]
    if sim >= TITLE_EXACT:
        kind = "Exact title match"
        rule = "exact title (reranker only breaks ties)"
    else:
        kind = "Best available match"
        rule = (f"blend {TITLE_SIM_WEIGHT:.0%} title similarity + {1 - TITLE_SIM_WEIGHT:.0%} reranker"
                if sim >= TITLE_BLEND_MIN else
                f"mostly reranker ({TITLE_SIM_WEIGHT:.0%}), title similarity below {TITLE_BLEND_MIN}")
        response.warnings.append("An exact title match was not found; the result is the best available match.")
    diag.append(f"[title] {len(pool)} candidates reranked ({TITLE_POOL_BY_SIM} best by title + Qdrant's top "
                f"{TITLE_POOL_BY_QDRANT}); decided by {rule}; best title similarity {sim:.2f}, "
                f"reranker {rr_best:.2f}.")
    extras = [f"reranker {rr_best:.2f}"]
    if a.authors:
        extras.append("author matches" if author_matches(best, a.authors) else "author not found on this paper")
    if a.year is not None or a.year_from is not None or a.year_to is not None:
        extras.append(f"year {year_of(best) or '?'}" + (" matches" if year_matches(best, a) else
                                                         " differs from the one given"))
    response.why_best = f"{kind} (title similarity {sim:.2f}" + (f"; {', '.join(extras)}" if extras else "") + ")."
    diag.track_sent(response.papers)
    diag.track_picks([best], [best], "title")
    return response


# Run a text chain; with on_token, stream the reply piece by piece (a failing on_token stops only the display).
def stream_text(chain, inputs: dict, on_token: TokenFn | None = None, config: dict | None = None) -> str:
    if on_token is None:
        return chain.invoke(inputs, config=config)
    parts = []
    for piece in chain.stream(inputs, config=config):
        if not piece:
            continue
        parts.append(piece)
        if on_token is not None:
            try:
                on_token(piece)
            except Exception:
                on_token = None   # display gone (e.g. browser closed): keep collecting the full reply
    return "".join(parts)


# Most authors listed for one paper (get_paper), capped for huge collaborations.
AGENT_MAX_AUTHORS = 40


# --------------------------------------------------------------------------- agent (tool calling)
# Every message without search= / title=: the LLM calls the tools below in a loop (Ollama tool calling via
# LangChain bind_tools) until it can answer, then answers citing arXiv ids. Only ids a tool returned count as
# sources; the answer's ids are renumbered [1], [2], ... to match the paper cards shown under it.
AGENT_SYSTEM_PROMPT = f"""You are a research assistant for a database of arXiv computer science papers ({DB_COVERAGE}).
Answer the user's question using ONLY papers returned by your tools.

Tools:
- search_papers(query, author, year_from, year_to, k): hybrid search + reranker. Use a short focused topic as
  query. Use author / year filters ONLY when the question names them; never add a year range on your own (query may be empty when author is given).
  Split a question with several parts into several searches.
- get_paper(arxiv_id): the full abstract and ALL authors of one paper, when a search snippet is not enough
  (search results list at most 5 authors).
- list_current_results(): the papers the user is looking at now (labels 1, 2, S1, ...), when the question
  refers to them.

Work in at most {AGENT_MAX_STEPS} tool rounds, then answer in at most about 200 words, in short paragraphs or
bullets. Cite every claim with the paper's arXiv id in square brackets, e.g. [2105.01234]. Never cite an id a
tool did not return, and never invent papers, results, numbers or datasets. If the papers found do not answer
the question, say so plainly.

Questions about the papers shown (by label, e.g. "compare 1 and 2", "list all authors of 1 and 3", "which is
newest?") are answered from them; call get_paper for details the listing lacks. Greetings, thanks or questions
about this tool need no tools: answer in one or two sentences."""

ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
# One citation in the agent's answer: the id with optional brackets and an optional 'arXiv:' / 'arXiv ' prefix
CITATION_RE = re.compile(r"(?:\[\s*)?(?:\barxiv\s*:?\s*)?\b(\d{4}\.\d{4,5})(?:v\d+)?\b(?:\s*\])?", re.IGNORECASE)


# Abstract of a hit (its text is 'title\n\nabstract').
def hit_abstract(hit) -> str:
    return hit.text.split("\n\n", 1)[-1]


# Author list of a hit's metadata (stored as a list or as 'A; B').
def hit_authors(hit) -> list[str]:
    authors = hit.metadata.get("authors") or []
    return authors.split("; ") if isinstance(authors, str) else [str(a) for a in authors]


# One paper as the model sees it in a tool result.
def agent_paper(hit, abstract_chars: int | None = AGENT_ABSTRACT_CHARS) -> dict:
    abstract = hit_abstract(hit)
    if abstract_chars and len(abstract) > abstract_chars:
        abstract = abstract[:abstract_chars] + "..."
    authors = hit_authors(hit)
    if abstract_chars:   # search results: short author list; get_paper (abstract_chars=None): all of them
        authors = authors[:5] + (["et al."] if len(authors) > 5 else [])
    return {"arxiv_id": hit.arxiv_id, "title": hit.title, "year": year_of(hit) or None,
            "authors": authors[:AGENT_MAX_AUTHORS], "abstract": abstract}


def to_int(value) -> int | None:
    m = YEAR_RE.search(str(value)) if value not in (None, "") else None
    return int(m.group(1)) if m else None


# The tools for one agent turn. `found` collects every paper a tool returned (arXiv id -> Evidence) so the
# answer's citations can be checked; `steps` collects one diagnostic line per call.
def agent_tools(found: dict, steps: list[str], shown: list[dict], known: dict[str, dict], log: LogFn,
                vouched: set | None = None, searches: list | None = None) -> list:
    vouched = vouched if vouched is not None else set()   # ids a tool returned that have no search hit (shown papers)
    searches = searches if searches is not None else []   # one record per search call (eval trace)
    def search(query: str = "", author: str = "", year_from: int | None = None, year_to: int | None = None,
               k: int = AGENT_DEFAULT_K) -> str:
        topic = " ".join(str(query or "").split())
        authors = [a.strip() for a in re.split(r";|\band\b", str(author or "")) if a.strip()]
        y_from, y_to = to_int(year_from), to_int(year_to)
        if y_from and y_to and y_from > y_to:
            y_from, y_to = y_to, y_from
        # a bound at or beyond the database's own range filters nothing: drop it (e.g. 2007-2026 = no filter)
        y_from = y_from if y_from and y_from > DB_FIRST_YEAR else None
        y_to = y_to if y_to and y_to < DB_LAST_YEAR else None
        try:
            k = max(1, min(int(k or AGENT_DEFAULT_K), AGENT_MAX_K))
        except (TypeError, ValueError):
            k = AGENT_DEFAULT_K
        call = f"search_papers(query='{topic}', author='{'; '.join(authors)}', years={y_from or ''}-{y_to or ''}, k={k})"
        if not topic and not authors:
            steps.append(f"[agent] {call} -> refused: no query and no author.")
            return json.dumps({"error": "Give a query or an author."})
        log(f"Searching: {topic or 'papers'}" + (f" by {', '.join(authors)}" if authors else "") + "...")
        a = QueryAnalysis(corrected_query=topic or " ".join(authors), authors=authors,
                          year_from=y_from, year_to=y_to)
        papers, surveys, _ = search_papers(a, False, log)
        last_name_only = []
        if not papers and not surveys and authors:   # 'Iryna Gurevych' -> also try the last name only
            papers, surveys, _ = search_papers(a, False, log, use_initials=False)
            last_name_only = last_name_matches(papers + surveys, authors)
        pool, seen = [], set()
        for h in papers + surveys:
            if h.arxiv_id not in seen:
                seen.add(h.arxiv_id)
                pool.append(h)
        ranked, scores = rerank_hits(a.corrected_query, pool[:AGENT_RERANK_POOL]) if pool else ([], [])
        top = ranked[:k]
        searches.append({"query": topic, "authors": authors, "year_from": y_from, "year_to": y_to, "k": k,
                         "last_name_only": bool(last_name_only),
                         "ranked": [{"arxiv_id": h.arxiv_id, "rerank_score": round(s, 5),
                                     "pool": "survey" if is_survey(h) else "paper"} for h, s in zip(ranked, scores)],
                         "returned": [h.arxiv_id for h in top]})
        for h in top:
            found.setdefault(h.arxiv_id, h)
        steps.append(f"[agent] {call} -> {len(top)} of {len(pool)}: " + ", ".join(h.arxiv_id for h in top) + "."
                     + (f" Last-name-only matches: {', '.join(last_name_only)}." if last_name_only else ""))
        result = {"papers": [agent_paper(h) for h in top]} if top else \
            {"papers": [], "note": "No papers matched; try a broader query or fewer filters."}
        if top and last_name_only:
            result["note"] = (f"No paper by {' & '.join(authors)} with that first name. These papers are by authors "
                              f"with the same last name ({', '.join(last_name_only)}), who may be different people: "
                              "say so in your answer.")
        return json.dumps(result, ensure_ascii=False)

    def get_paper(arxiv_id: str) -> str:
        m = ARXIV_ID_RE.search(str(arxiv_id or ""))
        if not m:
            steps.append(f"[agent] get_paper('{arxiv_id}') -> not an arXiv id.")
            return json.dumps({"error": "Not an arXiv id, e.g. 2105.01234."})
        pid = m.group(1)
        hit = found.get(pid)
        if hit is None:
            retriever = get_retriever(log)
            from qdrant_client import models
            points, _ = retriever.client.scroll(
                collection_name=COLLECTION, limit=1, with_payload=True,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key=f"{retriever.prefix}arxiv_id", match=models.MatchValue(value=pid))]))
            if points:
                _, meta, text = split_payload(points[0].payload or {})
                if text:
                    hit = Evidence(initial_rank=0, vector_score=0.0, point_id=str(points[0].id), arxiv_id=pid,
                                   title=str(meta.get("title") or text.split("\n", 1)[0]),
                                   url=str(meta.get("url") or f"https://arxiv.org/abs/{pid}"), metadata=meta, text=text)
                    found[pid] = hit
        if hit is None and pid in known:   # shown earlier in this chat (paper dict, not a search hit)
            p = known[pid]
            vouched.add(pid)
            steps.append(f"[agent] get_paper('{pid}') -> from this chat's shown papers.")
            return json.dumps({k: p.get(k) for k in ("arxiv_id", "title", "year", "authors", "abstract")},
                              ensure_ascii=False)
        if hit is None:
            steps.append(f"[agent] get_paper('{pid}') -> not in the database.")
            return json.dumps({"error": f"{pid} is not in the database."})
        steps.append(f"[agent] get_paper('{pid}') -> '{hit.title[:60]}'.")
        return json.dumps(agent_paper(hit, abstract_chars=None), ensure_ascii=False)

    def current_results() -> str:
        steps.append(f"[agent] list_current_results() -> {len(shown)} paper(s).")
        for p in shown:
            known.setdefault(p["arxiv_id"], p)
            vouched.add(p["arxiv_id"])
        return json.dumps({"papers": [{"label": p["label"], "arxiv_id": p["arxiv_id"], "title": p["title"],
                                       "year": p.get("year")} for p in shown]} if shown else
                          {"papers": [], "note": "No papers are shown yet."}, ensure_ascii=False)

    return [
        StructuredTool.from_function(search, name="search_papers", description=(
            "Search the arXiv CS paper database (hybrid search + reranker). query: short research topic; "
            "author: author name(s), several separated by ';'; year_from / year_to: year range; "
            f"k: papers to return (1-{AGENT_MAX_K}). Returns arxiv_id, title, year, authors, abstract.")),
        StructuredTool.from_function(get_paper, name="get_paper", description=(
            "Full abstract and metadata of one paper by its arXiv id, e.g. 2105.01234.")),
        StructuredTool.from_function(current_results, name="list_current_results", description=(
            "The papers currently shown to the user, with their labels (1, 2, S1, ...).")),
    ]


@functools.lru_cache(maxsize=1)
def get_agent_llm():
    return ollama_llm(temperature=0.2, reasoning=True, num_predict=AGENT_MAX_TOKENS)


# Research question -> answer from the tool loop. The cited papers become response.papers (cards 1, 2, ...).
def run_agent(question: str, log: LogFn = lambda msg: print(msg, flush=True), history: list[dict] | None = None,
              shown: list[dict] | None = None, known: dict[str, dict] | None = None) -> SearchResponse:
    response = SearchResponse(prompt=question, mode="agent", search_prompt=question)
    diag = response.diagnostics
    diag.original_prompt, diag.mode = question, "agent"
    found: dict[str, Evidence] = {}
    steps: list[str] = []
    vouched: set[str] = set()
    tools = agent_tools(found, steps, list(shown or []), dict(known or {}), log, vouched, response.agent_searches)
    by_name = {t.name: t for t in tools}
    llm = get_agent_llm()
    messages = [SystemMessage(AGENT_SYSTEM_PROMPT)]
    messages += [HumanMessage(e["content"]) if e["role"] == "user" else AIMessage(e["content"])
                 for e in (history or [])]
    if shown:   # the papers on screen, so labels like '2' or 'S1' need no tool call to resolve
        listing = "\n".join(f"[{p['label']}] {p['arxiv_id']} ({p.get('year') or '?'}) {p['title']}" for p in shown)
        messages.append(SystemMessage(f"Papers currently shown to the user:\n{listing}"))
        vouched.update(p["arxiv_id"] for p in shown)
    messages.append(HumanMessage(question))
    start_context_tracking(diag)
    answer = ""
    try:
        with_tools = llm.bind_tools(tools)
        for step in range(1, AGENT_MAX_STEPS + 1):
            log(f"Thinking (step {step})..." if step > 1 else "Planning the search...")
            ai = with_tools.invoke(messages, config=llm_config(f"agent step {step}"))
            messages.append(ai)
            if not ai.tool_calls:
                answer = strip_thinking(ai.content if isinstance(ai.content, str) else str(ai.content))
                break
            for call in ai.tool_calls:
                tool = by_name.get(call["name"])
                try:
                    result = tool.invoke(call.get("args") or {}) if tool else json.dumps(
                        {"error": f"Unknown tool {call['name']}."})
                except Exception as exc:
                    result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                    steps.append(f"[agent] {call['name']}({call.get('args')}) -> FAILED ({type(exc).__name__}).")
                messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id") or call["name"]))
        else:
            log("Writing the answer...")
            steps.append(f"[agent] Step limit ({AGENT_MAX_STEPS}) reached -> asked for the answer without tools.")
            messages.append(HumanMessage("Stop searching. Answer the question now from the papers above, "
                                         "citing their arXiv ids."))
            ai = llm.invoke(messages, config=llm_config("agent final"))
            answer = strip_thinking(ai.content if isinstance(ai.content, str) else str(ai.content))
    except Exception as exc:
        response.errors.append(f"The research assistant failed ({type(exc).__name__}: {exc}).")
        steps.append(f"[agent] FAILED ({type(exc).__name__}: {str(exc)[:200]}).")
    finally:
        stop_context_tracking()
    diag += steps
    if not answer and not response.errors:
        response.errors.append("The model gave no answer (it may have used its token budget on thinking); "
                               "please ask again, or use search= for a list of papers.")
        diag.append("[agent] FAILED: empty answer.")
    if response.errors:
        return response

    # Citations: ids a tool returned -> [1], [2] ... (cards below the answer); other ids are flagged
    cited: list[str] = []
    unverified: list[str] = []

    # The whole citation is replaced, with its brackets and any 'arXiv:' / 'arXiv ' prefix: '[2101.00190]',
    # 'arXiv:2101.00190' and '[arXiv:2101.00190v2]' all become '[1]' (no leftover 'arXiv:1').
    def renumber(m: re.Match) -> str:
        pid = m.group(1)
        if pid in found:
            if pid not in cited:
                cited.append(pid)
            return f"[{cited.index(pid) + 1}]"
        if pid in vouched:   # a paper shown earlier in the chat: real, but no new card for it
            return f"[{pid}]"
        if pid not in unverified:
            unverified.append(pid)
        return f"[{pid} (unverified)]"

    answer = CITATION_RE.sub(renumber, answer)
    # 'arXiv:2101.00190 [2101.00190]' -> '[1] [1]': keep one
    response.answer = re.sub(r"(\[(\d+)\])([*_\s]{0,4})\[\2\]", lambda m: m.group(1) + m.group(3).strip(), answer)
    response.papers = [found[pid] for pid in cited]
    response.top_papers = list(range(1, len(cited) + 1))
    response.memory_papers = list(cited)
    response.memory_note = "Papers cited: " + "; ".join(f"[{i}] [{h.arxiv_id}] {h.title}"
                                                         for i, h in enumerate(response.papers, start=1))
    diag.append(f"[agent] Answer cites {len(cited)} of {len(found)} paper(s) found"
                + (f"; UNVERIFIED ids (no tool returned them): {', '.join(unverified)}" if unverified else "") + ".")
    if unverified:
        response.warnings.append(f"Ids not found by the search (may be invented): {', '.join(unverified)}.")
    return response


# Input check failed: tell the user, run nothing (no model or search call). The diagnostics keep the
# parser notes so far plus one FAILED line.
def rejected(message: str, parsed: ParsedMessage, error: str, diag_line: str) -> SearchResponse:
    response = SearchResponse(prompt=message, mode=parsed.mode, search_prompt=parsed.text)
    response.diagnostics.original_prompt = " ".join(message.split()[:60]) + (
        " ..." if len(message.split()) > 60 else "")
    response.diagnostics.mode = parsed.mode
    response.errors.append(error)
    response.diagnostics += parsed.notes
    response.diagnostics.append(diag_line)
    return response


# Entry point for one user message. Input checks, in order (no model call until all pass):
#   1. length  : whole message as typed <= MAX_PROMPT_WORDS (too long -> nothing runs)
#   2. symbols : runs of the same symbol -> one, then keep only letters, digits and = , - : . ; ? ' &
#   3. keys    : search= / title= (with optional year= / author=) run the search or title pipeline exactly as
#                typed. Anything else goes to the agent, which decides for itself what to search or look up.
# There are no remembered modes: each message is handled on its own (the history is still read).
# shown / known = papers of the last result / all papers shown in the chat (the agent can read them).
# current_mode and on_mode are accepted for old callers and ignored.
def run_turn(message: str, log: LogFn = lambda msg: print(msg, flush=True),
             history: list[dict] | None = None, on_token: TokenFn | None = None,
             newest_first: bool = False, current_mode: str | None = None,
             shown: list[dict] | None = None, known: dict[str, dict] | None = None,
             on_mode: Callable[[str], None] | None = None) -> SearchResponse:
    message = message or ""
    problem = length_problem(message)
    if problem:
        return rejected(message, ParsedMessage(mode="search", text=message), *problem)

    clean_notes = []
    collapsed = collapse_repeats(message)
    if collapsed != " ".join(message.split()):
        clean_notes.append(f"[query] Step 2 symbols (collapse): '{' '.join(message.split())}' -> '{collapsed}'.")
    cleaned = remove_symbols(collapsed)
    if cleaned != collapsed:
        clean_notes.append(f"[query] Step 2 symbols (remove): '{collapsed}' -> '{cleaned}'.")

    parsed = parse_message(cleaned, None)
    parsed.notes[:0] = clean_notes
    mode = parsed.mode_key if parsed.mode_key in ("search", "title") else None

    if mode is None:   # no search= / title=: the agent answers (analyze= and stray keys are just text to it)
        question = parsed.text if parsed.mode_key == "analyze" and parsed.text.strip() else cleaned
        if not re.search(r"[^\W_]", question):
            return rejected(message, parsed, "Type a question or a topic, e.g. recent papers on RAG evaluation.",
                            "[query] Step 2 symbols (remove): no words left -> FAILED.")
        if AGENT_ENABLED:
            answer = run_agent(question, log, select_history(history)[0], shown, known)
            answer.prompt = message
            answer.diagnostics.messages[:0] = clean_notes + ["[query] No search= / title= -> the agent answers."]
            return answer
        parsed = parse_message(f"search= {question}", None)   # AGENT=0: searched as a topic
        parsed.notes[:0] = clean_notes + ["[query] No search= / title= and AGENT=0 -> searched as a topic."]
        mode = "search"

    if parsed.errors:
        return rejected(message, parsed, " ".join(parsed.errors),
                        f"[query] Step 3 keys: {parsed.summary()} -> FAILED: {' '.join(parsed.errors)}")
    if not re.search(r"[^\W_]", parsed.text):
        what = "paper title" if mode == "title" else "topic"
        return rejected(message, parsed, f"{mode}= has no words left after removing symbols; type a "
                                         f"{what}, e.g. {MODE_EXAMPLES[mode]}.",
                        f"[query] Step 2 symbols (remove): nothing but symbols in the {mode}= text -> FAILED.")
    query = parsed.text
    try:
        if mode == "title":
            response = run_title_query(query, log, meta=parsed)
        else:
            response = run_query(query, log, select_history(history)[0], newest_first=newest_first, meta=parsed)
    finally:
        stop_context_tracking()
    response.diagnostics.finish_search(query, NUM_CTX, keyword=mode,
                                       parser_notes=[f"[query] Parser: {parsed.summary()}"] + parsed.notes)
    return response


# One picked paper as '- [label] (year) title\\n   url'.
def format_hit(label: str, hit) -> str:
    return f"- [{label}] ({year_label(hit)}) {hit.title}\n   {hit.url}"


# Render a SearchResponse as plain text.
def format_response(r: SearchResponse, show_reasoning: bool = SHOW_REASONING) -> str:
    if r.mode == "agent":
        if not r.answer:
            return "Can't answer:\n" + "\n".join(f"  - {e}" for e in r.errors)
        sources = [f"  [{i}] ({year_label(h)}) {h.title}\n      {h.url}" for i, h in enumerate(r.papers, start=1)]
        return "\n".join([r.answer] + [f"Warning: {w}" for w in r.warnings] + (["\nSources:"] + sources if sources else []))
    lines = []
    if r.analysis and r.analysis.authors:
        how = "hint, not a filter" if r.mode == "title" else "filtered"
        lines.append(f"Authors ({how}): {', '.join(r.analysis.authors)}")
    lines += [f"Warning: {w}" for w in r.warnings]
    if r.errors:
        lines.append("\nCan't complete this query:")
        lines += [f"  - {e}" for e in r.errors]
        return "\n".join(lines)

    if r.search_prompt and r.search_prompt != r.prompt:
        lines.append(f"Follow-up understood as: {r.search_prompt}")
    if show_reasoning and r.reasoning:
        lines += ["\nModel reasoning:", r.reasoning]
    if r.criteria:
        lines.append("\nEvaluation criteria:")
        lines += [f"  - {c}" for c in r.criteria]

    if r.mode == "title":
        lines.append("\nBest matching paper:")
        lines += [format_hit(str(n), r.papers[n - 1]) for n in r.top_papers] or ["  (none)"]
    else:
        if not r.survey_focus:
            lines.append("\nThree best matches (newest first):")
            lines += [format_hit(str(n), r.papers[n - 1]) for n in r.top_papers] or ["  (none)"]
        lines.append("\nSurvey papers (top by reranker" + (", newest first):" if r.newest_first else "):"))
        lines += [format_hit(f"S{n}", r.surveys[n - 1]) for n in r.top_surveys] or ["  (no relevant surveys found)"]

    if r.best:
        label, hit = r.best
        lines += [f"\nBest match: [{label}] {hit.title}", f"Why: {r.why_best}"]

    lines.append("\nAll retrieved papers:")
    lines += [f"[{i}] {h.title} (rrf={h.vector_score:.4f}) {h.url}" for i, h in enumerate(r.papers, start=1)]
    lines += [f"[S{i}] {h.title} (rrf={h.vector_score:.4f}) {h.url}" for i, h in enumerate(r.surveys, start=1)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- user selection


# One tracked chunk: the hit data plus whether its GUI checkbox is currently ticked.
@dataclass
class TrackedChunk:
    hit: object              # Evidence from the retriever
    checked: bool = False    # True = checkmark shown in the GUI; only these are written


# Chunks shown to the user (insertion-ordered) with their checkmark status; writes only checked ones.
class ChunkSelection:
    # Start with no tracked chunks.
    def __init__(self) -> None:
        self._chunks: dict[str, TrackedChunk] = {}   # point_id -> TrackedChunk

    # Track a hit (keeps the existing checkmark if already tracked); returns its TrackedChunk.
    def add(self, hit, checked: bool = False) -> TrackedChunk:
        chunk = self._chunks.get(hit.point_id)
        if chunk is None:
            chunk = self._chunks[hit.point_id] = TrackedChunk(hit, checked)
        return chunk

    # Track every paper and survey of a response, unchecked (call when results are shown).
    def add_response(self, response: SearchResponse) -> None:
        for hit in response.papers + response.surveys:
            self.add(hit)

    # Stop tracking a chunk; returns False if it was not tracked.
    def remove(self, chunk_id: str) -> bool:
        return self._chunks.pop(chunk_id, None) is not None

    # Set the checkmark of a tracked chunk; returns False if the id is unknown.
    def set_checked(self, chunk_id: str, checked: bool) -> bool:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return False
        chunk.checked = checked
        return True

    # Flip the checkmark of a hit (tracking it first if needed); returns the new state.
    def toggle(self, hit) -> bool:
        chunk = self.add(hit)
        chunk.checked = not chunk.checked
        return chunk.checked

    # Current checkmark of a chunk (False if not tracked).
    def is_checked(self, chunk_id: str) -> bool:
        chunk = self._chunks.get(chunk_id)
        return bool(chunk and chunk.checked)

    # Untick every chunk but keep tracking them.
    def uncheck_all(self) -> None:
        for chunk in self._chunks.values():
            chunk.checked = False

    # Forget all tracked chunks.
    def clear(self) -> None:
        self._chunks.clear()

    # All tracked chunk ids in order.
    @property
    def ids(self) -> list[str]:
        return list(self._chunks)

    # Ids of chunks currently bearing a checkmark, in order.
    @property
    def checked_ids(self) -> list[str]:
        return [cid for cid, c in self._chunks.items() if c.checked]

    # True if the chunk id is tracked.
    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    # Number of tracked chunks.
    def __len__(self) -> int:
        return len(self._chunks)

    # Write only the checked chunks to a JSONL file (one per line, format in export.py); returns the path.
    def write(self, path: str | Path = DEFAULT_SELECTION_FILE) -> Path:
        return write_chunks_jsonl(((cid, self._chunks[cid].hit) for cid in self.checked_ids), path)


# Find a hit by its display label ("3" -> papers[2], "S2" -> surveys[1]); None if invalid.
def hit_by_label(response: SearchResponse, label: str):
    label = label.strip().upper()
    items, num = (response.surveys, label[1:]) if label.startswith("S") else (response.papers, label)
    return items[int(num) - 1] if num.isdigit() and 1 <= int(num) <= len(items) else None


# --------------------------------------------------------------------------- GUI action stubs (kept; Chainlit handlers below use ChunkSelection directly)
# GUI stub: user ticks/unticks a paper (by label) in the results list.
def on_paper_toggled(selection: ChunkSelection, response: SearchResponse, label: str) -> bool:
    hit = hit_by_label(response, label)
    if hit is None:
        response.diagnostics.append(f"[selection] Unknown paper label '{label}' ignored.")
        return False
    return selection.toggle(hit)


# GUI stub: new results are displayed; track them all as unchecked.
def on_results_shown(selection: ChunkSelection, response: SearchResponse) -> None:
    selection.add_response(response)


# GUI stub: user clicks "Clear selection".
def on_clear_selection(selection: ChunkSelection) -> None:
    selection.uncheck_all()


# Write checked chunks to a file; returns the path, or None if nothing is checked.
def on_save_selection(selection: ChunkSelection, path: str | Path = DEFAULT_SELECTION_FILE) -> Path | None:
    if not selection.checked_ids:
        return None
    return selection.write(path)


# --------------------------------------------------------------------------- GUI (Chainlit)
# Run with:  chainlit run threeofus_main.py -w   (GUI helpers live in app.py)
try:
    import chainlit as cl
    import app as gui
    import db
    GUI_AVAILABLE = True
except ImportError:          # CLI still works without chainlit installed
    GUI_AVAILABLE = False

if GUI_AVAILABLE:
    # Password login against the accounts table in history.db (python db.py adduser NAME). Each username
    # has its own chat history in the left pane; the chats from before the login belong to "local".
    @cl.password_auth_callback
    def gui_login(username: str, password: str):
        name = db.check_login(username, password)
        return cl.User(identifier=name) if name else None

    # ---- Sign-up page (/signup): creates an account in history.db, then sends the user to the login page.
    # The login page links here via public/custom.js. Chainlit serves its UI from a catch-all GET route,
    # so these routes are moved in front of it (and old copies dropped when -w reloads this file).
    import html as _html
    from chainlit.server import app as _server
    from fastapi import Form
    from fastapi.responses import HTMLResponse, RedirectResponse

    def _signup_page(error: str = "", username: str = "") -> HTMLResponse:
        err = f'<p class="err">{_html.escape(error)}</p>' if error else ""
        return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign up</title><style>
:root{{--bg:#fff;--fg:#111;--muted:#666;--card:#f5f5f5;--border:#ddd;--accent:#2563eb;--err:#b91c1c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#999;--card:#1c1c1c;--border:#333;--accent:#60a5fa;--err:#f87171}}}}
body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif}}
form{{width:min(360px,calc(100vw - 32px));display:flex;flex-direction:column;gap:12px}}
h1{{font-size:1.5rem;margin:0 0 8px}} label{{font-size:.9rem;display:flex;flex-direction:column;gap:4px}}
input{{padding:10px;border:1px solid var(--border);border-radius:8px;background:var(--card);color:var(--fg);font-size:1rem}}
button{{padding:10px;border:0;border-radius:8px;background:var(--accent);color:#fff;font-size:1rem;cursor:pointer}}
.hint{{color:var(--muted);font-size:.8rem}} .err{{color:var(--err);margin:0}} a{{color:var(--accent)}}
</style></head><body><form method="post" action="/signup">
<h1>Create an account</h1>{err}
<label>Username<input name="username" value="{_html.escape(username)}" required autofocus autocomplete="username"></label>
<span class="hint">{_html.escape(db.USERNAME_RULE)}</span>
<label>Password<input name="password" type="password" required minlength="{db.MIN_PASSWORD}" autocomplete="new-password"></label>
<label>Repeat password<input name="password2" type="password" required autocomplete="new-password"></label>
<button type="submit">Sign up</button>
<span class="hint">Already have an account? <a href="/login">Log in</a></span>
</form></body></html>""")

    async def gui_signup_form():
        return _signup_page()

    async def gui_signup(username: str = Form(""), password: str = Form(""), password2: str = Form("")):
        if password != password2:
            return _signup_page("Passwords do not match.", username)
        try:
            name = db.create_account(username, password)
        except ValueError as exc:
            return _signup_page(str(exc), username)
        print(f"[signup] created account '{name}'", flush=True)
        return RedirectResponse("/login?signup=ok", status_code=303)

    # ---- "Newest first" box in the top bar (public/custom.js): saved per user in history.db (table prefs).
    # Only the logged-in user's own value is read or written (Chainlit's auth cookie identifies them).
    from chainlit.auth import get_current_user
    from fastapi import Depends, HTTPException, Request
    from fastapi.responses import JSONResponse

    NEWEST_FIRST_PREF = "newest_first"

    async def gui_get_newest_first(current_user=Depends(get_current_user)):
        if not current_user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return JSONResponse({"value": db.get_pref(current_user.identifier, NEWEST_FIRST_PREF) == "1"})

    async def gui_set_newest_first(request: Request, current_user=Depends(get_current_user)):
        if not current_user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        value = bool((await request.json()).get("value"))
        db.set_pref(current_user.identifier, NEWEST_FIRST_PREF, "1" if value else "0")
        return JSONResponse({"value": value})

    _OWN_ROUTES = ("/signup", "/prefs/newest-first")
    _server.router.routes[:] = [r for r in _server.router.routes if getattr(r, "path", None) not in _OWN_ROUTES]
    _server.add_api_route("/signup", gui_signup_form, methods=["GET"], response_class=HTMLResponse)
    _server.add_api_route("/signup", gui_signup, methods=["POST"])
    _server.add_api_route("/prefs/newest-first", gui_get_newest_first, methods=["GET"])
    _server.add_api_route("/prefs/newest-first", gui_set_newest_first, methods=["POST"])
    # stable sort: our routes go in front of Chainlit's catch-all UI route
    _server.router.routes.sort(key=lambda r: getattr(r, "path", None) not in _OWN_ROUTES)

    # Chat storage for the left pane (SQLite history.db, see db.py); deleting a chat also deletes its files.
    @cl.data_layer
    def gui_data_layer():
        return db.make_data_layer()

    # If this session's chat was deleted (e.g. from the left pane or another tab while it stayed open), move the
    # session to a fresh chat so this message starts a new thread in the left pane instead of being dropped.
    async def ensure_live_thread(message: cl.Message) -> None:
        import uuid
        from chainlit.data import get_data_layer
        session = cl.context.session
        if session.thread_id not in getattr(get_data_layer(), "deleted_ids", set()):
            return
        old_id = session.thread_id
        session.thread_id = str(uuid.uuid4())
        cl.user_session.set(db.TRACE_FILES_KEY, [])              # the old chat's files are already deleted
        cl.user_session.set(db.TRACE_SESSIONS_KEY, [session.id])
        message.thread_id = session.thread_id
        await cl.context.emitter.init_thread(message.content)    # creates the chat row and tells the browser
        await message.update()                                   # save this user message under the new chat
        print(f"[delete] session was on deleted chat {old_id}; started new chat {session.thread_id}", flush=True)

    # Record what this chat leaves on disk (session folder, written files) in the session; Chainlit saves it in
    # the chat's metadata, so deleting the chat from the left pane can remove these traces (db.delete_chat_files).
    def record_trace(file_path: Path | None = None) -> None:
        sessions = cl.user_session.get(db.TRACE_SESSIONS_KEY) or []
        if cl.context.session.id not in sessions:
            sessions.append(cl.context.session.id)
            cl.user_session.set(db.TRACE_SESSIONS_KEY, sessions)
        if file_path is not None:
            files = cl.user_session.get(db.TRACE_FILES_KEY) or []
            if str(file_path) not in files:
                files.append(str(file_path))
                cl.user_session.set(db.TRACE_FILES_KEY, files)

    # Per-chat ChunkSelection (tracks every shown chunk and its checkmark); rebuilt if missing.
    def gui_selection() -> ChunkSelection:
        selection = cl.user_session.get("selection")
        if not isinstance(selection, ChunkSelection):
            selection = rebuild_selection()
            cl.user_session.set("selection", selection)
        return selection

    # Rebuild the ChunkSelection from the saved papers; reading-list papers are checked.
    def rebuild_selection() -> ChunkSelection:
        selection = ChunkSelection()
        checked = {p["arxiv_id"] for p in gui.reading_list()}
        for p in (cl.user_session.get("papers_by_id") or {}).values():
            selection.add(gui.paper_to_hit(p), checked=p["arxiv_id"] in checked)
        return selection

    # Old chat reopened from the left pane: restore state, buttons and sidebar.
    @cl.on_chat_resume
    async def gui_resume(thread: dict):
        gui.ensure_session()
        record_trace()
        cl.user_session.set("selection", rebuild_selection())
        await gui.resume(thread)

    # New chat: reset state and greet.
    @cl.on_chat_start
    async def gui_start():
        gui.init_session()
        record_trace()
        cl.user_session.set("selection", ChunkSelection())
        await gui.show_mode(None)   # banner: neutral until the first message
        await gui.send_welcome()
        status = cl.Message("Checking the models and the database...")
        await status.send()

        # Only shown when something actually has to be loaded or reconnected.
        def log(msg: str) -> None:
            status.content = msg
            cl.run_sync(status.update())

        try:
            lines = await cl.make_async(health_check)(log)
        except Exception as exc:
            status.content = f"**Setup failed:** {type(exc).__name__}: {exc}"
            await status.update()
            return
        status.content = "**Ready.**\n" + "\n".join(f"- {line}" for line in lines)
        await status.update()

    # User prompt: run the pipeline off the event loop, then show summary, paper cards and diagnostics.
    @cl.on_message
    async def gui_on_message(message: cl.Message):
        prompt = (message.content or "").strip()
        if not prompt:
            return
        await ensure_live_thread(message)   # chat deleted while open -> continue in a new chat (left pane entry)
        if not gui.is_pinned():          # unpinned panel auto-hides on a new question
            await gui.close_sidebar()
        status = cl.Message("Thinking...")
        await status.send()

        # Progress messages from the worker thread update the status message.
        def log(msg: str) -> None:
            status.content = msg
            cl.run_sync(status.update())

        # Non-search replies stream into one message; the status line goes away when the first words arrive.
        answer = cl.Message(content="")
        streaming = {"started": False}

        def on_token(piece: str) -> None:
            if not streaming["started"]:
                streaming["started"] = True
                cl.run_sync(status.remove())
            cl.run_sync(answer.stream_token(piece))

        # Banner and box colour show what this message runs (nothing is remembered): search= / title= as typed,
        # anything else the agent ("ask")
        shown = gui.last_result_papers()
        typed = None if length_problem(prompt) else parse_message(prompt, None).mode_key
        await gui.show_mode(typed if typed in ("search", "title") else "ask")

        user = cl.context.session.user
        newest_first = bool(user) and db.get_pref(user.identifier, NEWEST_FIRST_PREF) == "1"   # top-bar box
        r = await cl.make_async(run_turn)(prompt, log, list(gui.history()), on_token,
                                          newest_first=newest_first, shown=shown,
                                          known=dict(cl.user_session.get("papers_by_id") or {}))
        if not streaming["started"]:
            await status.remove()
        tags = memory_tags(r)   # decides whether later turns read this question/answer
        gui.remember("user", prompt, tags)
        gui.remember("assistant", assistant_turn_text(r), tags)
        selection = gui_selection()
        on_results_shown(selection, r)
        if r.mode in ("search", "title") and not r.errors:
            gui.set_last_results(gui.picked_hits(r), r.best[0] if r.best else None)   # labels the agent can read

        if r.mode == "agent":
            # Answer first, then a card per cited paper (labels 1, 2, ... match the [1], [2] in the answer)
            if r.errors:
                await gui.send_notice("**Can't answer:**\n" + "\n".join(f"- {e}" for e in r.errors))
            else:
                await gui.send_notice(r.answer + "".join(f"\n\n⚠️ {w}" for w in r.warnings))
                for label, hit in gui.picked_hits(r):
                    await gui.send_paper(gui.to_paper(hit, label), selection.is_checked(hit.point_id))
                if r.papers:
                    gui.set_last_results(gui.picked_hits(r))   # follow-ups like 'compare 1 and 2' work
        elif r.errors:
            await gui.send_notice("**Can't complete this query:**\n" + "\n".join(f"- {e}" for e in r.errors))
        else:
            await gui.send_summary(gui.summary_md(r, describe_filters(r.analysis)))
            best = r.best[1] if r.best else None
            for label, hit in gui.picked_hits(r):
                paper = gui.to_paper(hit, label, r.why_best if hit is best else "")
                await gui.send_paper(paper, selection.is_checked(hit.point_id))
        await gui.send_diagnostics_panel(r.diagnostics.to_dict())   # collapsible panel under the answer
        print(format_diagnostics(r), flush=True)   # console copy of the digest
        await gui.save_session()   # shown papers, last results, mode: kept even if the server is restarted

    # "Reading list" button: open the sidebar.
    @cl.action_callback("open_list")
    async def gui_open_list(action: cl.Action):
        await gui.refresh_sidebar()

    # Select button on a paper card: flip its checkmark and sync the reading list.
    @cl.action_callback("add_paper")
    async def gui_toggle_paper(action: cl.Action):
        arxiv_id = action.payload["arxiv_id"]
        paper = gui.paper_by_id(arxiv_id)
        if paper is None:
            return
        if gui_selection().is_checked(paper["chunk_id"]):
            await unselect_paper(arxiv_id, refresh_panel=True)   # same path as the bin in the panel
            return
        gui_selection().set_checked(paper["chunk_id"], True)
        await gui.add_to_list(arxiv_id)
        await gui.set_button_state(arxiv_id, True)
        await show_list_change()
        await gui.save_session()

    # After a select/unselect on a card: pin the panel and send the new list without redrawing it.
    # An open panel updates in place (no reload); a closed panel is opened by public/custom.js.
    async def show_list_change() -> None:
        gui.set_pinned(True)
        await gui.push_reading_list()

    # Unselect a paper: uncheck its chunk, remove it from the reading list, reset its card button.
    # Used by both the card button and the bin in the panel; refresh_panel=False when the panel already
    # removed the paper itself (bin), True when the change came from the card.
    async def unselect_paper(arxiv_id: str, refresh_panel: bool) -> None:
        paper = gui.paper_by_id(arxiv_id)
        if paper is not None:
            gui_selection().set_checked(paper["chunk_id"], False)
        await gui.remove_from_list(arxiv_id)
        await gui.set_button_state(arxiv_id, False)
        if refresh_panel:
            await show_list_change()
        await gui.save_session()

    # Bin button in the sidebar: unselect the paper (the panel already removed it itself, no refresh).
    @cl.action_callback("remove_paper")
    async def gui_remove_paper(action: cl.Action):
        await unselect_paper(action.payload["arxiv_id"], refresh_panel=False)

    # Drag/arrow reorder in the sidebar: save the order only (panel already reordered itself, no reload).
    @cl.action_callback("reorder")
    async def gui_reorder(action: cl.Action):
        gui.reorder(action.payload["order"])
        await gui.save_session()

    # Pin button in the sidebar: remember the pin state (default pinned).
    @cl.action_callback("pin_list")
    async def gui_pin_list(action: cl.Action):
        gui.set_pinned(action.payload.get("pinned", True))
        await gui.save_session()

    # Export button: write the checked chunks (JSONL) and build the zip (BibTeX + PDFs).
    @cl.action_callback("export")
    async def gui_export(action: cl.Action):
        items = gui.reading_list()
        print(f"[export] requested: {len(items)} paper(s) in the reading list", flush=True)
        if not items:
            await gui.send_notice("Your reading list is empty: click **☐ Select** under a paper first.")
            return
        # The reading list is what the user selected: make sure those chunks are checked before writing
        selection = gui_selection()
        for p in items:
            selection.add(gui.paper_to_hit(p))
            selection.set_checked(p["chunk_id"], True)
        try:
            # Named after the chat (thread) id, so deleting the chat can find and remove the file
            path = on_save_selection(selection, selection_file(cl.context.session.thread_id))
            record_trace(path)
            chunks = await cl.Message(f"Saved {len(selection.checked_ids)} selected chunk(s).").send()
            await cl.File(name="selected_chunks.jsonl", path=str(path), display="inline").send(for_id=chunks.id)
            print(f"[export] chunks written to {path}", flush=True)
        except Exception as exc:
            await gui.send_notice(f"❌ Could not write the selected chunks: {type(exc).__name__}: {exc}")
        ok = await gui.export_list()     # list unchanged, so the panel is not redrawn
        print(f"[export] zip {'sent' if ok else 'FAILED'}", flush=True)


# Run one message in the console: chat replies are printed while being written, search results at the end.
def run_turn_console(message: str, history: list[dict] | None = None) -> SearchResponse:
    streamed = {"any": False}

    def on_token(piece: str) -> None:
        if not streamed["any"]:
            streamed["any"] = True
            print("\nAssistant> ", end="", flush=True)
        print(piece, end="", flush=True)

    response = run_turn(message, history=history, on_token=on_token)
    if streamed["any"]:
        print(flush=True)                        # end the streamed line
        if response.errors:
            print(format_response(response))
    else:
        print(format_response(response))
    print(format_diagnostics(response))
    return response


# CLI entry point: run the prompt (argument or default), print the result, then the diagnostics.
def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    response = run_turn_console(" ".join(args) if args else DEFAULT_PROMPT)
    return 0 if response.ok else 1


# Interactive CLI with conversation memory: follow-ups use earlier turns; empty line quits.
def chat_loop() -> None:
    history: list[dict] = []
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        response = run_turn_console(prompt, history=list(history))
        add_turn(history, "user", prompt)
        add_turn(history, "assistant", assistant_turn_text(response))


if __name__ == "__main__":
    if sys.argv[1:] == ["--chat"]:
        chat_loop()
    else:
        sys.exit(main())


# --------------------------------------------------------------------------- notes
# Developer notes (running, pipeline, modes, resume, accounts) are in notes.md.
