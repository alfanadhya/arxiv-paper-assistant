"""Input checks for one user message, before any model call (used by threeofus_main.run_turn).

  1. length   : length_problem: the whole message as typed is at most MAX_PROMPT_WORDS words (every mode).
  2. symbols  : on the whole message, collapse_repeats turns runs of the same symbol into one, then
                remove_symbols keeps letters, digits and = , - : . ; ? ' & and turns other symbols into spaces.
  3. keys     : parse_message reads the mode keys search= / title= / analyze= (also analyse=) and the metadata
                keys year= / author= in one go; each value runs to the next key. A mode key switches the chat's
                mode, which is remembered for the next messages until another mode key is typed.
"""

import re
from dataclasses import dataclass, field

MESSAGE_KEY_RE = re.compile(r"(?<![\w=])(search|title|analy[sz]e|year|author)\s*=\s*", re.IGNORECASE)
MODES = ("search", "title", "analyze")   # mode keys; the last one used is remembered for the chat
YEAR_VALUE_RE = re.compile(r"^\s*(\d{4})?\s*(-)?\s*(\d{4})?\s*$")
AUTHOR_SPLIT_RE = re.compile(r";|,|&|\band\b", re.IGNORECASE)
# Cleaning of the whole message (after the length check, before the keys are read), in two steps:
#  1. collapse: repeated spaces to one, and a run of the same symbol to one ('...', '; ;', '???', '!!' , '//')
#  2. symbols: keep letters, digits, spaces and the wanted symbols = , - : . ; ? ' & ('=' for the keys,
#     '-' for year ranges, ';' ',' '&' between authors; GPT-3.5, cs.CR, don't); curly/back-tick apostrophes
#     become ', en/em dashes become '-'; every other symbol becomes a space (graph/network -> graph network).
APOSTROPHE_RE = re.compile(r"[’`´]")
DASH_RE = re.compile(r"[–—]")
SYMBOL_RE = re.compile(r"[^\w\s=,\-:.;?'&]|_")
REPEATED_SYMBOL_RE = re.compile(r"([^\w\s])(?:\s*\1)+")
SEARCH_EXAMPLE = "'search= graph neural networks for drug discovery year=2020-'"
TITLE_EXAMPLE = "'title= Attention is all you need'"
ANALYZE_EXAMPLE = "'analyze= compare 1 and 2'"
MODE_EXAMPLES = {"search": SEARCH_EXAMPLE, "title": TITLE_EXAMPLE, "analyze": ANALYZE_EXAMPLE}
MAX_PROMPT_WORDS = 200   # first input check: longer messages are not processed; the user is asked to shorten


# Result of the simple key parser (parse_message): mode, topic/title text and the metadata keys.
@dataclass
class ParsedMessage:
    mode: str = "chat"                       # "search", "title", "analyze", or "chat" (no key, no remembered mode)
    text: str = ""                           # the search= / title= / analyze= value (whole message for chat)
    from_memory: bool = False                # no mode key in the message: the chat's remembered mode is used
    mode_key: str | None = None              # the one valid mode key typed (the chat switches to it), else None
    authors: list[str] = field(default_factory=list)   # from author=
    year: int | None = None                  # year=2021
    year_from: int | None = None             # year=2019- / year=2019-2022
    year_to: int | None = None               # year=-2020 / year=2019-2022
    errors: list[str] = field(default_factory=list)    # shown to the user; nothing is run
    notes: list[str] = field(default_factory=list)     # diagnostics
    meta_keys: list[str] = field(default_factory=list) # the year= / author= keys as typed, e.g. ['year=2019-']

    # The text plus the metadata keys as typed: what the spell-check LLM sees (metadata is extracted, but
    # not removed from the request), e.g. 'transformers after year=2026-'.
    def with_meta(self, text: str | None = None) -> str:
        return " ".join([text if text is not None else self.text] + self.meta_keys).strip()

    # One line for the diagnostics, e.g. "mode=search, text='gnn', year=2019-2022, authors=Kipf".
    def summary(self) -> str:
        parts = [f"mode={self.mode}" + (" (remembered)" if self.from_memory else ""), f"text='{self.text}'"]
        if self.year is not None:
            parts.append(f"year={self.year}")
        elif self.year_from is not None or self.year_to is not None:
            parts.append(f"year={self.year_from or ''}-{self.year_to or ''}")
        if self.authors:
            parts.append(f"authors={'; '.join(self.authors)}")
        return ", ".join(parts)


# Step 2 of the input checks (whole message): repeated spaces -> one, a run of the same symbol -> one
# ('...' -> '.', '; ;' -> ';', '???' -> '?', '!!' -> '!', ',,' -> ',', '==' -> '=').
def collapse_repeats(text: str) -> str:
    return " ".join(REPEATED_SYMBOL_RE.sub(r"\1", text or "").split())


# Step 3 of the input checks (whole message): keep letters, digits and = , - : . ; ? ' & (curly apostrophes
# -> ', en/em dashes -> -), every other symbol becomes a space; tidy spaces around commas and trim stray
# , : ; at the ends ('-' is kept there: 'year=2019-' ends with it).
def remove_symbols(text: str) -> str:
    text = SYMBOL_RE.sub(" ", DASH_RE.sub("-", APOSTROPHE_RE.sub("'", text or "")))
    text = re.sub(r"\s*,\s*", ", ", text)                 # ' , ' -> ', '
    return " ".join(text.split()).strip(" ,:;")


# Words that count for the length checks: tokens with at least one letter or digit ('--' or '???' don't count).
def count_words(text: str) -> int:
    return sum(1 for w in (text or "").split() if re.search(r"[^\W_]", w))


# Step 1 of the input checks, the key parser (no model call): finds the mode keys search= / title= /
# analyze= (also 'analyse=') and the metadata keys year= / author= anywhere in the message; each value runs
# to the next key. A mode key switches the chat's mode; without one, `current_mode` (the mode remembered
# from an earlier message) is used and the text before any metadata key is its value. No key and no
# remembered mode -> chat. Problems (no or several modes, a key twice, an empty value, a bad year, an empty
# author) go to .errors and nothing is run. The text is kept as typed here (only spaces collapsed); run_turn
# checks its length (step 2) and then cleans it (steps 3-4).
#   'search= gnn for drug discovery year=2019-2022 author=Kipf; Welling'
#     -> mode=search, text='gnn for drug discovery', year_from=2019, year_to=2022, authors=['Kipf', 'Welling']
#   'gnn for drug discovery year=2020' with current_mode='search' -> mode=search (remembered), year=2020
def parse_message(message: str, current_mode: str | None = None) -> ParsedMessage:
    text = message or ""
    keys = list(MESSAGE_KEY_RE.finditer(text))
    p = ParsedMessage(text=" ".join(text.split()))
    if not keys and not current_mode:
        return p                                   # no key at all and no mode yet: chat
    lead = " ".join(text[:keys[0].start()].split()) if keys else p.text
    values: dict[str, list[str]] = {}
    for i, m in enumerate(keys):                   # a value runs to the next key (or the end)
        end = keys[i + 1].start() if i + 1 < len(keys) else len(text)
        key = m.group(1).lower().replace("analyse", "analyze")
        value = " ".join(text[m.end():end].split()).strip(" ,;")
        values.setdefault(key, []).append(value)
        if key in ("year", "author") and value:
            p.meta_keys.append(f"{key}={value}")

    modes = [k for k in MODES if k in values]
    if len(modes) > 1 or (modes and len(values[modes[0]]) > 1):
        p.mode, p.text = modes[0], ""
        p.errors.append("Use only one of search=, title= or analyze= per message, once: search= for a topic, "
                        "title= for one paper by its exact title, analyze= for a question about the papers shown.")
        return p
    if modes:                                      # a mode key: switch to it
        p.mode, p.text, p.mode_key = modes[0], values[modes[0]][0], modes[0]
        if lead:
            p.notes.append(f"[query] Ignored text before the {p.mode}= key: '{lead}'.")
    elif current_mode:                             # no mode key: stay in the remembered mode
        p.mode, p.text, p.from_memory = current_mode, lead.strip(" ,;"), True
        p.notes.append(f"[query] No mode key: using the remembered mode {current_mode}=.")
    else:
        p.mode, p.text = "search", ""
        p.errors.append("Add search=<topic>, title=<paper title> or analyze=<question> to your message, "
                        f"e.g. {SEARCH_EXAMPLE}.")
        return p
    if not p.text:
        p.errors.append(f"Type something after {p.mode}=, e.g. {MODE_EXAMPLES[p.mode]}." if not p.from_memory
                        else f"Type your {'question' if p.mode == 'analyze' else 'text'} (still in {p.mode}= "
                             f"mode), e.g. {MODE_EXAMPLES[p.mode]}.")

    years = values.get("year", [])
    if len(years) > 1:
        p.errors.append("Use year= only once, e.g. year=2021, year=2019-2022, year=2019- or year=-2020.")
    elif years:
        m = YEAR_VALUE_RE.match(years[0].replace("–", "-").replace("—", "-"))
        start, dash, stop = (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)
        if start and not dash and not stop:
            p.year = int(start)
        elif dash and (start or stop):
            p.year_from = int(start) if start else None
            p.year_to = int(stop) if stop else None
        else:
            p.errors.append(f"year={years[0] or '(empty)'} is not a year: use year=2021, year=2019-2022, "
                            "year=2019- (from) or year=-2020 (until).")

    for value in values.get("author", []):
        names = [" ".join(n.strip(" .'\"").split()) for n in AUTHOR_SPLIT_RE.split(value)]
        names = [n for n in names if n]
        if not names:
            p.errors.append("Type a name after author=, e.g. author=Thomas Kipf (several: author=Kipf; Welling).")
        for n in names:
            if n.lower() not in (a.lower() for a in p.authors):
                p.authors.append(n)
    if p.mode == "analyze" and (p.authors or years):
        p.notes.append("[query] year=/author= are ignored with analyze= (it only uses papers already shown).")
        p.authors, p.year, p.year_from, p.year_to, p.meta_keys = [], None, None, None, []
    return p


# Step 1 of the input checks, before anything else: the whole message as typed is at most MAX_PROMPT_WORDS
# words (only words with a letter/digit count). Same limit for every mode. Returns (user message, diagnostic
# line) if it is too long, else None.
def length_problem(message: str) -> tuple[str, str] | None:
    n_message = count_words(message)
    if n_message > MAX_PROMPT_WORDS:
        return (f"Your message has {n_message} words; please shorten it to at most {MAX_PROMPT_WORDS} words "
                "and send it again.",
                f"[validation] FAILED: message has {n_message} words (limit {MAX_PROMPT_WORDS}) -> not processed.")
    return None
