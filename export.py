"""Everything the Export button produces.

1. Selected chunks: selections/selected_chunks_<session>.jsonl, one checked chunk per line
   (chunk_record / write_chunks_jsonl / selection_file).
2. reading_list.zip: reading_list.bib + one PDF per paper (build_zip).
   - reading_list.bib can be imported into Zotero or Mendeley (File -> Import).
   - The PDFs come from arxiv.org. arXiv asks for at most 1 request every 3 seconds,
     so 5 papers take about 15 seconds. That's why we wait between downloads.

Self-contained: does not import threeofus_main or app.
"""

import asyncio
import io
import json
import re
import zipfile
from pathlib import Path

import httpx

ARXIV_DELAY_S = 3
DEFAULT_SELECTION_FILE = "selected_chunks.jsonl"                     # default name for the chunks file
SELECTION_DIR = Path(__file__).resolve().parent / "selections"       # GUI exports go here (not the cwd)


# ---------------------------------------------------------------- selected chunks (JSONL)
# One chunk as a JSON-serializable record (hit = retriever Evidence or a rebuilt equivalent).
def chunk_record(chunk_id: str, hit) -> dict:
    year = (hit.metadata or {}).get("year")
    return {"chunk_id": chunk_id, "arxiv_id": hit.arxiv_id, "title": hit.title, "url": hit.url,
            "year": year if isinstance(year, int) else None, "text": hit.text, "metadata": hit.metadata}


# Write (chunk_id, hit) pairs to a JSONL file, one chunk per line; returns the path.
def write_chunks_jsonl(chunks, path: str | Path = DEFAULT_SELECTION_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk_id, hit in chunks:
            f.write(json.dumps(chunk_record(chunk_id, hit), ensure_ascii=False) + "\n")
    return path


# Per-chat chunks file used by the GUI export: selections/selected_chunks_<chat id>.jsonl
# (named after the chat/thread id so deleting the chat can find and remove it).
def selection_file(chat_id: str | None) -> Path:
    return SELECTION_DIR / f"selected_chunks_{chat_id or 'local'}.jsonl"


# ---------------------------------------------------------------- reading-list zip (BibTeX + PDFs)


def bibtex_safe(text: str) -> str:
    """BibTeX breaks on unbalanced { }. Abstracts can contain them (LaTeX), so drop them if unbalanced."""
    depth = 0
    for ch in text:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            break
    return text if depth == 0 else text.replace("{", "").replace("}", "")


def bibtex_entry(p: dict) -> str:
    # Citation key: first author's last name + year + first title word, e.g. "lewis2020retrieval"
    last_name = re.sub(r"[^a-z]", "", p["authors"][0].split()[-1].lower()) if p["authors"] else "anon"
    first_word = re.sub(r"[^a-z]", "", p["title"].split()[0].lower())
    key = f"{last_name}{p['year']}{first_word}"
    # Papers added before the "abstract" field existed only have the snippet.
    abstract = bibtex_safe(p.get("abstract") or p.get("abstract_snippet", ""))
    return (
        f"@misc{{{key},\n"
        f"  title         = {{{p['title']}}},\n"
        f"  author        = {{{' and '.join(p['authors'])}}},\n"
        f"  year          = {{{p['year']}}},\n"
        f"  eprint        = {{{p['arxiv_id']}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  primaryClass  = {{{p['categories'][0]}}},\n"
        f"  url           = {{{p['abs_url']}}},\n"
        f"  abstract      = {{{abstract}}}\n"
        f"}}\n"
    )


def build_bibtex(papers: list[dict]) -> str:
    return "\n".join(bibtex_entry(p) for p in papers)


MAX_TITLE_CHARS = 80   # keeps full paths well under Windows' 260-character limit


# Make text safe for a file name on Windows/macOS/Linux: drop LaTeX/invalid characters, spaces -> '_'.
def safe_filename_part(text: str, max_chars: int = MAX_TITLE_CHARS) -> str:
    text = re.sub(r"\$[^$]*\$", "", text or "")                 # inline LaTeX math
    text = re.sub(r"\\[A-Za-z]+|[{}]", "", text)                # LaTeX commands and braces
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)          # characters not allowed in file names
    text = "_".join(text.split())
    text = text[:max_chars].rstrip("._-")
    return text or "untitled"


# PDF name inside the zip: '<nn>_<arXiv id>_<title>.pdf' (old-style ids like cs/0101001 -> cs_0101001).
def pdf_filename(index: int, p: dict) -> str:
    arxiv_id = p["arxiv_id"].replace("/", "_")
    return f"{index:02d}_{arxiv_id}_{safe_filename_part(p.get('title', ''))}.pdf"


async def build_zip(papers: list[dict], on_progress=None) -> tuple[bytes, list[str]]:
    """Returns (zip bytes, list of papers whose PDF failed to download)."""
    buffer = io.BytesIO()
    failed = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("reading_list.bib", build_bibtex(papers))
        async with httpx.AsyncClient(follow_redirects=True, timeout=60,
                                     headers={"User-Agent": "paper-assistant-prototype"}) as client:
            for i, p in enumerate(papers):
                if i > 0:
                    await asyncio.sleep(ARXIV_DELAY_S)   # be polite to arXiv
                if on_progress:
                    await on_progress(i + 1, len(papers), p)
                try:
                    r = await client.get(p["pdf_url"])
                    r.raise_for_status()
                    # Numbered to keep the reading-list order: 01_2306.16313_Attention_Is_All_You_Need.pdf
                    z.writestr(pdf_filename(i + 1, p), r.content)
                except Exception as err:   # any download problem: skip this PDF, keep the rest
                    failed.append(f"{p['arxiv_id']} ({type(err).__name__})")
    return buffer.getvalue(), failed
