"""Turn the reading list into a downloadable zip: reading_list.bib + one PDF per paper.

- reading_list.bib can be imported into Zotero or Mendeley (File -> Import).
- The PDFs come from arxiv.org. arXiv asks for at most 1 request every 3 seconds,
  so 5 papers take about 15 seconds. That's why we wait between downloads.

Try the BibTeX part on its own:  .venv/Scripts/python.exe -X utf8 export.py
"""

import asyncio
import io
import re
import zipfile

import httpx

ARXIV_DELAY_S = 3


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
                    # Number the files so the folder keeps the reading-list order: 01_2306.16313.pdf
                    z.writestr(f"{i + 1:02d}_{p['arxiv_id']}.pdf", r.content)
                except httpx.HTTPError as err:
                    failed.append(f"{p['arxiv_id']} ({type(err).__name__})")
    return buffer.getvalue(), failed


if __name__ == "__main__":
    from rag import ask
    print(build_bibtex(ask("retrieval augmented generation evaluation")["papers"][:2]))
