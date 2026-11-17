# arXiv Paper Assistant

Find computer science papers on arXiv (2007 to Oct 2026), collect them in a reading list and export them for Zotero or Mendeley.

## Asking

Type your question in your own words. The assistant searches the papers and answers, citing the papers it used; each cited paper is shown as a card under the answer.

- A research question: *How are GNNs used for drug discovery?*
- An author: *What has Iryna Gurevych published on fact checking?*
- A paper title: *Attention is all you need*

## Keywords for a faster answer

Start a message with a keyword to skip the assistant and run a fixed search. A keyword applies only to the message it is typed in.

| Keyword | What it does | Example |
|---|---|---|
| `search=` | Lists research papers and surveys on a topic | `search= graph auto-encoders year=2016- author=Kipf; Welling` |
| `title=` | Finds one paper by its exact title | `title= Attention is all you need` |

Filters, with `search=` or `title=`:
- `year=` takes `2021` (exact), `2019-2022` (range), `2019-` (from) or `-2020` (until).
- `author=` takes one or more names separated by `;`. With several names, a paper must have all of them.
- With `search=` they are strict filters; with `title=` they only nudge the score.

## Follow-up questions

Papers are numbered 1, 2, 3, … in each result, and every new result starts again at 1. Numbers always refer to the **latest** result:

- *Compare 1 and 2*: a short summary of each paper (topic, objective, method, results) and how they differ
- *List all authors of 1 and 3*
- *Which is newest?*

## Reading list and export

Click **☐ Select** under a paper to add it to your reading list. The panel on the right (open it with **Reading list** in the top bar; the number shows how many papers it holds) lets you reorder and remove papers, and exports a zip with a BibTeX file, the PDFs and the selected text.

## Settings

- **Newest first** (top bar): shows the papers of your next `search=` newest first instead of by relevance.
- **⚙ Settings**:
  - **Reranker cutoff**: papers scored below it are left out. Higher means fewer, more relevant papers.
  - **Show diagnostics expanded**: opens the diagnostics panel under every answer.

## Diagnostics

The **Diagnostics** panel under each answer shows how your message was read, what was searched, how the papers were ranked and which ones were used. It is saved with the chat.
