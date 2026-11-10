"""Turn CS-related arXiv files into RAG records and ingest them into Qdrant.

Input files are processed one at a time: each file's records are embedded (dense + sparse)
and upserted into a Qdrant collection as soon as the file has been read.
"""

from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
import uuid
from email.utils import parsedate_to_datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

MAX_AUTHORS = 20          # authors kept in the display list "authors"
MAX_AUTHOR_KEYS = 20      # authors kept in the search fields author_last_names / author_keys
OUTPUT_SUFFIX = "-rag"

MODEL_ID = "BAAI/bge-base-en-v1.5"                            # 768 dimensions
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"   # pinned commit; must match retriever_hybrid_rrf.py
SPARSE_MODEL_ID = "Qdrant/bm25"

# Sparse Vector Pruning Thresholds
MIN_SPARSE_WEIGHT = 0.30   # Minimum BM25 weight floor to drop noise
MAX_SPARSE_TERMS = 35      # Hard ceiling for maximum query keyword terms (queries only)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_COLLECTION_NAME = "sebun_bge768_hybrid_full"
DEFAULT_BATCH_SIZE = 128
PIPELINE_VERSION = 4      # Bumped: documents no longer pruned (full BM25 vectors), sparse IDF modifier
RETRIEVE_BATCH = 1000     # Point ids fetched per Qdrant call
PAYLOAD_UPDATE_BATCH = 500   # payload-only updates sent per Qdrant request
# Dense (HNSW) indexing is switched off while bulk loading a new collection and switched back on
# at the end (Qdrant's recommended bulk-upload pattern). 20000 KB is Qdrant's usual threshold.
INDEXING_THRESHOLD_ON = 20000

PAYLOAD_INDEXES = {
    "metadata.year": models.PayloadSchemaType.INTEGER,
    "metadata.primary_category": models.PayloadSchemaType.KEYWORD,
    "metadata.categories": models.PayloadSchemaType.KEYWORD,
    "metadata.is_cs_primary": models.PayloadSchemaType.BOOL,
    "metadata.authors": models.PayloadSchemaType.KEYWORD,
    # Normalized author fields for author search (see author_search_fields):
    "metadata.author_last_names": models.PayloadSchemaType.KEYWORD,   # ["goodfellow", "bengio"]
    "metadata.author_keys": models.PayloadSchemaType.KEYWORD,         # ["goodfellow_i", "bengio_y"]
}


def prune_sparse_vector(
    sparse_vec: models.SparseVector,
    min_weight: float = MIN_SPARSE_WEIGHT,
    max_terms: int = MAX_SPARSE_TERMS,
) -> models.SparseVector:
    """Filters sparse vector indices by weight floor and caps total count."""
    valid = [
        (idx, val) for idx, val in zip(sparse_vec.indices, sparse_vec.values)
        if val >= min_weight
    ]
    valid.sort(key=lambda item: item[1], reverse=True)
    valid = valid[:max_terms]

    if not valid and len(sparse_vec.indices) > 0:
        best_idx, best_val = max(
            zip(sparse_vec.indices, sparse_vec.values),
            key=lambda item: item[1]
        )
        valid = [(best_idx, best_val)]

    return models.SparseVector(
        indices=[item[0] for item in valid],
        values=[item[1] for item in valid]
    )


class PrunedFastEmbedSparse(FastEmbedSparse):
    """FastEmbedSparse wrapper applying weight floor and count ceiling pruning."""

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        # No pruning for documents: every word of a paper stays searchable by keyword.
        # (Pruning is only applied to queries, in embed_query below.)
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> models.SparseVector:
        vector = super().embed_query(text)
        return prune_sparse_vector(vector)


# --------------------------------------------------------------------------- helpers
def clean_text(value):
    return " ".join(value.split()) if value else ""


def format_author(parts):
    parts = [clean_text(p) for p in parts] + ["", ""]
    last, first, suffix = parts[0], parts[1], parts[2]
    name = f"{last}, {first}" if first else last
    return f"{name}, {suffix}" if suffix else name


# LaTeX accents used in arXiv metadata: M\"uller, Gr\"{o}bner, {\'E}cole, \c{c}, \ss, \o
LATEX_ACCENT_RE = re.compile(r"\\[`'\"^~=.uvHcdbtkr]\s*\{?\s*([A-Za-z])\s*\}?")
LATEX_LETTERS = {r"\ss": "ss", r"\o": "o", r"\O": "O", r"\l": "l", r"\L": "L",
                 r"\i": "i", r"\j": "j", r"\ae": "ae", r"\AE": "AE", r"\oe": "oe", r"\aa": "a", r"\AA": "A"}


def normalize_name(value: str) -> str:
    """'M\\"uller' / 'Müller' / 'MÜLLER' -> 'muller'; "O'Brien" -> 'obrien'; 'García-López' -> 'garcia-lopez'.
    The same function must be used on the query side so stored and searched names match."""
    s = LATEX_ACCENT_RE.sub(r"\1", value or "")
    for latex, plain in LATEX_LETTERS.items():
        s = re.sub(re.escape(latex) + r"(?![A-Za-z])", plain, s)
    s = re.sub(r"\\[A-Za-z]+", " ", s).replace("{", "").replace("}", "")      # other LaTeX commands
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()   # strip accents
    s = s.replace("'", "").replace("\u2019", "")
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    return " ".join(s.split()).strip("-")


def parse_author_string(authors: str) -> list[list[str]]:
    """Fallback when authors_parsed is missing: 'Ian Goodfellow, Yoshua Bengio and A. Courville (MIT)'
    -> [["Goodfellow", "Ian", ""], ["Bengio", "Yoshua", ""], ["Courville", "A.", ""]]."""
    text = re.sub(r"\([^)]*\)", " ", authors or "")                       # drop affiliations
    parts = []
    for name in re.split(r",|;|\band\b|&", text):
        tokens = clean_text(name).split()
        if tokens:
            parts.append([tokens[-1], " ".join(tokens[:-1]), ""])
    return parts


def author_search_fields(parsed: list[list[str]]) -> tuple[list[str], list[str]]:
    """[["Goodfellow", "Ian J.", ""], ...] -> (["goodfellow", ...], ["goodfellow_i", ...]).
    author_keys = last name + first initial; only added when a first name is known."""
    last_names, keys = [], []
    for p in parsed[:MAX_AUTHOR_KEYS]:
        p = list(p) + ["", ""]
        last, first = normalize_name(p[0]), normalize_name(p[1])
        if not last:
            continue
        if last not in last_names:
            last_names.append(last)
        if first:
            key = f"{last}_{first[0]}"
            if key not in keys:
                keys.append(key)
    return last_names, keys


def parse_created(value):
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def build_record(rec, max_authors=MAX_AUTHORS, flat_lists=False):
    arxiv_id = clean_text(rec.get("id"))
    title = clean_text(rec.get("title"))
    abstract = clean_text(rec.get("abstract"))
    if not arxiv_id or not (title or abstract):
        return None

    text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)

    # authors_parsed = [["Last", "First", "Suffix"], ...]; if missing, split the plain "authors" string
    parsed = [a for a in (rec.get("authors_parsed") or []) if a and any(clean_text(x) for x in a)]
    if not parsed and rec.get("authors"):
        parsed = parse_author_string(rec["authors"])
    authors = [format_author(a) for a in parsed]
    author_last_names, author_keys = author_search_fields(parsed)

    categories = (rec.get("categories") or "").split()
    cs_categories = [c for c in categories if c.startswith("cs.")]

    versions = rec.get("versions") or []
    published = parse_created(versions[0].get("created")) if versions else None
    latest_version = versions[-1].get("version") if versions else None

    metadata = {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors[:max_authors],
        "num_authors": len(authors),
        "author_last_names": author_last_names,
        "author_keys": author_keys,
        "categories": categories,
        "primary_category": categories[0] if categories else None,
        "cs_categories": cs_categories,
        "is_cs_primary": bool(categories) and categories[0].startswith("cs."),
        "published": published.isoformat() if published else None,
        "year": published.year if published else None,
        "updated": rec.get("update_date"),
        "latest_version": latest_version,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "doi": clean_text(rec.get("doi")),
        "journal_ref": clean_text(rec.get("journal-ref")),
    }

    if flat_lists:
        metadata["authors"] = "; ".join(metadata["authors"])
        # author_last_names / author_keys stay lists even here: a joined string would break keyword matching
        metadata["categories"] = " ".join(categories)
        metadata["cs_categories"] = " ".join(cs_categories)

    metadata = {k: v for k, v in metadata.items() if v is not None and v != "" and v != []}
    return {"id": arxiv_id.replace("/", "_"), "metadata": metadata, "main_body": {"text": text}}


# --------------------------------------------------------------------------- files
def iter_records(path):
    with open(path, encoding="utf-8") as f:
        first = ""
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return
            if line.strip():
                first = line.lstrip()[0]
                f.seek(pos)
                break
        if first == "[":
            yield from json.load(f)
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None


def find_inputs(specs):
    found = []
    for spec in specs:
        if os.path.isdir(spec):
            candidates = glob.glob(os.path.join(spec, "*"))
        elif os.path.isfile(spec):
            candidates = [spec]
        else:
            candidates = glob.glob(spec)
        for c in candidates:
            stem = os.path.splitext(os.path.basename(c))[0]
            if os.path.isfile(c) and not stem.endswith(OUTPUT_SUFFIX) and c not in found:
                found.append(c)
    return sorted(found)


def read_file(src, max_authors=MAX_AUTHORS, flat_lists=False):
    """Returns (records, lines read, skipped counts). Skipped records are counted by reason
    so the run can report them instead of dropping them silently."""
    read = 0
    objects = []
    skipped = {"invalid JSON": 0, "no arXiv id or no title/abstract": 0}
    for rec in iter_records(src):
        read += 1
        if not isinstance(rec, dict):
            skipped["invalid JSON"] += 1
            continue
        built = build_record(rec, max_authors, flat_lists)
        if built is None:
            skipped["no arXiv id or no title/abstract"] += 1
        else:
            objects.append(built)
    return objects, read, skipped


def write_file(src, objects, out_dir=None):
    folder = os.path.abspath(out_dir) if out_dir else os.path.dirname(os.path.abspath(src))
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(Path(folder).parent, f"{stem}{OUTPUT_SUFFIX}.jsonl")
    with open(dst, "w", encoding="utf-8") as out:
        for obj in objects:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return dst


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="folder(s), file(s) or glob pattern(s)")
    p.add_argument("-o", "--out-dir", default=None, help="folder for output files")
    p.add_argument("--db-dir", default=None, help="folder for embedded Qdrant database")
    p.add_argument("--qdrant-url", default=None, help="URL of a Qdrant server")
    p.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help=f"collection name (default: {DEFAULT_COLLECTION_NAME})")
    p.add_argument("--max-authors", type=int, default=MAX_AUTHORS, help=f"max authors listed (default: {MAX_AUTHORS})")
    p.add_argument("--flat-lists", action="store_true", help="store list fields as strings")
    p.add_argument("--no-write", action="store_true", help="do not write -rag.jsonl files")
    p.add_argument("--device", default="auto", help="embedding device: auto (default), cpu, cuda")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"batch size (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--fp16", action="store_true", help="run dense model in fp16 (GPU only)")
    p.add_argument("--force", action="store_true", help="re-embed and upsert every record")
    p.add_argument("--verify", metavar="QUERY", default=None, help="run similarity search after ingestion")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- qdrant
def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {requested} requested but CUDA GPU not visible.")
    return requested


@functools.lru_cache(maxsize=1)
def get_embeddings(device: str = "cpu", batch_size: int = DEFAULT_BATCH_SIZE, fp16: bool = False):
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs = {"device": device, "revision": MODEL_REVISION}
    if fp16 and device.startswith("cuda"):
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    return HuggingFaceEmbeddings(
        model_name=MODEL_ID,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": QUERY_PREFIX,
        },
    )


@functools.lru_cache(maxsize=1)
def get_sparse_embeddings():
    return PrunedFastEmbedSparse(model_name=SPARSE_MODEL_ID)


def point_id(record_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


def base_metadata(item: dict, src: str | Path) -> dict:
    metadata = dict(item["metadata"])
    metadata["source_file"] = os.path.basename(str(src))
    return metadata


def text_hash(item: dict, salt: str) -> str:
    meta = item["metadata"]
    payload = json.dumps(
        [item["main_body"]["text"], meta.get("updated"), meta.get("latest_version"), salt],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def meta_hash(metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_entry(item: dict, src: str | Path, salt: str) -> tuple:
    return item, text_hash(item, salt), meta_hash(base_metadata(item, src))


def stored_hashes(client: QdrantClient, collection_name: str, point_ids: list[str]) -> dict[str, tuple]:
    if not point_ids or not client.collection_exists(collection_name):
        return {}
    found = {}
    for i in range(0, len(point_ids), RETRIEVE_BATCH):
        batch = point_ids[i:i + RETRIEVE_BATCH]
        for point in client.retrieve(collection_name, ids=batch, with_payload=True, with_vectors=False):
            meta = (point.payload or {}).get("metadata") or {}
            found[str(point.id)] = (meta.get("text_hash"), meta.get("meta_hash"))
    return found


def classify(client: QdrantClient, collection_name: str, objects: list[dict], src: str | Path, salt: str):
    entries = [hash_entry(o, src, salt) for o in objects]
    stored = stored_hashes(client, collection_name, [point_id(o["id"]) for o in objects])
    to_embed, meta_only = [], []
    for entry in entries:
        item, th, mh = entry
        old = stored.get(point_id(item["id"]))
        if old is None or old[0] != th:
            to_embed.append(entry)
        elif old[1] != mh:
            meta_only.append(entry)
    return to_embed, meta_only, len(entries) - len(to_embed) - len(meta_only)


def update_payloads(client: QdrantClient, collection_name: str, entries: list[tuple], src: str | Path) -> None:
    """Payload-only updates (vector unchanged), sent in batches instead of one request per paper."""
    for i in range(0, len(entries), PAYLOAD_UPDATE_BATCH):
        operations = []
        for item, th, mh in entries[i:i + PAYLOAD_UPDATE_BATCH]:
            metadata = {**base_metadata(item, src), "text_hash": th, "meta_hash": mh}
            operations.append(models.OverwritePayloadOperation(
                overwrite_payload=models.SetPayload(
                    payload={"page_content": item["main_body"]["text"], "metadata": metadata},
                    points=[point_id(item["id"])],
                )
            ))
        client.batch_update_points(collection_name, update_operations=operations)


def db_path(args) -> Path:
    if args.db_dir:
        return Path(args.db_dir)
    base = Path(args.out_dir) if args.out_dir else Path.cwd()
    return base / "qdrant_db"


def open_client(args) -> QdrantClient:
    if args.qdrant_url:
        return QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(path=str(db_path(args)))


def ensure_collection(client: QdrantClient, collection_name: str, embeddings, remote: bool) -> None:
    if client.collection_exists(collection_name):
        return
    size = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=size, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            # modifier=IDF: Qdrant weights rare terms higher at query time, which completes BM25
            # (the Qdrant/bm25 model only stores term frequencies and expects the server to add IDF)
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(),
                modifier=models.Modifier.IDF,
            )
        },
        # indexing_threshold=0: no HNSW building during the bulk load (much faster upload).
        # restore_indexing() switches it back on at the end of the run.
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    print(f"Created collection '{collection_name}' (dense indexing paused during the load).", flush=True)
    if remote:
        for field, schema in PAYLOAD_INDEXES.items():
            client.create_payload_index(collection_name, field_name=field, field_schema=schema)


def restore_indexing(client: QdrantClient, collection_name: str) -> None:
    """Switch dense (HNSW) indexing back on. Runs at the end of every run, even after an error or
    Ctrl+C, so a collection is never left without its search index."""
    try:
        if not client.collection_exists(collection_name):
            return
        client.update_collection(
            collection_name,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD_ON),
        )
        print(f"\nIndexing switched back on for '{collection_name}'. Qdrant now builds the search index "
              "in the background; wait until the collection status is green before searching.", flush=True)
    except Exception as e:
        print(f"\nWARNING: could not switch indexing back on ({type(e).__name__}: {e}).\n"
              f"Searches will be slow until it is on. Re-run this script, or set indexing_threshold="
              f"{INDEXING_THRESHOLD_ON} for '{collection_name}' in the Qdrant dashboard.",
              file=sys.stderr, flush=True)


def load_documents(entries: list[tuple], src: str | Path) -> list[Document]:
    documents = []
    for item, th, mh in entries:
        metadata = {**base_metadata(item, src), "text_hash": th, "meta_hash": mh}
        documents.append(
            Document(page_content=item["main_body"]["text"], metadata=metadata, id=point_id(item["id"]))
        )
    return documents


def to_qdrant(store: QdrantVectorStore, collection_name: str, documents: list[Document], batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    if not documents:
        return
    print(f"Adding {len(documents)} document(s) to Qdrant hybrid collection '{collection_name}'...")
    store.add_documents(documents, ids=[doc.id for doc in documents], batch_size=batch_size)


def verify_persistence(args, query: str) -> None:
    print("\nReopening the Qdrant store to verify hybrid persistence...")
    client = open_client(args)
    collection = args.collection
    try:
        if not client.collection_exists(collection):
            raise RuntimeError(f"Failed to find collection '{collection}' after reopening!")

        embeddings = get_embeddings(resolve_device(args.device), args.batch_size, args.fp16)
        sparse_embeddings = get_sparse_embeddings()

        store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )
        info = client.get_collection(collection)
        print(f"Verified collection state: {info.points_count} total point(s) stored.")

        print(f"Running hybrid similarity search query: '{query}'")
        hits = store.similarity_search_with_score(query, k=2)

        print("\nSearch results from reopened store:")
        for rank, (doc, score) in enumerate(hits, start=1):
            print(f" [{rank}] Score: {score:.4f} | arXiv id: {doc.metadata.get('arxiv_id', doc.id)}")
            print(f"     Text: {doc.page_content.strip()[:200]}...\n")
    finally:
        client.close()


# --------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    files = find_inputs(args.inputs)
    if not files:
        sys.exit("No input files found for: " + " ".join(args.inputs))

    device = resolve_device(args.device)
    remote = bool(args.qdrant_url)
    collection = args.collection
    if not remote:
        print("Note: using embedded local Qdrant store.", file=sys.stderr)

    salt = f"{PIPELINE_VERSION}|{MODEL_ID}|{MODEL_REVISION}|{SPARSE_MODEL_ID}|min_w={MIN_SPARSE_WEIGHT},max_k={MAX_SPARSE_TERMS}"
    total_read = total_written = ingested = meta_updated = skipped_unchanged = 0
    failed_files: list[tuple[str, str]] = []   # (file name, reason) — reported at the end
    total_skipped: dict[str, int] = {}         # skipped records by reason, over all files
    store = None
    client = open_client(args)
    try:
        for src in files:
            name = os.path.basename(src)
            try:
                objects, records_read, skipped = read_file(src, args.max_authors, args.flat_lists)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"  {name}: FAILED to read ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
                failed_files.append((name, f"could not read: {type(e).__name__}: {e}"))
                continue
            for reason, count in skipped.items():
                total_skipped[reason] = total_skipped.get(reason, 0) + count
            if records_read and not objects:
                print(f"  {name}: WARNING no usable records in this file", file=sys.stderr, flush=True)

            unchanged = payload_only = 0
            if objects:
                try:
                    if args.force:
                        to_embed, meta_only = [hash_entry(o, src, salt) for o in objects], []
                    else:
                        to_embed, meta_only, _ = classify(client, collection, objects, src, salt)
                    if to_embed:
                        if store is None:
                            embeddings = get_embeddings(device, args.batch_size, args.fp16)
                            sparse_embeddings = get_sparse_embeddings()
                            ensure_collection(client, collection, embeddings, remote)
                            store = QdrantVectorStore(
                                client=client,
                                collection_name=collection,
                                embedding=embeddings,
                                sparse_embedding=sparse_embeddings,
                                retrieval_mode=RetrievalMode.HYBRID,
                                vector_name=DENSE_VECTOR_NAME,
                                sparse_vector_name=SPARSE_VECTOR_NAME,
                            )
                        to_qdrant(store, collection, load_documents(to_embed, src), args.batch_size)
                    update_payloads(client, collection, meta_only, src)
                    ingested += len(to_embed)
                    payload_only = len(meta_only)
                    meta_updated += payload_only
                    unchanged = len(objects) - len(to_embed) - payload_only
                    skipped_unchanged += unchanged
                except Exception as e:
                    print(f"  {name}: FAILED Qdrant ingestion ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
                    failed_files.append((name, f"ingestion failed: {type(e).__name__}: {e}"))

            if not args.no_write:
                try:
                    dst = write_file(src, objects, args.out_dir)
                    print(f"  {name}: output written to {dst}", flush=True)
                except OSError as e:
                    # A full disk or locked file must not stop the whole run; report it instead
                    print(f"  {name}: FAILED to write output file ({type(e).__name__}: {e})",
                          file=sys.stderr, flush=True)
                    failed_files.append((name, f"could not write output: {type(e).__name__}: {e}"))

            total_read += records_read
            total_written += len(objects)
            parts = []
            if unchanged:
                parts.append(f"{unchanged:,} unchanged, skipped")
            if payload_only:
                parts.append(f"{payload_only:,} payload-only updates")
            for reason, count in skipped.items():
                if count:
                    parts.append(f"{count:,} skipped: {reason}")
            note = f" ({', '.join(parts)})" if parts else ""
            print(f"  {name}: {len(objects):,} of {records_read:,} records processed{note}", flush=True)
    finally:
        # Always runs (also on errors / Ctrl+C): never leave the collection without its search index
        restore_indexing(client, collection)
        client.close()

    print(f"\nDone: {total_written:,} of {total_read:,} records from {len(files)} file(s); "
          f"{ingested:,} embedded into Qdrant, {meta_updated:,} payload-only updates, "
          f"{skipped_unchanged:,} unchanged and skipped")
    if any(total_skipped.values()):
        print("Records skipped (not ingested): " + ", ".join(
            f"{count:,} {reason}" for reason, count in total_skipped.items() if count))
    if args.verify is not None:
        if ingested or meta_updated or skipped_unchanged:
            try:
                verify_persistence(args, args.verify)
            except Exception as e:
                print(f"\nFAILED verification ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
                failed_files.append(("--verify", f"{type(e).__name__}: {e}"))
        else:
            print("Skipping verification: nothing was ingested and nothing was found in the store.")
    # Failure summary last, so it's the final thing on screen
    if failed_files:
        print(f"\n{len(failed_files)} problem(s) — re-run the script to retry "
              "(already stored papers are skipped):", file=sys.stderr)
        for name, reason in failed_files:
            print(f"  - {name}: {reason}", file=sys.stderr)
    else:
        print("\nCompleted without errors.")
    if failed_files:
        sys.exit(1)   # non-zero exit code so a failed run can't be mistaken for a successful one


if __name__ == "__main__":
    main()