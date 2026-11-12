"""Turn CS-related arXiv files into RAG records and ingest them into Qdrant (dense + sparse hybrid).

CLI:  python db_generator_hybrid_gpu.py <inputs...> [options]
GUI:  build an IngestConfig and call run_ingestion(config, log=..., progress=..., should_stop=...).
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
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

MAX_AUTHORS = 20          # authors kept in the display list "authors"
MAX_AUTHOR_KEYS = 20      # authors kept in author_last_names / author_keys
OUTPUT_SUFFIX = "-rag"

MODEL_ID = "BAAI/bge-base-en-v1.5"                            # 768 dimensions
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"   # must match retriever_hybrid_rrf.py
SPARSE_MODEL_ID = "Qdrant/bm25"

MIN_SPARSE_WEIGHT = 0.30   # query-side BM25 weight floor
MAX_SPARSE_TERMS = 35      # query-side max sparse terms

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_COLLECTION_NAME = "sebun_bge768_hybrid_full"
DEFAULT_BATCH_SIZE = 128
PIPELINE_VERSION = 4          # bump to force re-embedding of everything
RETRIEVE_BATCH = 1000         # point ids fetched per Qdrant call
PAYLOAD_UPDATE_BATCH = 500    # payload-only updates per Qdrant request
INDEXING_THRESHOLD_ON = 20000 # HNSW threshold restored after bulk load

PAYLOAD_INDEXES = {
    "metadata.year": models.PayloadSchemaType.INTEGER,
    "metadata.primary_category": models.PayloadSchemaType.KEYWORD,
    "metadata.categories": models.PayloadSchemaType.KEYWORD,
    "metadata.is_cs_primary": models.PayloadSchemaType.BOOL,
    "metadata.authors": models.PayloadSchemaType.KEYWORD,
    "metadata.author_last_names": models.PayloadSchemaType.KEYWORD,
    "metadata.author_keys": models.PayloadSchemaType.KEYWORD,
}

# LaTeX accents used in arXiv metadata: M\"uller, Gr\"{o}bner, {\'E}cole, \c{c}, \ss, \o
LATEX_ACCENT_RE = re.compile(r"\\[`'\"^~=.uvHcdbtkr]\s*\{?\s*([A-Za-z])\s*\}?")
LATEX_LETTERS = {r"\ss": "ss", r"\o": "o", r"\O": "O", r"\l": "l", r"\L": "L",
                 r"\i": "i", r"\j": "j", r"\ae": "ae", r"\AE": "AE", r"\oe": "oe", r"\aa": "a", r"\AA": "A"}

LogFn = Callable[[str, str], None]            # log(message, level) with level "info" | "warning" | "error"
ProgressFn = Callable[[int, int, str], None]  # progress(files_done, files_total, current_file_name)


# --------------------------------------------------------------------------- config / result
# All settings for one ingestion run (filled by the CLI or a GUI).
@dataclass
class IngestConfig:
    inputs: list[str]
    out_dir: str | None = None
    db_dir: str | None = None
    qdrant_url: str | None = None
    collection: str = DEFAULT_COLLECTION_NAME
    max_authors: int = MAX_AUTHORS
    flat_lists: bool = False
    no_write: bool = False
    device: str = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE
    fp16: bool = False
    force: bool = False
    verify: str | None = None


# Counters and problems collected during a run.
@dataclass
class IngestResult:
    files: int = 0
    records_read: int = 0
    records_built: int = 0
    embedded: int = 0
    payload_updated: int = 0
    unchanged: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    verify_hits: list[tuple[Document, float]] = field(default_factory=list)
    cancelled: bool = False

    # True when the run finished without failures or cancellation.
    @property
    def ok(self) -> bool:
        return not self.failures and not self.cancelled


# Default logger: info to stdout, warnings/errors to stderr.
def console_log(message: str, level: str = "info") -> None:
    print(message, file=sys.stdout if level == "info" else sys.stderr, flush=True)


# --------------------------------------------------------------------------- sparse embeddings
# Keep the top-weighted sparse terms above a weight floor (always at least one).
def prune_sparse_vector(sparse_vec: models.SparseVector, min_weight: float = MIN_SPARSE_WEIGHT,
                        max_terms: int = MAX_SPARSE_TERMS) -> models.SparseVector:
    pairs = list(zip(sparse_vec.indices, sparse_vec.values))
    valid = sorted((p for p in pairs if p[1] >= min_weight), key=lambda p: p[1], reverse=True)[:max_terms]
    if not valid and pairs:
        valid = [max(pairs, key=lambda p: p[1])]
    return models.SparseVector(indices=[i for i, _ in valid], values=[v for _, v in valid])


# BM25 embedder that keeps full document vectors but prunes query vectors.
class PrunedFastEmbedSparse(FastEmbedSparse):

    # Embed a query and prune it to its strongest terms.
    def embed_query(self, text: str) -> models.SparseVector:
        return prune_sparse_vector(super().embed_query(text))


# --------------------------------------------------------------------------- record building
# Collapse all whitespace runs into single spaces.
def clean_text(value) -> str:
    return " ".join(value.split()) if value else ""


# ["Last", "First", "Suffix"] -> 'Last, First, Suffix'.
def format_author(parts) -> str:
    last, first, suffix = ([clean_text(p) for p in parts] + ["", ""])[:3]
    name = f"{last}, {first}" if first else last
    return f"{name}, {suffix}" if suffix else name


# Normalize a name for matching (strip LaTeX/accents/punctuation, lowercase); use on query side too.
def normalize_name(value: str) -> str:
    s = LATEX_ACCENT_RE.sub(r"\1", value or "")
    for latex, plain in LATEX_LETTERS.items():
        s = re.sub(re.escape(latex) + r"(?![A-Za-z])", plain, s)
    s = re.sub(r"\\[A-Za-z]+", " ", s).replace("{", "").replace("}", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("'", "").replace("\u2019", "")
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    return " ".join(s.split()).strip("-")


# Split a plain 'A B, C D and E F (Affil)' author string into [[last, first, ""], ...].
def parse_author_string(authors: str) -> list[list[str]]:
    text = re.sub(r"\([^)]*\)", " ", authors or "")
    parts = []
    for name in re.split(r",|;|\band\b|&", text):
        tokens = clean_text(name).split()
        if tokens:
            parts.append([tokens[-1], " ".join(tokens[:-1]), ""])
    return parts


# Build normalized (last_names, 'last_initial' keys) for author search.
def author_search_fields(parsed: list[list[str]]) -> tuple[list[str], list[str]]:
    last_names, keys = [], []
    for p in parsed[:MAX_AUTHOR_KEYS]:
        p = list(p) + ["", ""]
        last, first = normalize_name(p[0]), normalize_name(p[1])
        if not last:
            continue
        if last not in last_names:
            last_names.append(last)
        if first and f"{last}_{first[0]}" not in keys:
            keys.append(f"{last}_{first[0]}")
    return last_names, keys


# Parse an RFC-2822 date string to a date, or None.
def parse_created(value):
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return None


# Convert one raw arXiv record into {id, metadata, main_body}; None if unusable.
def build_record(rec: dict, max_authors: int = MAX_AUTHORS, flat_lists: bool = False) -> dict | None:
    arxiv_id = clean_text(rec.get("id"))
    title = clean_text(rec.get("title"))
    abstract = clean_text(rec.get("abstract"))
    if not arxiv_id or not (title or abstract):
        return None

    text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)

    parsed = [a for a in (rec.get("authors_parsed") or []) if a and any(clean_text(x) for x in a)]
    if not parsed and rec.get("authors"):
        parsed = parse_author_string(rec["authors"])
    authors = [format_author(a) for a in parsed]
    author_last_names, author_keys = author_search_fields(parsed)

    categories = (rec.get("categories") or "").split()
    cs_categories = [c for c in categories if c.startswith("cs.")]

    versions = rec.get("versions") or []
    published = parse_created(versions[0].get("created")) if versions else None

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
        "latest_version": versions[-1].get("version") if versions else None,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "doi": clean_text(rec.get("doi")),
        "journal_ref": clean_text(rec.get("journal-ref")),
    }

    if flat_lists:  # author_last_names / author_keys stay lists for keyword matching
        metadata["authors"] = "; ".join(metadata["authors"])
        metadata["categories"] = " ".join(categories)
        metadata["cs_categories"] = " ".join(cs_categories)

    metadata = {k: v for k, v in metadata.items() if v is not None and v != "" and v != []}
    return {"id": arxiv_id.replace("/", "_"), "metadata": metadata, "main_body": {"text": text}}


# --------------------------------------------------------------------------- files
# Yield records from a JSON array or JSONL file (None for bad JSONL lines).
def iter_records(path):
    with open(path, encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return
            if line.strip():
                is_array = line.lstrip().startswith("[")
                f.seek(pos)
                break
        if is_array:
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


# Expand folders / files / globs into a sorted list of input files (excluding -rag outputs).
def find_inputs(specs: list[str]) -> list[str]:
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


# Read a file into built records; returns (records, lines_read, skipped_by_reason).
def read_file(src, max_authors: int = MAX_AUTHORS, flat_lists: bool = False):
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


# Write built records to '<stem>-rag.jsonl' and return its path.
def write_file(src, objects: list[dict], out_dir: str | None = None) -> str:
    folder = os.path.abspath(out_dir) if out_dir else os.path.dirname(os.path.abspath(src))
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(Path(folder).parent, f"{stem}{OUTPUT_SUFFIX}.jsonl")
    with open(dst, "w", encoding="utf-8") as out:
        for obj in objects:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return dst


# --------------------------------------------------------------------------- embeddings
# Pick 'cuda' or 'cpu' for 'auto'; fail if CUDA is requested but unavailable.
def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{requested}' requested but no CUDA GPU is visible.")
    return requested


# Load (once) the dense HuggingFace embedding model.
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
        query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_PREFIX},
    )


# Load (once) the BM25 sparse embedding model.
@functools.lru_cache(maxsize=1)
def get_sparse_embeddings():
    return PrunedFastEmbedSparse(model_name=SPARSE_MODEL_ID)


# Fingerprint of model/pipeline settings; changing it forces re-embedding.
def pipeline_salt() -> str:
    return (f"{PIPELINE_VERSION}|{MODEL_ID}|{MODEL_REVISION}|{SPARSE_MODEL_ID}"
            f"|min_w={MIN_SPARSE_WEIGHT},max_k={MAX_SPARSE_TERMS}")


# --------------------------------------------------------------------------- hashing / change detection
# Deterministic Qdrant point UUID for a record id.
def point_id(record_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


# Record metadata plus the source file name.
def base_metadata(item: dict, src) -> dict:
    return {**item["metadata"], "source_file": os.path.basename(str(src))}


# Hash of everything that affects the vectors.
def text_hash(item: dict, salt: str) -> str:
    meta = item["metadata"]
    payload = json.dumps([item["main_body"]["text"], meta.get("updated"), meta.get("latest_version"), salt],
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Hash of the metadata payload.
def meta_hash(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# Bundle a record with its (text_hash, meta_hash).
def hash_entry(item: dict, src, salt: str) -> tuple:
    return item, text_hash(item, salt), meta_hash(base_metadata(item, src))


# Fetch stored (text_hash, meta_hash) for the given point ids.
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


# Split records into (to_embed, metadata_only, unchanged_count) vs. what is stored.
def classify(client: QdrantClient, collection_name: str, objects: list[dict], src, salt: str):
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


# --------------------------------------------------------------------------- qdrant
# Folder of the embedded local Qdrant database.
def db_path(config) -> Path:
    if config.db_dir:
        return Path(config.db_dir)
    return (Path(config.out_dir) if config.out_dir else Path.cwd()) / "qdrant_db"


# Connect to a Qdrant server (if URL given) or open the embedded local store.
def open_client(config) -> QdrantClient:
    if config.qdrant_url:
        return QdrantClient(url=config.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(path=str(db_path(config)))


# Build a hybrid (dense + sparse) LangChain vector store.
def make_store(client: QdrantClient, collection_name: str, embeddings, sparse_embeddings) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


# Create the hybrid collection (HNSW paused for bulk load) if it does not exist.
def ensure_collection(client: QdrantClient, collection_name: str, embeddings, remote: bool,
                      log: LogFn = console_log) -> None:
    if client.collection_exists(collection_name):
        return
    size = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=size, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            # IDF modifier: Qdrant/bm25 stores only term frequencies, the server adds IDF
            SPARSE_VECTOR_NAME: models.SparseVectorParams(index=models.SparseIndexParams(),
                                                          modifier=models.Modifier.IDF)
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    log(f"Created collection '{collection_name}' (dense indexing paused during the load).", "info")
    if remote:
        for field_name, schema in PAYLOAD_INDEXES.items():
            client.create_payload_index(collection_name, field_name=field_name, field_schema=schema)


# Switch HNSW indexing back on after the bulk load (never raises).
def restore_indexing(client: QdrantClient, collection_name: str, log: LogFn = console_log) -> None:
    try:
        if not client.collection_exists(collection_name):
            return
        client.update_collection(
            collection_name,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD_ON),
        )
        log(f"Indexing switched back on for '{collection_name}'; wait for green status before searching.", "info")
    except Exception as e:
        log(f"WARNING: could not switch indexing back on ({type(e).__name__}: {e}). "
            f"Set indexing_threshold={INDEXING_THRESHOLD_ON} for '{collection_name}' manually.", "warning")


# Turn hashed entries into LangChain Documents with stable ids.
def load_documents(entries: list[tuple], src) -> list[Document]:
    return [
        Document(page_content=item["main_body"]["text"],
                 metadata={**base_metadata(item, src), "text_hash": th, "meta_hash": mh},
                 id=point_id(item["id"]))
        for item, th, mh in entries
    ]


# Overwrite payloads (vectors unchanged) in batches.
def update_payloads(client: QdrantClient, collection_name: str, entries: list[tuple], src) -> None:
    for i in range(0, len(entries), PAYLOAD_UPDATE_BATCH):
        operations = [
            models.OverwritePayloadOperation(overwrite_payload=models.SetPayload(
                payload={"page_content": item["main_body"]["text"],
                         "metadata": {**base_metadata(item, src), "text_hash": th, "meta_hash": mh}},
                points=[point_id(item["id"])],
            ))
            for item, th, mh in entries[i:i + PAYLOAD_UPDATE_BATCH]
        ]
        client.batch_update_points(collection_name, update_operations=operations)


# Reopen the store and run a hybrid search; returns the (doc, score) hits.
def verify_persistence(config, query: str, k: int = 2, log: LogFn = console_log) -> list[tuple[Document, float]]:
    client = open_client(config)
    try:
        if not client.collection_exists(config.collection):
            raise RuntimeError(f"Collection '{config.collection}' not found after reopening.")
        embeddings = get_embeddings(resolve_device(config.device), config.batch_size, config.fp16)
        store = make_store(client, config.collection, embeddings, get_sparse_embeddings())
        log(f"Verify: {client.get_collection(config.collection).points_count} point(s) stored; "
            f"searching '{query}'", "info")
        hits = store.similarity_search_with_score(query, k=k)
        for rank, (doc, score) in enumerate(hits, start=1):
            log(f" [{rank}] {score:.4f} | {doc.metadata.get('arxiv_id', doc.id)} | "
                f"{doc.page_content.strip()[:200]}", "info")
        return hits
    finally:
        client.close()


# --------------------------------------------------------------------------- pipeline
# Holds the Qdrant client and lazily-created store for one run.
class _Ingestor:

    # Store run settings; the embedding models are loaded on first use.
    def __init__(self, config: IngestConfig, client: QdrantClient, log: LogFn):
        self.config, self.client, self.log = config, client, log
        self.device = resolve_device(config.device)
        self.salt = pipeline_salt()
        self.store: QdrantVectorStore | None = None

    # Load models, create the collection if needed, and return the store.
    def get_store(self) -> QdrantVectorStore:
        if self.store is None:
            embeddings = get_embeddings(self.device, self.config.batch_size, self.config.fp16)
            ensure_collection(self.client, self.config.collection, embeddings, bool(self.config.qdrant_url), self.log)
            self.store = make_store(self.client, self.config.collection, embeddings, get_sparse_embeddings())
        return self.store

    # Embed new/changed records and patch metadata-only changes; returns (embedded, payload, unchanged).
    def ingest(self, objects: list[dict], src) -> tuple[int, int, int]:
        if self.config.force:
            to_embed, meta_only = [hash_entry(o, src, self.salt) for o in objects], []
        else:
            to_embed, meta_only, _ = classify(self.client, self.config.collection, objects, src, self.salt)
        if to_embed:
            docs = load_documents(to_embed, src)
            self.log(f"  adding {len(docs):,} document(s) to '{self.config.collection}'", "info")
            self.get_store().add_documents(docs, ids=[d.id for d in docs], batch_size=self.config.batch_size)
        update_payloads(self.client, self.config.collection, meta_only, src)
        return len(to_embed), len(meta_only), len(objects) - len(to_embed) - len(meta_only)


# Read, ingest and (optionally) write one input file, updating result in place.
def process_file(src: str, ingestor: _Ingestor, result: IngestResult, log: LogFn) -> None:
    config = ingestor.config
    name = os.path.basename(src)
    try:
        objects, records_read, skipped = read_file(src, config.max_authors, config.flat_lists)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        log(f"  {name}: FAILED to read ({type(e).__name__}: {e})", "error")
        result.failures.append((name, f"could not read: {type(e).__name__}: {e}"))
        return

    for reason, count in skipped.items():
        result.skipped[reason] = result.skipped.get(reason, 0) + count
    if records_read and not objects:
        log(f"  {name}: WARNING no usable records in this file", "warning")

    embedded = payload_only = unchanged = 0
    if objects:
        try:
            embedded, payload_only, unchanged = ingestor.ingest(objects, src)
        except Exception as e:
            log(f"  {name}: FAILED Qdrant ingestion ({type(e).__name__}: {e})", "error")
            result.failures.append((name, f"ingestion failed: {type(e).__name__}: {e}"))

    if not config.no_write:
        try:
            log(f"  {name}: output written to {write_file(src, objects, config.out_dir)}", "info")
        except OSError as e:
            log(f"  {name}: FAILED to write output file ({type(e).__name__}: {e})", "error")
            result.failures.append((name, f"could not write output: {type(e).__name__}: {e}"))

    result.records_read += records_read
    result.records_built += len(objects)
    result.embedded += embedded
    result.payload_updated += payload_only
    result.unchanged += unchanged

    parts = [f"{unchanged:,} unchanged"] if unchanged else []
    if payload_only:
        parts.append(f"{payload_only:,} payload-only updates")
    parts += [f"{c:,} skipped: {r}" for r, c in skipped.items() if c]
    note = f" ({', '.join(parts)})" if parts else ""
    log(f"  {name}: {len(objects):,} of {records_read:,} records processed{note}", "info")


# Run the full pipeline for a config; GUI entry point (no prints/exits, uses callbacks).
def run_ingestion(config: IngestConfig, log: LogFn = console_log, progress: ProgressFn | None = None,
                  should_stop: Callable[[], bool] | None = None) -> IngestResult:
    result = IngestResult()
    files = find_inputs(config.inputs)
    if not files:
        result.failures.append(("inputs", "no input files found for: " + " ".join(config.inputs)))
        log(result.failures[-1][1], "error")
        return result
    result.files = len(files)

    if not config.qdrant_url:
        log("Note: using embedded local Qdrant store.", "info")

    client = open_client(config)
    try:
        ingestor = _Ingestor(config, client, log)
        for i, src in enumerate(files):
            if should_stop and should_stop():
                result.cancelled = True
                log("Cancelled by user.", "warning")
                break
            if progress:
                progress(i, len(files), os.path.basename(src))
            process_file(src, ingestor, result, log)
        if progress:
            progress(len(files), len(files), "")
    finally:
        restore_indexing(client, config.collection, log)
        client.close()

    if config.verify and not result.cancelled:
        if result.embedded or result.payload_updated or result.unchanged:
            try:
                result.verify_hits = verify_persistence(config, config.verify, log=log)
            except Exception as e:
                log(f"FAILED verification ({type(e).__name__}: {e})", "error")
                result.failures.append(("verify", f"{type(e).__name__}: {e}"))
        else:
            log("Skipping verification: nothing in the store.", "info")
    return result


# Human-readable end-of-run summary.
def format_summary(result: IngestResult) -> str:
    lines = [f"Done: {result.records_built:,} of {result.records_read:,} records from {result.files} file(s); "
             f"{result.embedded:,} embedded, {result.payload_updated:,} payload-only updates, "
             f"{result.unchanged:,} unchanged"]
    if any(result.skipped.values()):
        lines.append("Records skipped: " + ", ".join(f"{c:,} {r}" for r, c in result.skipped.items() if c))
    if result.failures:
        lines.append(f"{len(result.failures)} problem(s) - re-run to retry (stored papers are skipped):")
        lines += [f"  - {name}: {reason}" for name, reason in result.failures]
    elif not result.cancelled:
        lines.append("Completed without errors.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli
# Parse command-line arguments into an IngestConfig.
def parse_args(argv: list[str]) -> IngestConfig:
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
    return IngestConfig(**vars(p.parse_args(argv)))


# CLI entry point: parse args, run, print summary, return exit code.
def main(argv: list[str] | None = None) -> int:
    result = run_ingestion(parse_args(sys.argv[1:] if argv is None else argv))
    console_log("\n" + format_summary(result), "info" if result.ok else "error")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
