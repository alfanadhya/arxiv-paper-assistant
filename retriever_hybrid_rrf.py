#!/usr/bin/env python3
"""
Retrieve arXiv papers using Native Hybrid Search (Dense + Sparse) from Qdrant with Reciprocal Rank Fusion (RRF).

Collection layout expected:
  * Named dense vector: default "dense" (BAAI/bge-base-en-v1.5, 768 dimensions)
  * Named sparse vector: default "sparse" (FastEmbedSparse / Qdrant/bm25)

Examples:
    python retriver_hybrid_rrf.py "graph decompositions and sparsity" --qdrant-path ./qdrant_db
    python retriver_hybrid_rrf.py "diffusion models" --category cs.CV --year 2023 --rerank
    python retriver_hybrid_rrf.py --cases retrieval_cases.json --k-retrieve 10 --k-final 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence
from qdrant_client import models
ROOT = Path(__file__).resolve().parent
MODELS_CACHE_DIR = ROOT / "models_cache"

DEFAULT_COLLECTION_NAME = "sebun_bge768_hybrid_full"
QDRANT_PATH = ROOT / "qdrant_db"
MODEL_ID = "BAAI/bge-base-en-v1.5"                            # must match db_generator_hybrid_gpu.py
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
SPARSE_MODEL_ID = "Qdrant/bm25"

# Sparse Vector Pruning Thresholds
MIN_SPARSE_WEIGHT = 0.30   # Minimum BM25 weight floor to drop noise
MAX_SPARSE_TERMS = 35      # Hard ceiling for maximum query keyword terms

DEFAULT_DENSE_VECTOR_NAME = "dense"
DEFAULT_SPARSE_VECTOR_NAME = "sparse"

VECTOR_SIZE = 768
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANKER_ID = "reciprocal-rank-fusion-rrf-v1"
REWRITE_POLICY_ID = "case-supplied-rewrite-v1"
RERANKER_DESCRIPTION = "native Qdrant server-side reciprocal rank fusion (RRF) score merging"

CATEGORY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z-]*(\.[A-Za-z-]+)?$")
MIN_YEAR, MAX_YEAR = 1991, 2100


# --------------------------------------------------------------------------- Sparse Pruning & Local Loader
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


class PrunedFastEmbedSparse:
    """Wrapper that handles auto-download/local loading and weight pruning."""

    def __init__(self, model_name: str = SPARSE_MODEL_ID, cache_dir: Path | str = MODELS_CACHE_DIR):
        from langchain_qdrant import FastEmbedSparse

        cache_path = str(cache_dir)
        os.makedirs(cache_path, exist_ok=True)
        try:
            # Try offline loading first if model files are already cached locally
            self._sparse = FastEmbedSparse(
                model_name=model_name,
                cache_dir=cache_path,
                local_files_only=True
            )
        except Exception:
            # Download model from Hugging Face on the first run
            print(f"Downloading sparse model '{model_name}' to local cache ({cache_path})...")
            self._sparse = FastEmbedSparse(
                model_name=model_name,
                cache_dir=cache_path,
                local_files_only=False
            )

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        vectors = self._sparse.embed_documents(texts)
        return [prune_sparse_vector(v) for v in vectors]

    def embed_query(self, text: str) -> models.SparseVector:
        if not isinstance(text, str) or not text.strip():
            text = "empty query"
        vector = self._sparse.embed_query(text)
        return prune_sparse_vector(vector)


# --------------------------------------------------------------------------- Data Models
@dataclass(frozen=True)
class MetadataIntent:
    """A validated, typed filter request."""

    primary_category: str | None = None
    category: str | None = None
    cs_primary_only: bool = False
    year_from: int | None = None
    year_to: int | None = None
    # Author conditions, all required (one per author): (payload field, accepted values), e.g.
    # ("author_keys", ("chen_m",)) or ("author_last_names", ("maaten", "der maaten", "van der maaten")).
    authors: tuple[tuple[str, tuple[str, ...]], ...] = ()

    ALLOWED = ("primary_category", "category", "cs_primary_only", "year", "year_from", "year_to")

    @classmethod
    def parse(cls, raw: Mapping[str, object] | None) -> "MetadataIntent":
        if raw is None:
            return cls()
        unknown = sorted(set(raw) - set(cls.ALLOWED))
        if unknown:
            raise ValueError(
                "unsupported filter key(s): " + ", ".join(unknown) + "; allowed: " + ", ".join(cls.ALLOWED)
            )

        def code(key: str) -> str | None:
            value = raw.get(key)
            if value is None:
                return None
            if not isinstance(value, str) or not CATEGORY_PATTERN.fullmatch(value.strip()):
                raise ValueError(f"{key} must look like 'cs.CL' or 'quant-ph'")
            return value.strip()

        def year(key: str) -> int | None:
            value = raw.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or not MIN_YEAR <= value <= MAX_YEAR:
                raise ValueError(f"{key} must be an integer between {MIN_YEAR} and {MAX_YEAR}")
            return value

        flag = raw.get("cs_primary_only", False)
        if not isinstance(flag, bool):
            raise ValueError("cs_primary_only must be true or false")

        year_from, year_to, exact = year("year_from"), year("year_to"), year("year")
        if exact is not None:
            if year_from is not None or year_to is not None:
                raise ValueError("use either year or year_from/year_to, not both")
            year_from = year_to = exact
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("year_from must not be after year_to")
        return cls(code("primary_category"), code("category"), flag, year_from, year_to)

    def to_filter(self, prefix: str = "metadata."):
        

        must = []
        if self.primary_category:
            must.append(models.FieldCondition(
                key=f"{prefix}primary_category", match=models.MatchValue(value=self.primary_category)))
        if self.category:
            must.append(models.FieldCondition(
                key=f"{prefix}categories", match=models.MatchValue(value=self.category)))
        if self.cs_primary_only:
            must.append(models.FieldCondition(
                key=f"{prefix}is_cs_primary", match=models.MatchValue(value=True)))
        if self.year_from is not None or self.year_to is not None:
            must.append(models.FieldCondition(
                key=f"{prefix}year", range=models.Range(gte=self.year_from, lte=self.year_to)))
        for field_name, values in self.authors:   # each author must appear (any of its name variants)
            must.append(models.FieldCondition(
                key=f"{prefix}{field_name}", match=models.MatchAny(any=list(values))))
        return models.Filter(must=must) if must else None


@dataclass(frozen=True)
class Evidence:
    initial_rank: int
    vector_score: float
    point_id: str
    arxiv_id: str
    title: str
    url: str
    metadata: dict
    text: str
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class RankedEvidence:
    initial_rank: int
    final_rank: int
    rrf_score: float
    point_id: str
    arxiv_id: str
    title: str
    url: str
    metadata: dict
    text: str
    rerank_reason: str


@dataclass(frozen=True)
class SearchCase:
    case_id: str
    query: str
    filters: Mapping[str, object] | None = None
    rewritten_query: str | None = None
    expected_arxiv_ids: tuple[str, ...] = ()
    expect_empty: bool = False
    expect_rejected: bool = False
    purpose: str = ""


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    policy_id: str
    applied: bool
    reason: str


class Retriever(Protocol):
    provider: str
    index_mode: str
    collection: str

    def retrieve(self, query: str, k_retrieve: int, metadata: MetadataIntent) -> list[Evidence]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- RRF Reranking & Decision Tracking
def rrf_rerank(evidence: Sequence[Evidence]) -> list[RankedEvidence]:
    """Reranks candidates strictly by Reciprocal Rank Fusion (RRF) score."""
    scored = sorted(evidence, key=lambda item: -item.vector_score)
    ranked_evidence = []

    for final_rank, item in enumerate(scored, start=1):
        rank_change = item.initial_rank - final_rank
        dense_str = f"dense={item.dense_rank}" if item.dense_rank is not None else "dense=N/A"
        sparse_str = f"sparse={item.sparse_rank}" if item.sparse_rank is not None else "sparse=N/A"

        reason = (
            f"rrf rank changed: initial_rank={item.initial_rank} -> final_rank={final_rank} "
            f"(shift={rank_change:+d}; {dense_str}, {sparse_str}, rrf_score={item.vector_score:.4f})"
        )

        ranked_evidence.append(RankedEvidence(
            initial_rank=item.initial_rank,
            final_rank=final_rank,
            rrf_score=item.vector_score,
            point_id=item.point_id,
            arxiv_id=item.arxiv_id,
            title=item.title,
            url=item.url,
            metadata=item.metadata,
            text=item.text,
            rerank_reason=reason,
        ))
    return ranked_evidence


def relevance_gate(ranked: Sequence[RankedEvidence], threshold: float) -> dict[str, object]:
    """Applies an abstention decision based on top candidate's fused RRF score."""
    observed = ranked[0].rrf_score if ranked else None
    passed = bool(ranked) and observed is not None and observed >= threshold
    reason = (
        "no candidate survived retrieval and metadata filtering"
        if not ranked
        else f"top RRF score {observed:.4f} {'>=' if passed else '<'} {threshold:.4f}"
    )
    return {"passed": passed, "threshold": threshold, "observed_top_rrf_score": observed, "reason": reason}


def track_rerank_decisions(reranked: Sequence[RankedEvidence]) -> list[dict[str, object]]:
    """Generates structured log traces of all rank shifts and RRF decisions."""
    return [
        {
            "point_id": item.point_id,
            "arxiv_id": item.arxiv_id,
            "initial_rank": item.initial_rank,
            "final_rank": item.final_rank,
            "rank_shift": item.initial_rank - item.final_rank,
            "rrf_score": item.rrf_score,
            "rerank_reason": item.rerank_reason,
        }
        for item in reranked
    ]


# --------------------------------------------------------------------------- Payload Detection
def split_payload(payload: Mapping[str, object]) -> tuple[str, dict, str | None]:
    nested = payload.get("metadata")
    if isinstance(nested, dict):
        prefix, metadata = "metadata.", nested
    else:
        prefix = ""
        metadata = {k: v for k, v in payload.items() if k not in ("text", "page_content", "main_body")}
    body = payload.get("main_body")
    text = payload.get("text") or payload.get("page_content")
    if text is None and isinstance(body, dict):
        text = body.get("text")
    return prefix, metadata, (str(text) if text is not None else None)


# --------------------------------------------------------------------------- Qdrant Hybrid Retriever
class QdrantHybridRetriever:
    def __init__(
        self, client, dense_encoder, sparse_encoder, provider: str,
        collection: str = DEFAULT_COLLECTION_NAME,
        dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
        sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
        index_mode: str = "Qdrant Hybrid",
    ) -> None:
        self.client = client
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.provider = provider
        self.index_mode = index_mode
        self.collection = collection
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.points_count = None
        self._check_collection()
        self.prefix = self._detect_layout()

    def _check_collection(self) -> None:
        try:
            info = self.client.get_collection(self.collection)
        except Exception as exc:
            raise SystemExit(f"Cannot open collection {self.collection!r}: {exc}") from exc
        self.points_count = getattr(info, "points_count", None)

    def _detect_layout(self) -> str:
        points, _ = self.client.scroll(collection_name=self.collection, limit=1, with_payload=True)
        if not points:
            raise SystemExit(f"Collection {self.collection!r} is empty")
        payload = points[0].payload or {}
        prefix, _, text = split_payload(payload)
        if text is None:
            raise SystemExit("Stored payload contains no readable text content.")
        return prefix

    def retrieve(self, query: str, k_retrieve: int, metadata: MetadataIntent) -> list[Evidence]:
        from qdrant_client import models

        # 1. Generate Query Embeddings and Apply Threshold Pruning
        dense_vec = self.dense_encoder.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0].tolist()
        sparse_vec = self.sparse_encoder.embed_query(query)

        query_filter = metadata.to_filter(self.prefix)
        prefetch_limit = max(50, k_retrieve * 2)

        # 2. Fetch Dense & Sparse Sub-Ranks for Decision Logging
        dense_hits = self.client.query_points(
            collection_name=self.collection,
            query=dense_vec,
            using=self.dense_vector_name,
            query_filter=query_filter,
            limit=prefetch_limit,
            with_payload=False,
        ).points
        dense_ranks = {str(pt.id): r for r, pt in enumerate(dense_hits, start=1)}

        sparse_hits = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vec,
            using=self.sparse_vector_name,
            query_filter=query_filter,
            limit=prefetch_limit,
            with_payload=False,
        ).points
        sparse_ranks = {str(pt.id): r for r, pt in enumerate(sparse_hits, start=1)}

        # 3. Server-side Native RRF Fusion Query
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=self.dense_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=self.sparse_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k_retrieve,
            with_payload=True,
        )

        evidence = []
        for rank, point in enumerate(result.points, start=1):
            _, meta, text = split_payload(point.payload or {})
            if text is None:
                continue
            arxiv_id = str(meta.get("arxiv_id", ""))
            pid = str(point.id)
            evidence.append(Evidence(
                initial_rank=rank,
                vector_score=float(point.score),
                point_id=pid,
                arxiv_id=arxiv_id,
                title=str(meta.get("title") or text.split("\n", 1)[0]),
                url=str(meta.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")),
                metadata=meta,
                text=text,
                dense_rank=dense_ranks.get(pid),
                sparse_rank=sparse_ranks.get(pid),
            ))
        return evidence

    def close(self) -> None:
        self.client.close()


def build_retriever(
    qdrant_path: str | None = None,
    qdrant_url: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
    sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
    model_revision: str = MODEL_REVISION,
) -> QdrantHybridRetriever:
    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install qdrant-client sentence-transformers langchain-qdrant fastembed") from exc

    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
        index_mode = f"Qdrant server hybrid ({qdrant_url})"
    else:
        path = Path(qdrant_path) if qdrant_path else QDRANT_PATH
        if not path.exists():
            raise SystemExit(f"Qdrant folder not found: {path}\nPass --qdrant-path or --qdrant-url.")
        client = QdrantClient(path=str(path))
        index_mode = f"Qdrant local persistent hybrid ({path})"

    # Auto-downloads on first run to ./models_cache, then reuses local files
    os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
    try:
        # Load from local cache only (no Hugging Face calls) if already downloaded
        dense_encoder = SentenceTransformer(
            MODEL_ID,
            revision=model_revision or None,
            cache_folder=str(MODELS_CACHE_DIR),
            local_files_only=True,
        )
    except Exception:
        # Not in cache yet: download from Hugging Face on the first run
        print(f"Downloading dense model '{MODEL_ID}' to local cache ({MODELS_CACHE_DIR})...")
        dense_encoder = SentenceTransformer(
            MODEL_ID,
            revision=model_revision or None,
            cache_folder=str(MODELS_CACHE_DIR),
            local_files_only=False,
        )
    sparse_encoder = PrunedFastEmbedSparse(
        model_name=SPARSE_MODEL_ID,
        cache_dir=MODELS_CACHE_DIR
    )
    provider = f"Dense({MODEL_ID}) + Sparse({SPARSE_MODEL_ID}, min_w={MIN_SPARSE_WEIGHT}, max_k={MAX_SPARSE_TERMS})"

    return QdrantHybridRetriever(
        client, dense_encoder, sparse_encoder, provider,
        collection=collection_name, dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name, index_mode=index_mode
    )


# --------------------------------------------------------------------------- Search & Execution Logic
def search(retriever: Retriever, query: str, metadata: MetadataIntent, k_retrieve: int, k_final: int,
           rerank: bool = True, threshold: float | None = None) -> dict[str, object]:
    initial = retriever.retrieve(query, k_retrieve, metadata)
    reranked = rrf_rerank(initial) if rerank else [
        RankedEvidence(
            initial_rank=item.initial_rank,
            final_rank=item.initial_rank,
            rrf_score=item.vector_score,
            point_id=item.point_id,
            arxiv_id=item.arxiv_id,
            title=item.title,
            url=item.url,
            metadata=item.metadata,
            text=item.text,
            rerank_reason="retrieval order retained; no RRF reranker applied",
        )
        for item in initial
    ]
    final = reranked[:k_final]

    return {
        "query": query,
        "provider": retriever.provider,
        "collection": retriever.collection,
        "metadata_intent": asdict(metadata),
        "reranker": RERANKER_ID if rerank else None,
        "k_retrieve": k_retrieve,
        "k_final": k_final,
        "retrieved_count": len(initial),
        "candidates": [asdict(item) for item in final],
        "rerank_trace": track_rerank_decisions(final),
        "gate": relevance_gate(final, threshold) if threshold is not None else None,
    }


def load_cases(path: Path) -> list[SearchCase]:
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise SystemExit(f"cases file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc

    data = json.loads(raw_text)
    items = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise SystemExit(f"{path} must hold a non-empty list of cases.")

    cases = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"query": item}
        case_id = item.get("case_id", f"case_{index}")
        cases.append(SearchCase(
            case_id=case_id.strip(),
            query=item["query"].strip(),
            filters=item.get("filters"),
            rewritten_query=item.get("rewritten_query"),
            expected_arxiv_ids=tuple(item.get("expected_arxiv_ids", ())),
            expect_empty=item.get("expect_empty", False),
            expect_rejected=item.get("expect_rejected", False),
            purpose=item.get("purpose", ""),
        ))
    return cases


def make_rewrite(case: SearchCase) -> RewriteResult:
    if case.rewritten_query and case.rewritten_query != case.query:
        return RewriteResult(case.query, case.rewritten_query, REWRITE_POLICY_ID, True, "rewritten query supplied")
    return RewriteResult(case.query, case.query, REWRITE_POLICY_ID, False, "no rewrite supplied")


def run_case(retriever: Retriever, case: SearchCase, k_retrieve: int, k_final: int, threshold: float | None) -> dict[str, object]:
    try:
        metadata = MetadataIntent.parse(case.filters)
    except ValueError as exc:
        return {
            "case_id": case.case_id,
            "status": "validation_rejected",
            "validation_error": str(exc),
            "baseline": None,
            "improved": None,
            "regression_passed": case.expect_rejected,
        }

    initial_base = retriever.retrieve(case.query, k_retrieve, MetadataIntent())
    baseline_final = rrf_rerank(initial_base)[:k_final]

    rewrite = make_rewrite(case)
    initial_imp = retriever.retrieve(rewrite.rewritten_query, k_retrieve, metadata)
    improved_final = rrf_rerank(initial_imp)[:k_final]

    baseline_top = baseline_final[0].arxiv_id if baseline_final else None
    improved_top = improved_final[0].arxiv_id if improved_final else None
    expected = set(case.expected_arxiv_ids)

    return {
        "case_id": case.case_id,
        "status": "completed",
        "validated_metadata": asdict(metadata),
        "baseline": {
            "query_used": case.query,
            "candidates": [asdict(item) for item in baseline_final],
            "gate": relevance_gate(baseline_final, threshold) if threshold is not None else None,
        },
        "improved": {
            "query_used": rewrite.rewritten_query,
            "rewrite": asdict(rewrite),
            "candidates": [asdict(item) for item in improved_final],
            "rerank_trace": track_rerank_decisions(improved_final),
            "gate": relevance_gate(improved_final, threshold) if threshold is not None else None,
        },
        "comparison": {
            "baseline_top_arxiv_id": baseline_top,
            "improved_top_arxiv_id": improved_top,
            "baseline_expected_at_top": (baseline_top in expected) if expected else None,
            "improved_expected_at_top": (improved_top in expected) if expected else None,
        },
    }


def generate_summary_report(results: Sequence[Mapping[str, object]], retriever: Retriever, cases_file: str,
                            k_retrieve: int, k_final: int, threshold: float | None) -> dict[str, object]:
    done = [r for r in results if r["status"] == "completed"]
    expecting = [r for r in done if r.get("expected_arxiv_ids")]

    return {
        "report": "hybrid rrf retrieval comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": retriever.collection,
        "cases_file": cases_file,
        "provider": retriever.provider,
        "index_mode": retriever.index_mode,
        "reranker_id": RERANKER_ID,
        "reranker_description": RERANKER_DESCRIPTION,
        "k_retrieve": k_retrieve,
        "k_final": k_final,
        "gate_threshold": threshold,
        "case_count": len(results),
        "completed_case_count": len(done),
        "summary": {
            "cases_with_expected_papers": len(expecting),
            "baseline_expected_at_top": sum(bool(r["comparison"]["baseline_expected_at_top"]) for r in expecting),
            "improved_expected_at_top": sum(bool(r["comparison"]["improved_expected_at_top"]) for r in expecting),
        },
        "cases": results,
    }


# --------------------------------------------------------------------------- CLI Main
def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="?", help="query text (or use --cases)")
    p.add_argument("--cases", help="JSON file with search cases")
    p.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME})")
    p.add_argument("--dense-vector-name", default=DEFAULT_DENSE_VECTOR_NAME, help=f"dense vector key (default: {DEFAULT_DENSE_VECTOR_NAME})")
    p.add_argument("--sparse-vector-name", default=DEFAULT_SPARSE_VECTOR_NAME, help=f"sparse vector key (default: {DEFAULT_SPARSE_VECTOR_NAME})")
    p.add_argument("--qdrant-path", help=f"local Qdrant dir (default: {QDRANT_PATH})")
    p.add_argument("--qdrant-url", help="Qdrant server URL")
    p.add_argument("--primary-category", help="primary category (e.g. cs.CL)")
    p.add_argument("--category", help="any category (e.g. cs.CG)")
    p.add_argument("--year", type=int, help="submission year")
    p.add_argument("--k-retrieve", type=int, default=20)
    p.add_argument("--k-final", type=int, default=5)
    p.add_argument("--rerank", action="store_true", help="enable RRF rerank tracking")
    p.add_argument("--gate-threshold", type=float, help="abstention threshold based on RRF score")
    p.add_argument("--output", help="output file path for JSON report")

    args = p.parse_args(argv)

    if (args.query is None) == (args.cases is None):
        raise SystemExit("Specify either a query or --cases FILE, but not both.")

    # Lazy execution: validate inputs before loading models into VRAM
    if args.cases:
        cases_path = Path(args.cases)
        cases = load_cases(cases_path)
        retriever = build_retriever(
            args.qdrant_path, args.qdrant_url, args.collection,
            args.dense_vector_name, args.sparse_vector_name
        )
        try:
            results = [run_case(retriever, case, args.k_retrieve, args.k_final, args.gate_threshold) for case in cases]
            report = generate_summary_report(results, retriever, str(cases_path), args.k_retrieve, args.k_final, args.gate_threshold)
            output_path = Path(args.output) if args.output else cases_path.with_name(cases_path.stem + "-hybrid-rrf-report.json")
            output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Hybrid RRF retrieval report saved to {output_path}")
        finally:
            retriever.close()
    else:
        raw = {"primary_category": args.primary_category, "category": args.category, "year": args.year}
        metadata = MetadataIntent.parse({k: v for k, v in raw.items() if v is not None})
        retriever = build_retriever(
            args.qdrant_path, args.qdrant_url, args.collection,
            args.dense_vector_name, args.sparse_vector_name
        )
        try:
            report = search(retriever, args.query, metadata, args.k_retrieve, args.k_final, args.rerank, args.gate_threshold)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            if args.output:
                Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        finally:
            retriever.close()


if __name__ == "__main__":
    main()