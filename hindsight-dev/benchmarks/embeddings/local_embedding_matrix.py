"""Benchmark local SentenceTransformers embedding model/dimension candidates.

This is intentionally offline from the Hindsight runtime: it does not connect
to Postgres, mutate indexes, or activate a model. It loads candidate local
models, applies the same prompt/truncation knobs used by LocalSTEmbeddings, and
prints enough timing plus tiny retrieval-sanity metrics to decide what deserves
a full reprocess trial.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from rich.console import Console
from rich.table import Table

PGVECTOR_HNSW_MAX_DIM = 2000

console = Console()


@dataclass(frozen=True)
class QueryExample:
    text: str
    positive_document_index: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    dimensions: tuple[int, ...]
    trust_remote_code: bool = False
    query_prompt_name: str | None = None
    query_prompt: str | None = None
    document_prompt_name: str | None = None
    document_prompt: str | None = None
    note: str = ""


@dataclass(frozen=True)
class BenchmarkCase:
    spec: ModelSpec
    dimension: int

    @property
    def pgvector_hnsw_safe(self) -> bool:
        return self.dimension <= PGVECTOR_HNSW_MAX_DIM


@dataclass(frozen=True)
class EncodeTiming:
    seconds: float
    vectors_per_second: float


@dataclass(frozen=True)
class QualityMetrics:
    top1: float
    mrr: float
    average_positive_margin: float


@dataclass(frozen=True)
class BenchmarkResult:
    key: str
    model_name: str
    dimension: int
    native_dimension: int | None
    pgvector_hnsw_safe: bool
    trust_remote_code: bool
    query_prompt_name: str | None
    query_prompt: str | None
    document_prompt_name: str | None
    document_prompt: str | None
    normalize_embeddings: bool
    profile_fingerprint: str | None
    load_seconds: float | None
    document_seconds: float | None
    document_vectors_per_second: float | None
    query_seconds: float | None
    query_vectors_per_second: float | None
    top1: float | None
    mrr: float | None
    average_positive_margin: float | None
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkRun:
    timestamp: str
    repeat: int
    batch_size: int
    device: str | None
    normalize_embeddings: bool
    local_files_only: bool
    max_dimension: int
    results: list[BenchmarkResult]


DOCUMENT_EXAMPLES = [
    "Alice owns the local Hindsight runtime upgrade and keeps the fork changes source-controlled.",
    "The recall path embeds user search queries before semantic retrieval and reranking.",
    "Retain extraction stores concise distilled memory units with source and temporal metadata.",
    "The database uses pgvector HNSW indexes and rejects dimensions above 2000 in the default strategy.",
    "Codex OAuth routing should use the local codex subscription path when API access is available.",
    "Reflect runs with higher reasoning effort because synthesis quality matters more than speed.",
    "Consolidation merges related retained facts into mental models after ingestion catches up.",
    "The current local reranker was selected as a speed and quality compromise for this machine.",
]

QUERY_EXAMPLES = [
    QueryExample("Who owns the local Hindsight fork upgrade?", 0),
    QueryExample("Where are user recall queries embedded?", 1),
    QueryExample("What kind of facts does retain store?", 2),
    QueryExample("What embedding width is safe for the default HNSW setup?", 3),
    QueryExample("Which path should use the Codex subscription?", 4),
    QueryExample("Why should reflect use higher thinking effort?", 5),
]


def build_model_specs(include_curiosities: bool, include_over_limit: bool) -> list[ModelSpec]:
    snowflake_dimensions = (1024, 256) if include_curiosities else (1024,)
    voyage_dimensions = (1024, 2000, 2048) if include_over_limit else (1024, 2000)

    return [
        ModelSpec(
            key="qwen3-embedding-4b",
            model_name="Qwen/Qwen3-Embedding-4B",
            dimensions=(1024, 2000),
            trust_remote_code=True,
            query_prompt_name="query",
            note="MRL-capable heavy local candidate; 2000d is the clean HNSW ceiling.",
        ),
        ModelSpec(
            key="voyage-4-nano",
            model_name="voyageai/voyage-4-nano",
            dimensions=voyage_dimensions,
            trust_remote_code=True,
            note="2048d is over the default pgvector HNSW ceiling and only included with --include-over-limit.",
        ),
        ModelSpec(
            key="qwen3-embedding-0.6b",
            model_name="Qwen/Qwen3-Embedding-0.6B",
            dimensions=(1024,),
            trust_remote_code=True,
            query_prompt_name="query",
            note="Smaller Qwen baseline; 1024d native max.",
        ),
        ModelSpec(
            key="snowflake-arctic-l-v2",
            model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
            dimensions=snowflake_dimensions,
            trust_remote_code=True,
            note="1024d candidate; 256d is only a speed/storage curiosity.",
        ),
        ModelSpec(
            key="bge-large-baseline",
            model_name="BAAI/bge-large-en-v1.5",
            dimensions=(1024,),
            note="Current local quality baseline.",
        ),
    ]


def expand_cases(specs: list[ModelSpec], max_dimension: int, include_over_limit: bool) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for spec in specs:
        for dimension in spec.dimensions:
            if dimension > max_dimension and not include_over_limit:
                continue
            cases.append(BenchmarkCase(spec=spec, dimension=dimension))
    return cases


def selected_specs(specs: list[ModelSpec], requested_models: str) -> list[ModelSpec]:
    if requested_models == "all":
        return specs

    requested = {item.strip() for item in requested_models.split(",") if item.strip()}
    specs_by_key = {spec.key: spec for spec in specs}
    unknown = sorted(requested - set(specs_by_key))
    if unknown:
        known = ", ".join(sorted(specs_by_key))
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown)}. Known keys: {known}")

    return [spec for spec in specs if spec.key in requested]


def print_case_table(cases: list[BenchmarkCase]) -> None:
    table = Table(title="Local Embedding Benchmark Matrix")
    table.add_column("Key")
    table.add_column("Model")
    table.add_column("Dim", justify="right")
    table.add_column("HNSW Safe")
    table.add_column("Query Prompt")
    table.add_column("Note")

    for case in cases:
        spec = case.spec
        prompt = spec.query_prompt_name or spec.query_prompt or ""
        table.add_row(
            spec.key,
            spec.model_name,
            str(case.dimension),
            "yes" if case.pgvector_hnsw_safe else "no",
            prompt,
            spec.note,
        )

    console.print(table)


def encode_vectors(
    model,
    texts: list[str],
    spec: ModelSpec,
    truncate_dim: int | None,
    input_type: Literal["document", "query"],
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    prompt_name = spec.query_prompt_name if input_type == "query" else spec.document_prompt_name
    prompt = spec.query_prompt if input_type == "query" else spec.document_prompt

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=batch_size,
        prompt_name=prompt_name,
        prompt=prompt,
        truncate_dim=truncate_dim,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(vectors, dtype=np.float32)


def time_encode(
    model,
    texts: list[str],
    spec: ModelSpec,
    truncate_dim: int | None,
    input_type: Literal["document", "query"],
    batch_size: int,
    normalize_embeddings: bool,
) -> EncodeTiming:
    started = time.perf_counter()
    encode_vectors(model, texts, spec, truncate_dim, input_type, batch_size, normalize_embeddings)
    seconds = time.perf_counter() - started
    vectors_per_second = len(texts) / seconds if seconds > 0 else 0.0
    return EncodeTiming(seconds=seconds, vectors_per_second=vectors_per_second)


def cosine_scores(query_vectors: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    document_norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    query_safe = query_vectors / np.maximum(query_norms, 1e-12)
    document_safe = document_vectors / np.maximum(document_norms, 1e-12)
    return np.matmul(query_safe, document_safe.T)


def quality_metrics(
    model,
    spec: ModelSpec,
    truncate_dim: int | None,
    batch_size: int,
    normalize_embeddings: bool,
) -> QualityMetrics:
    documents = DOCUMENT_EXAMPLES
    queries = [query.text for query in QUERY_EXAMPLES]
    document_vectors = encode_vectors(
        model, documents, spec, truncate_dim, "document", batch_size, normalize_embeddings
    )
    query_vectors = encode_vectors(model, queries, spec, truncate_dim, "query", batch_size, normalize_embeddings)
    scores = cosine_scores(query_vectors, document_vectors)

    reciprocal_ranks: list[float] = []
    top1_hits = 0
    positive_margins: list[float] = []

    for query_index, query in enumerate(QUERY_EXAMPLES):
        ordered = np.argsort(scores[query_index])[::-1]
        positive_rank = int(np.where(ordered == query.positive_document_index)[0][0]) + 1
        reciprocal_ranks.append(1.0 / positive_rank)
        if positive_rank == 1:
            top1_hits += 1

        positive_score = float(scores[query_index, query.positive_document_index])
        negative_scores = np.delete(scores[query_index], query.positive_document_index)
        positive_margins.append(positive_score - float(np.max(negative_scores)))

    return QualityMetrics(
        top1=top1_hits / len(QUERY_EXAMPLES),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        average_positive_margin=sum(positive_margins) / len(positive_margins),
    )


def profile_fingerprint(
    spec: ModelSpec, effective_dimension: int, truncate_dim: int | None, normalize_embeddings: bool
) -> str:
    from hindsight_api.engine.embeddings import LocalSTEmbeddingProfile

    profile = LocalSTEmbeddingProfile(
        model_name=spec.model_name,
        dimension=effective_dimension,
        trust_remote_code=spec.trust_remote_code,
        query_prompt_name=spec.query_prompt_name,
        query_prompt=spec.query_prompt,
        document_prompt_name=spec.document_prompt_name,
        document_prompt=spec.document_prompt,
        truncate_dim=truncate_dim,
        normalize_embeddings=normalize_embeddings,
    )
    return profile.fingerprint


def error_result(
    case: BenchmarkCase,
    normalize_embeddings: bool,
    load_seconds: float | None,
    native_dimension: int | None,
    error: str,
) -> BenchmarkResult:
    spec = case.spec
    return BenchmarkResult(
        key=spec.key,
        model_name=spec.model_name,
        dimension=case.dimension,
        native_dimension=native_dimension,
        pgvector_hnsw_safe=case.pgvector_hnsw_safe,
        trust_remote_code=spec.trust_remote_code,
        query_prompt_name=spec.query_prompt_name,
        query_prompt=spec.query_prompt,
        document_prompt_name=spec.document_prompt_name,
        document_prompt=spec.document_prompt,
        normalize_embeddings=normalize_embeddings,
        profile_fingerprint=None,
        load_seconds=load_seconds,
        document_seconds=None,
        document_vectors_per_second=None,
        query_seconds=None,
        query_vectors_per_second=None,
        top1=None,
        mrr=None,
        average_positive_margin=None,
        error=error,
    )


def benchmark_case(
    case: BenchmarkCase,
    model,
    native_dimension: int,
    load_seconds: float,
    repeat: int,
    batch_size: int,
    normalize_embeddings: bool,
) -> BenchmarkResult:
    if case.dimension > native_dimension:
        return error_result(
            case,
            normalize_embeddings,
            load_seconds,
            native_dimension,
            f"Requested dimension {case.dimension} exceeds native dimension {native_dimension}",
        )

    spec = case.spec
    truncate_dim = None if case.dimension == native_dimension else case.dimension
    perf_documents = DOCUMENT_EXAMPLES * repeat
    perf_queries = [query.text for query in QUERY_EXAMPLES] * repeat

    encode_vectors(model, DOCUMENT_EXAMPLES[:2], spec, truncate_dim, "document", batch_size, normalize_embeddings)
    document_timing = time_encode(
        model, perf_documents, spec, truncate_dim, "document", batch_size, normalize_embeddings
    )
    query_timing = time_encode(model, perf_queries, spec, truncate_dim, "query", batch_size, normalize_embeddings)
    quality = quality_metrics(model, spec, truncate_dim, batch_size, normalize_embeddings)

    return BenchmarkResult(
        key=spec.key,
        model_name=spec.model_name,
        dimension=case.dimension,
        native_dimension=native_dimension,
        pgvector_hnsw_safe=case.pgvector_hnsw_safe,
        trust_remote_code=spec.trust_remote_code,
        query_prompt_name=spec.query_prompt_name,
        query_prompt=spec.query_prompt,
        document_prompt_name=spec.document_prompt_name,
        document_prompt=spec.document_prompt,
        normalize_embeddings=normalize_embeddings,
        profile_fingerprint=profile_fingerprint(spec, case.dimension, truncate_dim, normalize_embeddings),
        load_seconds=load_seconds,
        document_seconds=document_timing.seconds,
        document_vectors_per_second=document_timing.vectors_per_second,
        query_seconds=query_timing.seconds,
        query_vectors_per_second=query_timing.vectors_per_second,
        top1=quality.top1,
        mrr=quality.mrr,
        average_positive_margin=quality.average_positive_margin,
    )


def run_benchmarks(
    cases: list[BenchmarkCase],
    repeat: int,
    batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    local_files_only: bool,
) -> list[BenchmarkResult]:
    from sentence_transformers import SentenceTransformer

    results: list[BenchmarkResult] = []
    cases_by_model: dict[str, list[BenchmarkCase]] = {}
    for case in cases:
        cases_by_model.setdefault(case.spec.key, []).append(case)

    for model_cases in cases_by_model.values():
        spec = model_cases[0].spec
        console.print(f"\n[bold]Loading[/bold] {spec.key}: {spec.model_name}")
        started = time.perf_counter()
        native_dimension: int | None = None
        try:
            if device and local_files_only:
                model = SentenceTransformer(
                    spec.model_name,
                    trust_remote_code=spec.trust_remote_code,
                    device=device,
                    local_files_only=True,
                )
            elif device:
                model = SentenceTransformer(spec.model_name, trust_remote_code=spec.trust_remote_code, device=device)
            elif local_files_only:
                model = SentenceTransformer(
                    spec.model_name,
                    trust_remote_code=spec.trust_remote_code,
                    local_files_only=True,
                )
            else:
                model = SentenceTransformer(spec.model_name, trust_remote_code=spec.trust_remote_code)
            load_seconds = time.perf_counter() - started
            native_dimension = model.get_sentence_embedding_dimension()
            if native_dimension is None:
                raise RuntimeError("SentenceTransformer returned no native embedding dimension")

            for case in model_cases:
                console.print(f"  benchmarking {case.dimension}d")
                try:
                    results.append(
                        benchmark_case(
                            case,
                            model,
                            native_dimension,
                            load_seconds,
                            repeat,
                            batch_size,
                            normalize_embeddings,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep matrix runs going.
                    results.append(error_result(case, normalize_embeddings, load_seconds, native_dimension, str(exc)))
        except Exception as exc:  # noqa: BLE001 - report load failures per requested dim.
            load_seconds = time.perf_counter() - started
            for case in model_cases:
                results.append(error_result(case, normalize_embeddings, load_seconds, native_dimension, str(exc)))

    return results


def print_result_table(results: list[BenchmarkResult]) -> None:
    table = Table(title="Local Embedding Benchmark Results")
    table.add_column("Key")
    table.add_column("Dim", justify="right")
    table.add_column("Native", justify="right")
    table.add_column("Docs/s", justify="right")
    table.add_column("Queries/s", justify="right")
    table.add_column("Top1", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Margin", justify="right")
    table.add_column("Error")

    for result in results:
        table.add_row(
            result.key,
            str(result.dimension),
            "" if result.native_dimension is None else str(result.native_dimension),
            "" if result.document_vectors_per_second is None else f"{result.document_vectors_per_second:.2f}",
            "" if result.query_vectors_per_second is None else f"{result.query_vectors_per_second:.2f}",
            "" if result.top1 is None else f"{result.top1:.3f}",
            "" if result.mrr is None else f"{result.mrr:.3f}",
            "" if result.average_positive_margin is None else f"{result.average_positive_margin:.4f}",
            result.error or "",
        )

    console.print(table)


def write_results(run: BenchmarkRun, output: Path | None) -> Path:
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("hindsight-dev/benchmarks/results/embeddings") / f"local_embedding_matrix-{timestamp}.json"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(run), indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="Comma-separated model keys, or 'all'.")
    parser.add_argument("--repeat", type=int, default=8, help="Repeat built-in examples for throughput timing.")
    parser.add_argument("--batch-size", type=int, default=8, help="SentenceTransformers encode batch size.")
    parser.add_argument("--device", default=None, help="Optional SentenceTransformers device, such as cpu or cuda.")
    parser.add_argument("--max-dimension", type=int, default=PGVECTOR_HNSW_MAX_DIM)
    parser.add_argument("--include-curiosities", action="store_true", help="Include Snowflake 256d curiosity run.")
    parser.add_argument(
        "--include-over-limit", action="store_true", help="Include >2000d offline-only runs such as 2048d."
    )
    parser.add_argument("--no-normalize", action="store_true", help="Do not request normalized embeddings.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached Hugging Face files only.")
    parser.add_argument(
        "--list", action="store_true", help="Print selected cases without downloading or loading models."
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_dimension < 1:
        raise SystemExit("--max-dimension must be >= 1")

    specs = selected_specs(
        build_model_specs(args.include_curiosities, args.include_over_limit),
        args.models,
    )
    cases = expand_cases(specs, args.max_dimension, args.include_over_limit)
    print_case_table(cases)

    if args.list:
        return

    normalize_embeddings = not args.no_normalize
    results = run_benchmarks(
        cases=cases,
        repeat=args.repeat,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=args.local_files_only,
    )
    print_result_table(results)

    run = BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(),
        repeat=args.repeat,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=normalize_embeddings,
        local_files_only=args.local_files_only,
        max_dimension=args.max_dimension,
        results=results,
    )
    output = write_results(run, args.output)
    console.print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
