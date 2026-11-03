"""
Baseline evaluation runner.

Runs all baselines, with per-baseline caching so the run is resumable
on interruption.

Usage:
    python baselines/eval_baselines.py                            # all, openai
    python baselines/eval_baselines.py --backend ollama           # all, open-source
    python baselines/eval_baselines.py --baseline crag            # single baseline
    python baselines/eval_baselines.py --baseline gear --limit 50

Output: data/eval/baseline_results_{backend}.json
        data/eval/baseline_cache_{name}_{backend}.json  (one cache per baseline × backend)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from baselines.crag_baseline         import CRAGBaseline
from baselines.adaptive_rag_baseline import AdaptiveRAGBaseline
from baselines.gear_baseline         import GEARBaseline
from baselines.hyde_baseline         import HyDEBaseline
from baselines.naive_rag_baseline    import NaiveRAGBaseline
from baselines.hybrid_baseline       import HybridRAGBaseline
from baselines.rerank_baseline       import RerankRAGBaseline
from baselines.deep_rag_baseline     import DeepRAGBaseline
from baselines.r3_rag_baseline       import R3RAGBaseline
from baselines.search_r1_baseline    import SearchR1Baseline
from baselines.closed_book_baseline  import ClosedBookBaseline
from baselines.gold_context_baseline import GoldContextBaseline

DEFAULT_EVAL_FILE  = os.path.join("data", "eval", "synthetic_test_set_v7.json")
DEFAULT_OUTPUT_DIR = os.path.join("data", "eval")

BASELINES = {
    # Naive RAG (Gao 2023 §3.1)
    "naive_rag":    NaiveRAGBaseline,
    # Advanced RAG (Gao 2023 §3.2)
    "hybrid":       HybridRAGBaseline,
    "rerank":       RerankRAGBaseline,
    "hyde":         HyDEBaseline,
    # Modular RAG (Gao 2023 §3.3)
    "crag":         CRAGBaseline,
    "adaptive_rag": AdaptiveRAGBaseline,
    # GraphRAG family
    "gear":         GEARBaseline,
    # Agentic RAG (2025) — prompted reproductions, same backend LLM as the
    # rest of the suite (see deep_rag_baseline.py / r3_rag_baseline.py /
    # search_r1_baseline.py docstrings for the fidelity tradeoff)
    "deep_rag":     DeepRAGBaseline,
    "r3_rag":       R3RAGBaseline,
    "search_r1":    SearchR1Baseline,
}

# Reference points, not RAG systems: run only when named with --baseline, so
# a bare run (all baselines) still means the RAG comparison suite.
REFERENCE_POINTS = {
    "closed_book":  ClosedBookBaseline,
    "gold_context": GoldContextBaseline,
}


def require_openai_api_key(backend: str):
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "ERROR: OPENAI_API_KEY is not set. Add it to .env or export it in the shell."
        )
    if backend == "qwen":
        import urllib.request
        base_url = os.getenv("QWEN_FAST_BASE_URL", "http://localhost:8000/v1")
        try:
            urllib.request.urlopen(base_url.replace("/v1", "/health"), timeout=3)
        except Exception:
            raise SystemExit(
                f"ERROR: Qwen vLLM server not reachable at {base_url}.\n"
                "Start it first, e.g.:\n"
                "  vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \\\n"
                "      --port 8000 --gpu-memory-utilization 0.85 \\\n"
                "      --max-model-len 16384 --dtype auto"
            )


def result_file(backend: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"baseline_results_{backend}.json")


def cache_path(name: str, backend: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"baseline_cache_{name}_{backend}.json")


def load_cache(name: str, backend: str, output_dir: str) -> dict:
    path = cache_path(name, backend, output_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not v.get("error")}
    return {}


def save_cache(name: str, backend: str, cache: dict, output_dir: str):
    path = cache_path(name, backend, output_dir)
    # Write to a temp file and swap it in, so stopping the run mid-write never
    # leaves a truncated cache that fails to load on resume.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_baseline(name: str, cls, dataset: list, limit: int | None, backend: str, output_dir: str, top_k: int = 5) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Running baseline: {name.upper()}  [{backend}]")
    print(f"{'='*60}")

    cache = load_cache(name, backend, output_dir)
    print(f"Cache hits: {len(cache)} / {len(dataset)}")

    baseline = cls(top_k=top_k, backend=backend)
    rows = []
    questions = dataset[:limit] if limit else dataset

    for i, item in enumerate(questions):
        query    = item["input"]
        expected = item["expected_output"]
        q_type   = item.get("question_type", "Unknown")
        source   = item.get("source", "Unknown")

        print(f"  [{i+1}/{len(questions)}] [{q_type}] {query[:65]}{'...' if len(query)>65 else ''}", end="", flush=True)

        if query in cache:
            entry = cache[query]
            print(" (cached)")
        else:
            entry = baseline.run(query)
            cache[query] = entry
            save_cache(name, backend, cache, output_dir)
            status = "ERR" if entry.get("error") else f"{entry.get('latency_seconds', 0):.1f}s"
            print(f" → {status}")

        rows.append({
            "index":           i + 1,
            "source":          source,
            "question_type":   q_type,
            "input":           query,
            "expected_output": expected,
            "actual_output":   entry.get("answer", ""),
            "context":         entry.get("context", ""),
            "latency_seconds": entry.get("latency_seconds", 0),
            "error":           entry.get("error"),
            # baseline-specific extras (may be absent)
            "verdict":         entry.get("verdict"),
            "strategy":        entry.get("strategy"),
            "iterations":      entry.get("iterations"),
            "graph_passages":  entry.get("graph_passages"),
        })

    return rows


def compute_summary(rows: list[dict]) -> dict:
    total    = len(rows)
    errors   = sum(1 for r in rows if r["error"])
    answered = total - errors
    avg_lat  = round(sum(r["latency_seconds"] for r in rows) / total, 2) if total else 0

    by_type: dict = {}
    for r in rows:
        qt = r["question_type"]
        if qt not in by_type:
            by_type[qt] = {"total": 0, "errors": 0, "latency_sum": 0.0}
        by_type[qt]["total"]       += 1
        by_type[qt]["errors"]      += int(bool(r["error"]))
        by_type[qt]["latency_sum"] += r["latency_seconds"]

    type_summary = {
        qt: {
            "total":       v["total"],
            "errors":      v["errors"],
            "avg_latency": round(v["latency_sum"] / v["total"], 2) if v["total"] else 0,
        }
        for qt, v in by_type.items()
    }

    return {
        "total_questions": total,
        "answered":        answered,
        "errors":          errors,
        "error_rate_pct":  round(errors / total * 100, 1) if total else 0,
        "avg_latency_seconds": avg_lat,
        "by_question_type":    type_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "ollama", "qwen"], default="openai",
                        help="LLM backend (default: openai)")
    parser.add_argument("--baseline", choices=list(BASELINES) + list(REFERENCE_POINTS), default=None,
                        help="Run a single baseline (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions (for quick testing)")
    parser.add_argument("--dataset", default=DEFAULT_EVAL_FILE,
                        help=f"Eval dataset JSON (default: {DEFAULT_EVAL_FILE})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory for cache + results files "
                             f"(default: {DEFAULT_OUTPUT_DIR}). Per-baseline caches "
                             f"and the merged results file land here.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Retrieval depth passed to the baseline's own top_k "
                             "parameter (default: 5, matching each baseline's own "
                             "default). Used for volume-controlled comparisons against "
                             "idiom_rag's TOP_K_RERANK -- e.g. adaptive_rag effectively "
                             "uses top_k*2 (strategy B) or up to 3*top_k accumulated "
                             "(strategy C), so pass a smaller value here to match "
                             "idiom_rag's actual final context-chunk count.")
    args = parser.parse_args()
    require_openai_api_key(args.backend)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Dataset   : {args.dataset}  ({len(dataset)} questions)")
    print(f"Output dir: {args.output_dir}")
    if args.limit:
        print(f"  (limited to first {args.limit})")

    backend = args.backend
    out_file = result_file(backend, args.output_dir)
    targets = {args.baseline: {**BASELINES, **REFERENCE_POINTS}[args.baseline]} if args.baseline else BASELINES
    all_results: dict = {}

    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            all_results = json.load(f).get("baselines", {})

    for name, cls in targets.items():
        rows = run_baseline(name, cls, dataset, args.limit, backend, args.output_dir, top_k=args.top_k)
        all_results[name] = {
            "summary": compute_summary(rows),
            "results": rows,
        }
        report = {
            "run_timestamp": datetime.now().isoformat(),
            "backend":       backend,
            "baselines":     all_results,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  {name} complete → saved to {out_file}")

    # Final summary table
    print(f"\n{'='*60}")
    print("ALL BASELINES COMPLETE")
    print(f"{'='*60}")
    header = f"{'Baseline':<20} {'Total':>7} {'Errors':>7} {'Err%':>6} {'AvgLat':>8}"
    print(header)
    print("-" * len(header))
    for name, data in all_results.items():
        s = data["summary"]
        print(f"{name:<20} {s['total_questions']:>7} {s['errors']:>7} "
              f"{s['error_rate_pct']:>5.1f}% {s['avg_latency_seconds']:>7.2f}s")
    print(f"\nFull results: {out_file}")


if __name__ == "__main__":
    main()
