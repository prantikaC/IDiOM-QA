"""
Evaluation runner for IDiOM-RAG (LangGraph multi-agent system).

Runs the full pipeline on every question, logs rich results per question
(actual output, context, agent contributions, plan, retries, latency),
and writes everything to data/eval/idiom_rag_results_{dataset-stem}_{backend}.json.

Resumes from cache on interruption — safe to re-run at any time.

Usage:
    # Default: run on the current eval set with OpenAI backend
    python scripts/run_idiom_rag.py

    # Explicit dataset (current default benchmark)
    python scripts/run_idiom_rag.py --dataset data/eval/v4/idiom_qa_v4.json

    # Different backend / quick smoke
    python scripts/run_idiom_rag.py --backend ollama
    python scripts/run_idiom_rag.py --backend qwen        # local vLLM (Qwen2.5-72B-AWQ)
    python scripts/run_idiom_rag.py --limit 20

Note: this script was previously named scripts/run_eval.py; the new name
makes it clearer that this is the IDiOM-RAG entry point, distinct from
scripts/evaluate_baselines.py, evaluate_surface_metrics.py, and
evaluate_ragas_metrics.py.
"""

import os
import sys
import math
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.core.graph import build_sequential_graph
import config

DEFAULT_EVAL_FILE  = os.path.join("data", "eval", "synthetic_test_set_v8.json")
DEFAULT_OUTPUT_DIR = os.path.join("data", "eval")


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


def _paths(dataset_path: str, backend: str, output_dir: str,
           smoke: bool = False) -> tuple[str, str]:
    """Derive cache + result file paths from the dataset name so that
    runs on v7/v8/etc. don't collide."""
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    # e.g. "synthetic_test_set_v8" -> "v8"; fall back to full stem for anything else
    short = stem.split("_")[-1] if stem.startswith("synthetic_test_set_") else stem
    suffix = f"_{short}_{backend}{'_smoke' if smoke else ''}"
    cache  = os.path.join(output_dir, f"idiom_rag_cache{suffix}.json")
    result = os.path.join(output_dir, f"idiom_rag_results{suffix}.json")
    return cache, result


def _finite_or_none(values):
    """Map non-finite floats (±inf, NaN) to None so the cache/results stay
    strict-JSON valid. The reranker emits -inf as its 'no context found'
    sentinel for top_rerank_score, which the critic then logs; Python's json
    would otherwise write it as `-Infinity`, which strict parsers reject."""
    return [v if isinstance(v, (int, float)) and math.isfinite(v) else None
            for v in values]


def load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not v.get("error")}
    return {}


def save_cache(cache: dict, cache_file: str):
    # Write to a temp file and swap it in, so stopping the run mid-write never
    # leaves a truncated cache that fails to load on resume.
    tmp = cache_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, cache_file)


def run(dataset_path: str = DEFAULT_EVAL_FILE,
        backend: str = "openai",
        limit: int | None = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        shard: tuple[int, int] | None = None):
    require_openai_api_key(backend)
    os.makedirs(output_dir, exist_ok=True)
    cache_file, result_file = _paths(dataset_path, backend, output_dir,
                                     smoke=limit is not None)

    print("=" * 60)
    print(f"IDiOM-RAG EVALUATION — {os.path.basename(dataset_path)}  [{backend}]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]
    if shard:
        # Every n-th question, so two GPUs can split one dataset; each shard
        # writes its own output dir and the caches are merged afterwards.
        k, n = shard
        dataset = dataset[k::n]

    cache = load_cache(cache_file)
    print(f"Dataset: {len(dataset)} questions  |  Cache hits: {len(cache)}\n")

    print("Building LangGraph application...")
    app = build_sequential_graph(backend=backend)
    print("Graph ready.\n")

    results = []

    for i, item in enumerate(dataset):
        query          = item["input"]
        expected       = item["expected_output"]
        q_type         = item.get("question_type", "Unknown")
        source         = item.get("source", "Unknown")
        cache_key      = query

        print(f"[{i+1}/{len(dataset)}] [{q_type}] {query[:70]}{'...' if len(query)>70 else ''}")

        if cache_key in cache:
            print("  -> cache hit, skipping LangGraph call")
            entry = cache[cache_key]
        else:
            initial_state = {
                "original_query":       query,
                "messages":             [],
                "next":                 [],
                "gathered_context":     [],
                "final_context_string": "",
                "final_answer":         "",
                "loop_count":           0,
                "web_search_needed":    False,
                "is_web_searchable":    False,
                "plan":                 [],
                "past_steps":           [],
                "subtask_contexts":     [],
                "current_task":         "",
                "cypher_queries":       [],
                "total_retries":        0,
                "all_agent_sources":    [],
                "rewrite_count":        0,
                "task_variants":        [],
                # New fields (added with architecture improvements)
                "critic_reasoning":     "",
                "out_of_scope_tasks":   [],
                "active_retrievers":    ["vector", "bm25", "kg"],
                "top_rerank_score":     float("-inf"),
                "rerank_score_log":     [],
                "gate_fired_log":       [],
                "critic_escalated_log": [],
                "override_used_log":    [],
                "use_stronger_critic":  False,
            }

            t0 = time.time()
            try:
                final_state = app.invoke(
                    initial_state,
                    config={"recursion_limit": config.GRAPH_RECURSION_LIMIT},
                )
                latency = round(time.time() - t0, 2)

                actual_output        = final_state.get("final_answer", "")
                final_context_string = final_state.get("final_context_string", "")
                plan                 = final_state.get("plan", [])
                past_steps           = final_state.get("past_steps", [])
                subtask_contexts     = final_state.get("subtask_contexts", [])
                cypher_queries       = final_state.get("cypher_queries", [])
                total_retries        = final_state.get("total_retries", 0)
                all_agent_sources    = list(set(final_state.get("all_agent_sources", [])))
                task_variants        = final_state.get("task_variants", [])
                rerank_score_log     = _finite_or_none(final_state.get("rerank_score_log", []))
                gate_fired_log        = final_state.get("gate_fired_log", [])
                critic_escalated_log  = final_state.get("critic_escalated_log", [])

                entry = {
                    "actual_output":        actual_output,
                    "final_context_string": final_context_string,
                    "plan":                 plan,
                    "past_steps":           [[t, r] for t, r in past_steps],
                    "subtask_contexts":     subtask_contexts,
                    "total_retries":        total_retries,
                    "all_agent_sources":    all_agent_sources,
                    "cypher_queries":       cypher_queries,
                    "task_variants":        task_variants,
                    "rerank_score_log":     rerank_score_log,
                    "gate_fired_log":       gate_fired_log,
                    "critic_escalated_log": critic_escalated_log,
                    "latency_seconds":      latency,
                    "error":                None
                }

            except Exception as e:
                latency = round(time.time() - t0, 2)
                print(f"  -> ERROR: {e}")
                entry = {
                    "actual_output":        "",
                    "final_context_string": "",
                    "plan":                 [],
                    "past_steps":           [],
                    "subtask_contexts":     [],
                    "total_retries":        0,
                    "all_agent_sources":    [],
                    "cypher_queries":       [],
                    "task_variants":        [],
                    "rerank_score_log":     [],
                    "gate_fired_log":       [],
                    "critic_escalated_log": [],
                    "latency_seconds":      latency,
                    "error":                str(e)
                }

            cache[cache_key] = entry
            save_cache(cache, cache_file)

        results.append({
            "index":                  i + 1,
            "source":                 source,
            "question_type":          q_type,
            "input":                  query,
            "expected_output":        expected,
            "actual_output":          entry.get("actual_output", ""),
            "final_context_string":   entry.get("final_context_string", ""),
            "plan":                   entry.get("plan", []),
            "past_steps":             entry.get("past_steps", []),
            "subtask_contexts":       entry.get("subtask_contexts", []),
            "total_retries":          entry.get("total_retries", 0),
            "all_agent_sources":      entry.get("all_agent_sources", []),
            "cypher_queries":         entry.get("cypher_queries", []),
            "task_variants":          entry.get("task_variants", []),
            "rerank_score_log":       entry.get("rerank_score_log", []),
            "gate_fired_log":         entry.get("gate_fired_log", []),
            "critic_escalated_log":   entry.get("critic_escalated_log", []),
            "latency_seconds":        entry.get("latency_seconds", 0),
            "error":                  entry["error"]
        })

    # ── Summary stats ────────────────────────────────────────────────────────
    total    = len(results)
    errors   = sum(1 for r in results if r["error"])
    answered = total - errors
    avg_lat  = round(sum(r["latency_seconds"] for r in results) / total, 2) if total else 0

    by_type: dict = defaultdict(lambda: {"total": 0, "errors": 0, "avg_latency": []})
    for r in results:
        qt = r["question_type"]
        by_type[qt]["total"] += 1
        if r["error"]:
            by_type[qt]["errors"] += 1
        by_type[qt]["avg_latency"].append(r["latency_seconds"])

    type_summary = {}
    for qt, v in by_type.items():
        lats = v["avg_latency"]
        type_summary[qt] = {
            "total":       v["total"],
            "errors":      v["errors"],
            "avg_latency": round(sum(lats) / len(lats), 2) if lats else 0
        }

    # ── Error type breakdown ─────────────────────────────────────────────────
    error_counts: dict = defaultdict(int)
    for r in results:
        if r["error"]:
            msg = str(r["error"])
            if "timed out" in msg.lower() or "timeout" in msg.lower():
                error_counts["timeout"] += 1
            elif "rate limit" in msg.lower() or "429" in msg:
                error_counts["rate_limit"] += 1
            elif "context_length" in msg.lower() or "maximum context" in msg.lower():
                error_counts["context_length"] += 1
            elif "connection" in msg.lower():
                error_counts["connection"] += 1
            else:
                error_counts["other"] += 1

    error_breakdown = {
        k: {"count": v, "pct": round(v / errors * 100, 1) if errors else 0}
        for k, v in sorted(error_counts.items(), key=lambda x: -x[1])
    }

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_questions":     total,
            "answered":            answered,
            "errors":              errors,
            "error_rate_pct":      round(errors / total * 100, 1) if total else 0,
            "avg_latency_seconds": avg_lat
        },
        "error_breakdown": error_breakdown,
        "by_question_type": type_summary,
        "results": results
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"  Total : {total}  |  Answered: {answered}  |  Errors: {errors} ({round(errors/total*100,1) if total else 0}%)")
    print(f"  Avg latency: {avg_lat}s")
    if error_breakdown:
        print("\nError breakdown:")
        for etype, v in error_breakdown.items():
            print(f"  {etype:<20} {v['count']:>4}  ({v['pct']}%)")
    print("\nBy question type:")
    for qt, v in type_summary.items():
        print(f"  {qt:<20} total={v['total']}  errors={v['errors']}  avg_lat={v['avg_latency']}s")
    print(f"\nResults saved to: {result_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_EVAL_FILE,
                        help=f"Path to QA dataset JSON (default: {DEFAULT_EVAL_FILE})")
    parser.add_argument("--backend", choices=["openai", "ollama", "qwen"], default="openai",
                        help="LLM backend to use (default: openai)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions (for testing)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory for cache + results files "
                             f"(default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="Run only questions K, K+N, K+2N, ... (0-based), e.g. 0/2 and 1/2.")
    args = parser.parse_args()
    shard = tuple(int(v) for v in args.shard.split("/")) if args.shard else None
    run(dataset_path=args.dataset, backend=args.backend, limit=args.limit,
        output_dir=args.output_dir, shard=shard)
