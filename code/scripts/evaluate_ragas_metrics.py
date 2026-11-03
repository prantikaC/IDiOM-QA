"""
RAGAS LLM-judge evaluation for IDiOM-QA.

Faithful .py port of `scripts/IDIOM_QA_Evaluation_Ragas.ipynb` (authored by
the original notebook). Two material fixes vs the notebook:

  * The notebook hardcoded a live OpenAI API key. THIS script reads the key
    from the `OPENAI_API_KEY` environment variable (loaded via .env if
    present). Never commit a key to git.
  * Output paths default to the repo's `data/eval/eval_results/ragas_metrics/`
    rather than the original notebook's hard-coded paths.

Scores each system in the consolidated results file on six RAGAS metrics:
  faithfulness, answer_relevancy, context_precision, context_recall,
  answer_similarity, answer_correctness.

USAGE
    python scripts/evaluate_ragas_metrics.py
    python scripts/evaluate_ragas_metrics.py --systems idiom_rag rerank
    python scripts/evaluate_ragas_metrics.py --judge gpt-4o     # stronger judge
    python scripts/evaluate_ragas_metrics.py --metrics answer_relevancy context_precision answer_similarity answer_correctness
        # drops faithfulness/context_recall, the two statement-decomposition
        # metrics that issue multiple LLM calls per row (cost scales with
        # the number of atomic statements/claims extracted) -- the biggest
        # lever for cutting per-row judge cost against a slow local model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_DATA = "data/eval/idiom_qa_baselines/all_systems_results_openai.json"
DEFAULT_OUTPUT_DIR = "data/eval/eval_results/ragas_metrics"

ALL_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_similarity",
    "answer_correctness",
]


def build_dataset(rows: list, drop_empty_context: bool) -> "pd.DataFrame":
    import pandas as pd
    out = []
    for item in rows:
        if item.get("error") is not None:
            continue
        q   = str(item.get("input", "")).strip()
        a   = str(item.get("actual_output", "")).strip()
        gt  = str(item.get("expected_output", "")).strip()
        ctx = str(item.get("context", "")).strip()
        if not q or not a or not gt:
            continue
        if drop_empty_context and not ctx:
            continue
        out.append({
            "question":      q,
            "answer":        a,
            "ground_truth":  gt,
            # RAGAS expects a list of strings under "contexts".
            "contexts":      [ctx] if ctx else [""],
            "question_type": item.get("question_type", "Unknown"),
            "index":         item.get("index"),
        })
    return pd.DataFrame(out)


def _load_checkpoint(path: "Path", required_columns: list) -> "pd.DataFrame | None":
    """Load a prior partial (or complete) detailed_scores CSV for resuming.

    Returns None if the file doesn't exist, is unreadable, or is missing any
    of the currently-requested metric columns (e.g. a prior run used
    different --metrics) -- in all those cases we start the system fresh
    rather than risk silently mixing incompatible checkpoints.
    """
    import pandas as pd
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] Could not read checkpoint {path}: {e}. Starting fresh.")
        return None
    missing = [c for c in required_columns + ["index", "question_type"] if c not in df.columns]
    if missing:
        print(f"  [warn] Checkpoint {path} missing columns {missing} "
              f"(likely a different --metrics set). Starting fresh.")
        return None
    return df


def evaluate_system(name: str, rows: list, judge_llm, embeddings,
                    out_dirs: dict, drop_empty_context: bool,
                    metric_names: list, run_config=None,
                    chunk_size: int = 20) -> "pd.DataFrame":
    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision,
        context_recall, answer_similarity, answer_correctness,
    )

    metric_lookup = {
        "faithfulness":       faithfulness,
        "answer_relevancy":   answer_relevancy,
        "context_precision":  context_precision,
        "context_recall":     context_recall,
        "answer_similarity":  answer_similarity,
        "answer_correctness": answer_correctness,
    }
    selected_metrics = [metric_lookup[m] for m in metric_names]

    print("\n" + "=" * 80)
    print(f"RAGAS: {name}")
    print("=" * 80)

    df_in = build_dataset(rows, drop_empty_context=drop_empty_context)
    if df_in.empty:
        print(f"  No valid rows for {name}; skipping.")
        return None
    print(f"  {len(df_in)} valid samples (drop_empty_context={drop_empty_context})")

    # Checkpointing: reuses the final detailed_scores CSV as the checkpoint
    # file itself -- a completed run just IS a full checkpoint. Evaluated in
    # chunks of `chunk_size` rows so a crash (OOM, SIGKILL, daemon restart,
    # timeout pile-up) loses at most one chunk's progress instead of the
    # whole system's evaluate() call -- previously there was no way to
    # recover partial progress at all, since evaluate() only returned (and
    # got written to disk) after ALL rows in the system finished.
    checkpoint_path = out_dirs["detailed"] / f"{name}_ragas_detailed.csv"
    accumulated = _load_checkpoint(checkpoint_path, metric_names)
    if accumulated is not None and not accumulated.empty:
        done_indices = set(accumulated["index"].tolist())
        remaining_df = df_in[~df_in["index"].isin(done_indices)].reset_index(drop=True)
        print(f"  Resuming from checkpoint: {len(accumulated)}/{len(df_in)} rows "
              f"already scored, {len(remaining_df)} remaining.")
    else:
        accumulated = pd.DataFrame()
        remaining_df = df_in

    if remaining_df.empty:
        print(f"  All rows already scored for {name} (checkpoint complete).")
    else:
        n_chunks = (len(remaining_df) + chunk_size - 1) // chunk_size
        print(f"  Evaluating {len(remaining_df)} rows in {n_chunks} chunk(s) "
              f"of up to {chunk_size} (checkpointed after each chunk).")
        for ci in range(n_chunks):
            chunk_df = remaining_df.iloc[ci * chunk_size:(ci + 1) * chunk_size].reset_index(drop=True)
            print(f"  --- {name}: chunk {ci + 1}/{n_chunks} ({len(chunk_df)} rows) ---")
            hf_dataset = Dataset.from_pandas(chunk_df)
            eval_kwargs = dict(
                dataset=hf_dataset,
                metrics=selected_metrics,
                llm=judge_llm,
                embeddings=embeddings,
            )
            if run_config is not None:
                eval_kwargs["run_config"] = run_config
            result = evaluate(**eval_kwargs)
            chunk_result_df = result.to_pandas()
            chunk_result_df["question_type"] = chunk_df["question_type"].values
            chunk_result_df["index"]         = chunk_df["index"].values

            accumulated = pd.concat([accumulated, chunk_result_df], ignore_index=True)
            # Write-then-swap so stopping the run mid-write can't leave a
            # truncated checkpoint (which would restart this system from zero).
            tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
            accumulated.to_csv(tmp_path, index=False)
            os.replace(tmp_path, checkpoint_path)
            print(f"  Checkpoint saved: {len(accumulated)}/{len(df_in)} rows scored so far "
                  f"-> {checkpoint_path}")

    result_df = accumulated

    qt = result_df.groupby("question_type")[metric_names].mean().reset_index()
    qt.to_csv(out_dirs["qtype"] / f"{name}_question_type.csv", index=False)

    overall = {"Baseline": name, **{m: result_df[m].mean() for m in metric_names}}
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(out_dirs["overall"] / f"{name}_overall.csv", index=False)
    return overall_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"Consolidated results JSON (default: {DEFAULT_DATA})")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help=f"Output root (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--systems", nargs="*", default=None,
                    help="Subset of system names (default: all 10).")
    ap.add_argument("--metrics", nargs="*", default=None, choices=ALL_METRIC_NAMES,
                    help=f"Subset of metrics to compute (default: all 6: "
                         f"{ALL_METRIC_NAMES}). faithfulness and context_recall "
                         f"are the most expensive against a slow local judge -- "
                         f"they decompose the answer/ground_truth into atomic "
                         f"statements/claims and issue one verification call "
                         f"per statement, so cost scales with answer length, "
                         f"not just a flat per-row cost like the others.")
    ap.add_argument("--judge", default="gpt-4o-mini",
                    help="Judge LLM. OpenAI model name (e.g. gpt-4o-mini); a Claude "
                         "model (claude-haiku-4-5, claude-sonnet-4-6); or a local "
                         "open-source model via Ollama as 'ollama:<model>' "
                         "(e.g. ollama:qwen2.5:32b-instruct). Cross-family judging "
                         "(Claude or Ollama, differing from the gpt-4o-mini generator) "
                         "avoids self-preference bias. NB: answer_similarity uses the "
                         "embedder, not the judge.")
    ap.add_argument("--embeddings", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HuggingFace embedding model for relevancy/similarity.")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="Concurrent judge requests. Default 16 for cloud APIs, "
                         "auto-lowered to 4 for ollama: judges (local models can't "
                         "serve 16 in parallel -> TimeoutErrors).")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Per-request timeout (s). Default 180 for cloud APIs, "
                         "auto-raised to 900 for ollama: judges (local generation is slow).")
    ap.add_argument("--drop-empty-context", action="store_true",
                    help="Drop rows where the system returned an empty context "
                         "(matches the notebook behaviour; may under-count "
                         "hard-refusal rows in IDiOM-RAG).")
    ap.add_argument("--chunk-size", type=int, default=20,
                    help="Rows evaluated per ragas.evaluate() call within a system "
                         "(default 20). The detailed_scores CSV is checkpointed after "
                         "every chunk, so re-running the same command after a crash/"
                         "kill resumes from the last completed chunk instead of "
                         "re-scoring the whole system. Lower it for more frequent "
                         "checkpoints against a slow/unstable local judge; raise it "
                         "to reduce per-chunk overhead once the judge is stable.")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    _judge_lower = args.judge.lower()
    _judge_is_openai = not (_judge_lower.startswith("ollama:") or _judge_lower.startswith("claude"))
    if _judge_is_openai and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set. Put it in .env or export it, "
                 "or pass --judge ollama:<model> / --judge claude-... to avoid OpenAI.")

    import pandas as pd
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    evaluation_data = json.loads(data_path.read_text(encoding="utf-8"))

    out_root = Path(args.output_dir)
    out_dirs = {
        "detailed": out_root / "detailed_scores",
        "qtype":    out_root / "question_type_analysis",
        "overall":  out_root / "overall_scores",
        "reports":  out_root / "reports",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    systems = list(evaluation_data["baselines"].keys())
    if args.systems:
        unknown = set(args.systems) - set(systems)
        if unknown:
            raise SystemExit(f"Unknown systems: {unknown}. Available: {systems}")
        systems = args.systems

    metric_names = args.metrics if args.metrics else list(ALL_METRIC_NAMES)

    print(f"Data:        {data_path}")
    print(f"Output:      {out_root}")
    print(f"Systems:     {systems}")
    print(f"Metrics:     {metric_names}")
    print(f"Judge:       {args.judge}")
    print(f"Embeddings:  {args.embeddings}")

    # Judge routing by --judge prefix:
    #   "ollama:<model>" -> local open-source model via Ollama (free, cross-family)
    #   "claude..."      -> Anthropic
    #   anything else     -> OpenAI
    jl = args.judge.lower()
    if jl.startswith("ollama:"):
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        model_name = args.judge.split(":", 1)[1]
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        judge_llm = LangchainLLMWrapper(
            ChatOllama(model=model_name, temperature=0, base_url=base_url))
        print(f"Judge family: Ollama / open-source ({model_name}) @ {base_url}")
    elif jl.startswith("claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ERROR: judge is a Claude model but ANTHROPIC_API_KEY is not set.")
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper
        judge_llm = LangchainLLMWrapper(ChatAnthropic(model=args.judge, temperature=0))
        print("Judge family: Anthropic (cross-family)")
    else:
        judge_llm = ChatOpenAI(model=args.judge, temperature=0)
        print("Judge family: OpenAI")
    embeddings = HuggingFaceEmbeddings(model_name=args.embeddings)

    # Concurrency/timeout: cloud-API defaults overwhelm a local Ollama server,
    # so auto-tune down for ollama: judges (override with --max-workers/--timeout).
    from ragas.run_config import RunConfig
    is_local = jl.startswith("ollama:")
    max_workers = args.max_workers if args.max_workers is not None else (4 if is_local else 16)
    timeout = args.timeout if args.timeout is not None else (900 if is_local else 180)
    run_config = RunConfig(timeout=timeout, max_workers=max_workers)
    print(f"RunConfig: max_workers={max_workers}, timeout={timeout}s")

    overall_dfs = []
    for name in systems:
        rows = evaluation_data["baselines"][name]["results"]
        try:
            df = evaluate_system(name, rows, judge_llm, embeddings, out_dirs,
                                 drop_empty_context=args.drop_empty_context,
                                 metric_names=metric_names,
                                 run_config=run_config,
                                 chunk_size=args.chunk_size)
            if df is not None:
                overall_dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            continue

    if not overall_dfs:
        raise SystemExit("No systems produced RAGAS results.")

    final_df = pd.concat(overall_dfs, ignore_index=True)
    sort_col = "answer_correctness" if "answer_correctness" in metric_names else metric_names[0]
    final_df = final_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    final_df.to_csv(out_dirs["reports"] / "final_ragas_comparison.csv", index=False)
    try:
        final_df.to_excel(out_dirs["reports"] / "final_ragas_comparison.xlsx", index=False)
    except Exception as e:
        print(f"  [warn] Skipped Excel export: {e}")

    print("\n" + "=" * 80)
    print("FINAL RAGAS — sorted by answer_correctness")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print(f"\nReports → {out_dirs['reports']}")


if __name__ == "__main__":
    main()
