"""
Surface-metric evaluation for IDiOM-QA — ROUGE / BLEU / METEOR / BERTScore.

Faithful .py port of `scripts/IDIOM_QA_Evaluation.ipynb` (authored by a co-author),
adapted so it runs from the repo without requiring the original directory
layout. Reads the consolidated 10-system results file produced by
`scripts/consolidate_results.py`, computes per-row scores, and emits:

  <output-dir>/detailed_scores/<system>_detailed_scores.csv
  <output-dir>/question_type_analysis/<system>_question_type.csv
  <output-dir>/overall_scores/<system>_overall.csv
  <output-dir>/reports/final_baseline_comparison.{csv,xlsx}
  <output-dir>/reports/paper_ready_table.csv

USAGE
    python scripts/evaluate_surface_metrics.py
    python scripts/evaluate_surface_metrics.py \
        --data data/eval/idiom_qa_baselines/all_systems_results_openai.json \
        --output-dir data/eval/eval_results/surface_metrics
    python scripts/evaluate_surface_metrics.py --systems idiom_rag rerank   # subset
    python scripts/evaluate_surface_metrics.py --device cpu                  # no GPU
"""

import argparse
import os
import warnings
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bert_score import score as bertscore
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")

DEFAULT_DATA = "data/eval/idiom_qa_baselines/all_systems_results_openai.json"
DEFAULT_OUTPUT_DIR = "data/eval/eval_results/surface_metrics"

METRIC_COLUMNS = [
    "ROUGE-1", "ROUGE-2", "ROUGE-L",
    "BLEU-1", "BLEU-2", "BLEU-4",
    "METEOR",
    "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
]


# ── Metric helpers ───────────────────────────────────────────────────────────
def compute_rouge_scores(reference, prediction, scorer):
    s = scorer.score(reference, prediction)
    return {
        "ROUGE-1": s["rouge1"].fmeasure,
        "ROUGE-2": s["rouge2"].fmeasure,
        "ROUGE-L": s["rougeL"].fmeasure,
    }


def compute_bleu_scores(reference, prediction, smoothie):
    ref_tokens = [reference.split()]
    pred_tokens = prediction.split()
    return {
        "BLEU-1": sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0),
                                smoothing_function=smoothie),
        "BLEU-2": sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0),
                                smoothing_function=smoothie),
        "BLEU-4": sentence_bleu(ref_tokens, pred_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smoothie),
    }


def compute_meteor(reference, prediction):
    return meteor_score([reference.split()], prediction.split())


# ── Per-system evaluation ────────────────────────────────────────────────────
def evaluate_system(name: str, rows: list, out_dirs: dict, device: str) -> pd.DataFrame | None:
    print("\n" + "=" * 80)
    print(f"Evaluating: {name}")
    print("=" * 80)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    results, references, predictions = [], [], []

    for row in rows:
        if row.get("error") is not None:
            continue
        reference = str(row.get("expected_output", "")).strip()
        prediction = str(row.get("actual_output", "")).strip()
        if not reference or not prediction:
            continue

        references.append(reference)
        predictions.append(prediction)

        scores = {
            "index": row.get("index"),
            "question_type": row.get("question_type", "Unknown"),
            "input": row.get("input", ""),
            "expected_output": reference,
            "actual_output": prediction,
        }
        scores.update(compute_rouge_scores(reference, prediction, rouge))
        scores.update(compute_bleu_scores(reference, prediction, smoothie))
        scores["METEOR"] = compute_meteor(reference, prediction)
        results.append(scores)

    if not results:
        print(f"  No valid rows for {name}; skipping.")
        return None

    print(f"  Computing BERTScore for {name} ({len(predictions)} rows, device={device})...")
    P, R, F1 = bertscore(predictions, references, lang="en", device=device, verbose=False)
    for i, _ in enumerate(results):
        results[i]["BERTScore_Precision"] = P[i].item()
        results[i]["BERTScore_Recall"] = R[i].item()
        results[i]["BERTScore_F1"] = F1[i].item()

    detailed_df = pd.DataFrame(results)
    detailed_df.to_csv(out_dirs["detailed"] / f"{name}_detailed_scores.csv", index=False)

    qt_df = detailed_df.groupby("question_type")[METRIC_COLUMNS].mean().reset_index()
    qt_df.to_csv(out_dirs["qtype"] / f"{name}_question_type.csv", index=False)

    overall = {"Baseline": name}
    for m in METRIC_COLUMNS:
        overall[f"Average_{m}"] = detailed_df[m].mean()
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(out_dirs["overall"] / f"{name}_overall.csv", index=False)

    return overall_df


def setup_nltk():
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"Consolidated results JSON (default: {DEFAULT_DATA})")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help=f"Output root (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--systems", nargs="*", default=None,
                    help="Subset of system names to evaluate (default: all 10).")
    ap.add_argument("--device", default=None,
                    help="BERTScore device: 'cuda', 'cuda:0', 'cpu'. "
                         "Default: cuda if available else cpu.")
    args = ap.parse_args()

    setup_nltk()

    import json
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"Data:    {data_path}")
    print(f"Output:  {out_root}")
    print(f"Systems: {systems}")
    print(f"Device:  {device}")

    overall_dfs = []
    for name in systems:
        rows = evaluation_data["baselines"][name]["results"]
        df = evaluate_system(name, rows, out_dirs, device=device)
        if df is not None:
            overall_dfs.append(df)

    if not overall_dfs:
        raise SystemExit("No systems produced results.")

    final_df = pd.concat(overall_dfs, ignore_index=True)
    final_df = final_df.sort_values("Average_BERTScore_F1", ascending=False).reset_index(drop=True)

    reports = out_dirs["reports"]
    final_df.to_csv(reports / "final_baseline_comparison.csv", index=False)
    try:
        final_df.to_excel(reports / "final_baseline_comparison.xlsx", index=False)
    except Exception as e:
        print(f"  [warn] Skipped Excel export: {e}")

    paper_cols = ["Baseline",
                  "Average_ROUGE-1", "Average_ROUGE-2", "Average_ROUGE-L",
                  "Average_BLEU-1",  "Average_BLEU-2",  "Average_BLEU-4",
                  "Average_METEOR",  "Average_BERTScore_F1"]
    final_df[paper_cols].to_csv(reports / "paper_ready_table.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL — sorted by BERTScore F1")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print(f"\nReports → {reports}")


if __name__ == "__main__":
    main()
