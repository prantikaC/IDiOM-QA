"""
Faithfulness evaluation via Vectara's HHEM (Hughes Hallucination Evaluation
Model) -- a lightweight, purpose-built alternative to RAGAS's LLM-judge
Faithfulness metric.

Unlike ROUGE/BLEU/METEOR/BERTScore (which compare the generated answer
against the GOLD REFERENCE), HHEM compares the generated answer against the
RETRIEVED CONTEXT: it asks "is this answer actually supported by what was
retrieved?", independent of how closely it's worded relative to the gold
answer. This is the same methodological role RAGAS's Faithfulness metric
fills, via a small cross-encoder (no generation, single forward pass) rather
than an LLM reasoning through it -- orders of magnitude faster and free to
run locally.

Model: vectara/hallucination_evaluation_model (HHEM-2.1-Open), loaded via
transformers.AutoModelForSequenceClassification(trust_remote_code=True) per
Vectara's own documented usage (confirmed against the model card, 2026-09) --
NOT sentence_transformers.CrossEncoder, despite that being a common guess.
.predict() takes a list of (premise, hypothesis) tuples, i.e.
(context, answer) here, and returns a tensor of scores in [0, 1]: 0 = not
supported by the context (hallucinated), 1 = fully supported (consistent).

Caveat: calibrated mostly on summarization-style data (news/general text),
not administrative/legal registers like Office Memoranda -- spot-check a
handful of known-good/known-bad rows from this corpus before trusting it
wholesale (see --spot-check).

Requires: pip install transformers torch
(run this in the project's actual env, e.g. the omAgentic conda env --
NOT necessarily whatever bare `python3` resolves to in an ad-hoc shell).

Usage:
    python scripts/evaluate_hhem.py \\
        --data data/eval/idiom_qa_baselines/consolidated_all_volume_matched_validation.json \\
        --output-dir data/eval/eval_results/hhem/all_volume_matched \\
        --device cuda:0

    # sanity-check the model against a handful of hand-inspected rows first:
    python scripts/evaluate_hhem.py --data ... --spot-check 5
"""

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_DATA = "data/eval/idiom_qa_baselines/consolidated_all_volume_matched_validation.json"
DEFAULT_OUTPUT_DIR = "data/eval/eval_results/hhem"
MODEL_NAME = "vectara/hallucination_evaluation_model"


def load_model(device: str):
    from transformers import AutoModelForSequenceClassification
    print(f"Loading {MODEL_NAME} on {device} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    return model.to(device)


def evaluate_system(name: str, rows: list, model, out_dirs: dict, spot_check: int = 0,
                    batch_size: int = 16) -> "pd.DataFrame | None":
    pairs, meta = [], []
    for row in rows:
        if row.get("error") is not None:
            continue
        context = str(row.get("context", "")).strip()
        answer = str(row.get("actual_output", "")).strip()
        if not context or not answer:
            continue
        pairs.append((context, answer))
        meta.append({
            "index": row.get("index"),
            "question_type": row.get("question_type", "Unknown"),
            "input": row.get("input", ""),
            "actual_output": answer,
        })

    if not pairs:
        print(f"  No valid rows for {name}; skipping.")
        return None

    print(f"  Scoring {name} ({len(pairs)} rows) ...")
    # Batch manually -- the model card's own example only demonstrates a
    # single pair at a time; batching in chunks avoids relying on unverified
    # internal batching behaviour for a large row count.
    all_scores = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        out = model.predict(batch)
        all_scores.extend(out.tolist() if hasattr(out, "tolist") else list(out))

    for i, s in enumerate(all_scores):
        meta[i]["HHEM_score"] = float(s)

    if spot_check:
        print(f"\n  --- spot-check: {name}, first {spot_check} rows ---")
        for m, (ctx, ans) in zip(meta[:spot_check], pairs[:spot_check]):
            print(f"    score={m['HHEM_score']:.3f}  Q[:80]={m['input'][:80]!r}")
            print(f"      context[:150]={ctx[:150]!r}")
            print(f"      answer[:150]={ans[:150]!r}")
        print("  --- end spot-check ---\n")

    detailed_df = pd.DataFrame(meta)
    detailed_df.to_csv(out_dirs["detailed"] / f"{name}_hhem_detailed.csv", index=False)

    qt_df = detailed_df.groupby("question_type")[["HHEM_score"]].mean().reset_index()
    qt_df.to_csv(out_dirs["qtype"] / f"{name}_question_type.csv", index=False)

    overall = {"System": name, "Average_HHEM": detailed_df["HHEM_score"].mean()}
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(out_dirs["overall"] / f"{name}_overall.csv", index=False)

    return overall_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"Consolidated results JSON (default: {DEFAULT_DATA})")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help=f"Output root (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--systems", nargs="*", default=None,
                    help="Subset of system names to evaluate (default: all).")
    ap.add_argument("--device", default=None,
                    help="'cuda', 'cuda:0', 'cpu'. Default: cuda if available else cpu.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Pairs per forward pass; lower it if long contexts run out of GPU memory.")
    ap.add_argument("--spot-check", type=int, default=0,
                    help="Print the first N (context, answer, score) triples per system "
                         "for manual sanity-checking against this corpus's register "
                         "(HHEM is calibrated on general/news text, not OMs).")
    args = ap.parse_args()

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

    model = load_model(device)

    overall_dfs = []
    for name in systems:
        rows = evaluation_data["baselines"][name]["results"]
        df = evaluate_system(name, rows, model, out_dirs, spot_check=args.spot_check,
                             batch_size=args.batch_size)
        if df is not None:
            overall_dfs.append(df)

    if not overall_dfs:
        raise SystemExit("No systems produced results.")

    final_df = pd.concat(overall_dfs, ignore_index=True)
    final_df = final_df.sort_values("Average_HHEM", ascending=False).reset_index(drop=True)

    reports = out_dirs["reports"]
    final_df.to_csv(reports / "final_hhem_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL -- sorted by HHEM score (higher = more faithful to retrieved context)")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print(f"\nReports -> {reports}")


if __name__ == "__main__":
    main()
