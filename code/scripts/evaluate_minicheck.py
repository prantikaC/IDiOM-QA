"""
Faithfulness evaluation via MiniCheck (Tang et al., 2024) -- small models
trained on synthetic data specifically to reproduce GPT-4-level fact-checking
judgments at a fraction of the cost/latency. Validated on LLM-AggreFact, a
benchmark aggregating multiple fact-checking datasets; the paper's whole
pitch is "same accuracy as an expensive LLM judge, far cheaper/faster" --
which is exactly the tradeoff motivating this script over the RAGAS/deepseek
judge route.

Same methodological role as HHEM/AlignScore/RAGAS Faithfulness: compares the
generated ANSWER against the retrieved CONTEXT (not the gold reference), so
it can catch confident hallucination that ROUGE/BLEU/METEOR/BERTScore cannot.

Requires: pip install minicheck
(HF-hosted backbones, e.g. flan-t5-large-based, pulled automatically on
first run -- no manual checkpoint download needed, unlike AlignScore).

NOTE ON API: this script targets the `minicheck` PyPI package's documented
`MiniCheck` class and `.score(docs=..., claims=...)` method, returning
(pred_label, raw_prob, ...) from memory of the published interface -- if the
installed version's signature differs, adjust `load_model`/`evaluate_system`
below to match (`pip show minicheck` / the package's own README will have
the exact current signature).

Usage:
    python scripts/evaluate_minicheck.py \\
        --data data/eval/idiom_qa_baselines/consolidated_all_volume_matched_validation.json \\
        --output-dir data/eval/eval_results/minicheck/all_volume_matched \\
        --device cuda:0
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA = "data/eval/idiom_qa_baselines/consolidated_all_volume_matched_validation.json"
DEFAULT_OUTPUT_DIR = "data/eval/eval_results/minicheck"
DEFAULT_MODEL = "flan-t5-large"  # smallest/fastest MiniCheck backbone; swap to
                                 # "Bespoke-MiniCheck-7B" for the strongest variant
                                 # at real extra cost/latency


def load_model(model_name: str, device: str, cache_dir: str):
    from minicheck.minicheck import MiniCheck
    print(f"Loading MiniCheck ({model_name}) on {device} ...")
    return MiniCheck(model_name=model_name, cache_dir=cache_dir, device=device)


def evaluate_system(name: str, rows: list, scorer, out_dirs: dict) -> "pd.DataFrame | None":
    docs, claims, meta = [], [], []
    for row in rows:
        if row.get("error") is not None:
            continue
        context = str(row.get("context", "")).strip()
        answer = str(row.get("actual_output", "")).strip()
        if not context or not answer:
            continue
        docs.append(context)
        claims.append(answer)
        meta.append({
            "index": row.get("index"),
            "question_type": row.get("question_type", "Unknown"),
            "input": row.get("input", ""),
            "actual_output": answer,
        })

    if not docs:
        print(f"  No valid rows for {name}; skipping.")
        return None

    print(f"  Scoring {name} ({len(docs)} rows) ...")
    pred_label, raw_prob, _, _ = scorer.score(docs=docs, claims=claims)

    for i in range(len(meta)):
        meta[i]["MiniCheck_label"] = int(pred_label[i])   # 1 = supported, 0 = unsupported
        meta[i]["MiniCheck_prob"] = float(raw_prob[i])     # continuous support probability

    detailed_df = pd.DataFrame(meta)
    detailed_df.to_csv(out_dirs["detailed"] / f"{name}_minicheck_detailed.csv", index=False)

    qt_df = detailed_df.groupby("question_type")[["MiniCheck_label", "MiniCheck_prob"]].mean().reset_index()
    qt_df.to_csv(out_dirs["qtype"] / f"{name}_question_type.csv", index=False)

    overall = {
        "System": name,
        "Average_MiniCheck_label": detailed_df["MiniCheck_label"].mean(),
        "Average_MiniCheck_prob": detailed_df["MiniCheck_prob"].mean(),
    }
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(out_dirs["overall"] / f"{name}_overall.csv", index=False)

    return overall_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"Consolidated results JSON (default: {DEFAULT_DATA})")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help=f"Output root (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"MiniCheck backbone (default: {DEFAULT_MODEL}).")
    ap.add_argument("--cache-dir", default="./ckpts",
                    help="Where MiniCheck downloads/caches its backbone weights.")
    ap.add_argument("--systems", nargs="*", default=None,
                    help="Subset of system names to evaluate (default: all).")
    ap.add_argument("--device", default=None,
                    help="'cuda:0', 'cpu'. Default: cuda if available else cpu.")
    args = ap.parse_args()

    import json
    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

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

    scorer = load_model(args.model, device, args.cache_dir)

    overall_dfs = []
    for name in systems:
        rows = evaluation_data["baselines"][name]["results"]
        df = evaluate_system(name, rows, scorer, out_dirs)
        if df is not None:
            overall_dfs.append(df)

    if not overall_dfs:
        raise SystemExit("No systems produced results.")

    final_df = pd.concat(overall_dfs, ignore_index=True)
    final_df = final_df.sort_values("Average_MiniCheck_prob", ascending=False).reset_index(drop=True)

    reports = out_dirs["reports"]
    final_df.to_csv(reports / "final_minicheck_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL -- sorted by MiniCheck support probability (higher = more faithful)")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print(f"\nReports -> {reports}")


if __name__ == "__main__":
    main()
