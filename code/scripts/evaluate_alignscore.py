"""
Faithfulness evaluation via AlignScore (Zha et al., ACL 2023) -- a
RoBERTa-based "information alignment" model trained across 15+ reformulated
NLP tasks (NLI, QA, paraphrase, fact-checking) to score whether a claim is
supported by a context. Reported to beat earlier factual-consistency metrics
(FactCC, QuestEval, SummaC) on standard benchmarks.

Same methodological role as HHEM/RAGAS Faithfulness: compares the generated
ANSWER against the retrieved CONTEXT (not the gold reference), so it can
catch confident hallucination that ROUGE/BLEU/METEOR/BERTScore cannot.

Requires:
    pip install alignscore
    python -m spacy download en_core_web_sm     # AlignScore's sentence splitter needs this

AlignScore also needs a checkpoint file downloaded separately (it is NOT
pulled from the HuggingFace Hub automatically) -- get it from the official
repo's release page: https://github.com/yuh-zha/AlignScore
(look for AlignScore-base or AlignScore-large under "Checkpoints" in the
README; base is ~125M params/~1.4GB, large ~355M/~3GB). Pass the downloaded
.ckpt path via --ckpt-path.

NOTE ON API: this script targets the `alignscore` PyPI package's documented
`AlignScore` class and `.score(contexts=..., claims=...)` method from memory
of the published interface -- if the installed version's signature differs
slightly, adjust the `load_model`/`score_batch` functions below to match
(`pip show alignscore` / the package's own README will have the exact
current signature).

Usage:
    python scripts/evaluate_alignscore.py \\
        --data data/eval/idiom_qa_baselines/consolidated_all_volume_matched_validation.json \\
        --output-dir data/eval/eval_results/alignscore/all_volume_matched \\
        --ckpt-path /path/to/AlignScore-base.ckpt \\
        --device cuda:0
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA = "data/eval/idiom_qa_baselines/consolidated_all_volume_matched_validation.json"
DEFAULT_OUTPUT_DIR = "data/eval/eval_results/alignscore"


def load_model(ckpt_path: str, model: str, device: str):
    from alignscore import AlignScore
    print(f"Loading AlignScore ({model}) from {ckpt_path} on {device} ...")
    return AlignScore(
        model=model,
        batch_size=32,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",  # NLI + smart-splitting, the mode AlignScore's own
                                   # paper recommends for long-document factual consistency
    )


def evaluate_system(name: str, rows: list, scorer, out_dirs: dict) -> "pd.DataFrame | None":
    contexts, claims, meta = [], [], []
    for row in rows:
        if row.get("error") is not None:
            continue
        context = str(row.get("context", "")).strip()
        answer = str(row.get("actual_output", "")).strip()
        if not context or not answer:
            continue
        contexts.append(context)
        claims.append(answer)
        meta.append({
            "index": row.get("index"),
            "question_type": row.get("question_type", "Unknown"),
            "input": row.get("input", ""),
            "actual_output": answer,
        })

    if not contexts:
        print(f"  No valid rows for {name}; skipping.")
        return None

    print(f"  Scoring {name} ({len(contexts)} rows) ...")
    scores = scorer.score(contexts=contexts, claims=claims)

    for i, s in enumerate(scores):
        meta[i]["AlignScore"] = float(s)

    detailed_df = pd.DataFrame(meta)
    detailed_df.to_csv(out_dirs["detailed"] / f"{name}_alignscore_detailed.csv", index=False)

    qt_df = detailed_df.groupby("question_type")[["AlignScore"]].mean().reset_index()
    qt_df.to_csv(out_dirs["qtype"] / f"{name}_question_type.csv", index=False)

    overall = {"System": name, "Average_AlignScore": detailed_df["AlignScore"].mean()}
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(out_dirs["overall"] / f"{name}_overall.csv", index=False)

    return overall_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"Consolidated results JSON (default: {DEFAULT_DATA})")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help=f"Output root (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--ckpt-path", required=True,
                    help="Path to the downloaded AlignScore .ckpt checkpoint "
                         "(see docstring for the download link).")
    ap.add_argument("--model", default="roberta-large",
                    choices=["roberta-base", "roberta-large"],
                    help="Backbone matching the checkpoint you downloaded "
                         "(base checkpoint -> roberta-base, large -> roberta-large).")
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

    scorer = load_model(args.ckpt_path, args.model, device)

    overall_dfs = []
    for name in systems:
        rows = evaluation_data["baselines"][name]["results"]
        df = evaluate_system(name, rows, scorer, out_dirs)
        if df is not None:
            overall_dfs.append(df)

    if not overall_dfs:
        raise SystemExit("No systems produced results.")

    final_df = pd.concat(overall_dfs, ignore_index=True)
    final_df = final_df.sort_values("Average_AlignScore", ascending=False).reset_index(drop=True)

    reports = out_dirs["reports"]
    final_df.to_csv(reports / "final_alignscore_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL -- sorted by AlignScore (higher = more supported by retrieved context)")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print(f"\nReports -> {reports}")


if __name__ == "__main__":
    main()
