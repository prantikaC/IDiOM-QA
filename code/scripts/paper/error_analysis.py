"""Error analysis of the three strongest systems (by RAGAS Answer Correctness
and BERTScore-F1: IDiOM-RAG, R3-RAG, Rerank RAG) on IDiOM-QA v4.

A response is a failure if its RAGAS Answer Correctness is below 0.4 (about
the lowest quarter for each system). Each failure gets one category from the
evaluation signals, applied in this order:
  answered-unanswerable  Adversarial question, response is not a refusal
  false-refusal          answerable question, response is a refusal
  retrieval-miss         context recall < 0.5: the gold facts were not retrieved
  unfaithful             evidence retrieved (CR >= 0.5) but faithfulness < 0.5
  incomplete/imprecise   evidence retrieved and the answer is grounded in it,
                         but it is partial or misses the asked detail
  (residual)             Adversarial question correctly refused but scored low
                         against the gold refusal wording
Writes data/eval/eval_results/error_analysis/{failures.csv,summary.json}.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from refusal import is_refusal  # noqa: E402

R = Path("data/eval/eval_results/ragas_deepseek_full475_matched/detailed_scores")
OUT = Path("data/eval/eval_results/error_analysis")
SYSTEMS = {"idiom_rag": "IDiOM-RAG", "r3_rag": "R3-RAG", "rerank": "Rerank RAG"}
THRESHOLD = 0.4


def category(r):
    adv = r.question_type == "Adversarial"
    ref = is_refusal(r.response)
    if adv:
        return "answered-unanswerable" if not ref else "refusal scored low"
    if ref:
        return "false-refusal"
    if r.context_recall < 0.5:
        return "retrieval-miss"
    if r.faithfulness < 0.5:
        return "unfaithful"
    return "incomplete/imprecise"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, summary = [], {}
    for key, name in SYSTEMS.items():
        x = pd.read_csv(R / f"{key}_ragas_detailed.csv")
        x["response"] = x.response.fillna("")
        f = x[x.answer_correctness < THRESHOLD].copy()
        f["category"] = f.apply(category, axis=1)
        f["system"] = name
        frames.append(f)
        summary[name] = {"n_fail": int(len(f)), "share": round(len(f) / len(x), 3),
                         "by_category": f.category.value_counts().to_dict(),
                         "by_type": f.question_type.value_counts().to_dict()}
    fails = pd.concat(frames)
    fails[["system", "question_type", "category", "answer_correctness", "faithfulness", "context_recall",
           "user_input", "reference", "response"]].to_csv(OUT / "failures.csv", index=False)

    per_q = fails.groupby("user_input").system.nunique()
    summary["overlap"] = {"failed_by_all_3": int((per_q == 3).sum()), "failed_by_exactly_2": int((per_q == 2).sum()),
                          "failed_by_1": int((per_q == 1).sum())}
    all3 = set(per_q[per_q == 3].index)
    t = fails[fails.user_input.isin(all3)].drop_duplicates("user_input").question_type.value_counts()
    summary["failed_by_all_3_by_type"] = t.to_dict()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
