"""
Build TREC-format document-level qrels for IDiOM-QA v4.

A document is one OM (the retrieval index's `source_file`, minus ".txt";
spaces become "_" so IDs are TREC-safe). Relevance is binary:
  - Factoid / Conceptual / Relational / Multi-part: the question's source OM
  - CrossHop: both OMs (om_a_id, om_b_id)
  - Temporal / Aggregation: every OM in seed_docs (identical to the documents
    in the gold evidence, context_used)
  - Adversarial: no relevant documents by construction, so no qrels lines;
    these are scored on abstention instead and excluded from retrieval metrics.

Outputs (data/eval/v4/qrels/):
  idiom_qa_v4.qrels   qid 0 docid 1
  queries.tsv         qid, question_type, question
  docids.tsv          docid, source_file (as stored in the index)
Query IDs are q001..q475 in dataset order, matching the `index` field used by
the RAGAS / surface-metric outputs.
"""
import glob
import json
import os
import sys
from collections import Counter

DATASET = "data/eval/v4/idiom_qa_v4.json"
CORPUS_GLOB = "data/DataFiles_v2/**/*.txt"
OUT = "data/eval/v4/qrels"


def docid(stem: str) -> str:
    return stem.replace(" ", "_")


def main():
    corpus = sorted(os.path.basename(f)[:-4] for f in glob.glob(CORPUS_GLOB, recursive=True)
                    if not f.endswith("_summary.txt"))
    ids = {docid(s): s for s in corpus}
    assert len(ids) == len(corpus), "docid collision after replacing spaces"

    dataset = json.load(open(DATASET, encoding="utf-8"))
    qrels, missing = [], []
    for i, r in enumerate(dataset, start=1):
        qid, t = f"q{i:03d}", r["question_type"]
        if t == "Adversarial":
            continue
        if t == "CrossHop":
            rel = [r["om_a_id"], r["om_b_id"]]
        elif t in ("Temporal", "Aggregation"):
            rel = r["seed_docs"]
        else:
            rel = [r["source"]]
        for s in dict.fromkeys(rel):  # dedupe, keep order
            if s not in corpus:
                missing.append((qid, s))
            qrels.append((qid, docid(s)))
    if missing:
        sys.exit(f"Relevant documents not in the corpus: {missing[:5]}")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/idiom_qa_v4.qrels", "w") as f:
        f.writelines(f"{q} 0 {d} 1\n" for q, d in qrels)
    with open(f"{OUT}/queries.tsv", "w", encoding="utf-8") as f:
        f.write("qid\tquestion_type\tquestion\n")
        f.writelines(f"q{i:03d}\t{r['question_type']}\t{r['input']}\n" for i, r in enumerate(dataset, start=1))
    with open(f"{OUT}/docids.tsv", "w", encoding="utf-8") as f:
        f.write("docid\tsource_file\n")
        f.writelines(f"{d}\t{s}.txt\n" for d, s in ids.items())

    per_q = Counter(q for q, _ in qrels)
    by_type = Counter()
    for i, r in enumerate(dataset, start=1):
        if f"q{i:03d}" in per_q:
            by_type[r["question_type"]] += per_q[f"q{i:03d}"]
    print(f"{len(qrels)} qrels lines for {len(per_q)} queries (Adversarial excluded), {len(ids)} documents")
    print("relevant docs per type:", dict(by_type))
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
