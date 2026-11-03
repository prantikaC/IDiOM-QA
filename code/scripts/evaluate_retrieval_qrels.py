"""
Document-level retrieval evaluation of IDiOM-QA v4 against its TREC qrels
(data/eval/v4/qrels/, built by scripts/build_qrels.py).

Every retriever ranks all 490 chunks of the production index for each
question; a document's score is its best chunk's score (MaxP), so the run is
a ranking of the 116 OMs. Retrievers:

  bm25                  rank_bm25 Okapi over the production BM25 corpus,
                        whitespace tokens (same as LangChain's BM25Retriever)
  dense-bge-small       BAAI/bge-small-en-v1.5, the production FAISS model
  dense-bge-large       BAAI/bge-large-en-v1.5
  dense-bge-m3          BAAI/bge-m3
  hybrid                RRF (k=60) of bm25 and dense-bge-small chunk ranks,
                        as in the Hybrid baseline
  hybrid+minilm         hybrid top-50 chunks reranked by
                        cross-encoder/ms-marco-MiniLM-L-6-v2 (IDiOM-RAG's reranker)
  hybrid+bge-reranker   hybrid top-50 chunks reranked by BAAI/bge-reranker-base
                        (the Rerank baseline's model)

No query instruction is prepended for any dense model, matching production.
No recency weighting is applied: these are the retrievers on their own.

Metrics (binary relevance, Adversarial questions have no qrels and are
excluded): nDCG@10, MRR@10, Recall@10, Recall@20. Recall@100 is not reported
because the corpus has only 116 documents.

Outputs (data/eval/eval_results/retrieval_qrels_v4/):
  runs/<retriever>.run    TREC run files (top 100 documents per query)
  overall.csv             one row per retriever
  by_type.csv             nDCG@10 and Recall@10 per retriever and question type

    python scripts/evaluate_retrieval_qrels.py --device cuda
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

QRELS_DIR = "data/eval/v4/qrels"
BM25_DATA = "data/local_db/bm25_data.json"
OUT = "data/eval/eval_results/retrieval_qrels_v4"
DENSE = {"dense-bge-small": "BAAI/bge-small-en-v1.5",
         "dense-bge-large": "BAAI/bge-large-en-v1.5",
         "dense-bge-m3": "BAAI/bge-m3"}
RERANKERS = {"hybrid+minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
             "hybrid+bge-reranker": "BAAI/bge-reranker-base"}
RRF_K, RERANK_POOL, RUN_DEPTH = 60, 50, 100


def load_qrels():
    qrels = defaultdict(set)
    for line in open(f"{QRELS_DIR}/idiom_qa_v4.qrels"):
        q, _, d, rel = line.split()
        if int(rel) > 0:
            qrels[q].add(d)
    queries, qtype = {}, {}
    with open(f"{QRELS_DIR}/queries.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            q, t, text = line.rstrip("\n").split("\t", 2)
            queries[q], qtype[q] = text, t
    return qrels, queries, qtype


def doc_ranking(chunk_scores: np.ndarray, chunk_doc: list[str]) -> list[str]:
    """Documents ordered by their best chunk's score (MaxP)."""
    best = {}
    for i in np.argsort(-chunk_scores):
        best.setdefault(chunk_doc[i], None)
        if len(best) == RUN_DEPTH:
            break
    return list(best)


def metrics(ranking: list[str], rel: set[str]) -> dict:
    dcg = sum(1 / math.log2(r + 2) for r, d in enumerate(ranking[:10]) if d in rel)
    idcg = sum(1 / math.log2(r + 2) for r in range(min(len(rel), 10)))
    rr = next((1 / (r + 1) for r, d in enumerate(ranking[:10]) if d in rel), 0.0)
    return {"nDCG@10": dcg / idcg, "MRR@10": rr,
            "Recall@10": len(rel & set(ranking[:10])) / len(rel),
            "Recall@20": len(rel & set(ranking[:20])) / len(rel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer

    qrels, queries, qtype = load_qrels()
    qids = sorted(qrels)
    data = json.load(open(BM25_DATA, encoding="utf-8"))
    chunks = data["corpus"]
    chunk_doc = [m["source_file"][:-4].replace(" ", "_") for m in data["metadata"]]
    texts = [queries[q] for q in qids]
    print(f"{len(qids)} queries with qrels, {len(chunks)} chunks over {len(set(chunk_doc))} documents")

    scores = {}  # retriever -> (n_queries, n_chunks) chunk-score matrix
    bm25 = BM25Okapi([c.split() for c in chunks])
    scores["bm25"] = np.array([bm25.get_scores(t.split()) for t in texts])
    for name, model in DENSE.items():
        print(f"Encoding with {model}")
        m = SentenceTransformer(model, device=args.device)
        c = m.encode(chunks, normalize_embeddings=True, batch_size=32)
        q = m.encode(texts, normalize_embeddings=True, batch_size=32)
        scores[name] = q @ c.T
        del m

    def ranks(s):  # chunk rank (0 = best) per query
        return np.argsort(np.argsort(-s, axis=1), axis=1)
    rrf = 1 / (RRF_K + 1 + ranks(scores["bm25"])) + 1 / (RRF_K + 1 + ranks(scores["dense-bge-small"]))
    scores["hybrid"] = rrf

    for name, model in RERANKERS.items():
        print(f"Reranking hybrid top-{RERANK_POOL} with {model}")
        ce = CrossEncoder(model, device=args.device)
        s = np.full_like(rrf, -np.inf)
        for qi, t in enumerate(texts):
            pool = np.argsort(-rrf[qi])[:RERANK_POOL]
            s[qi, pool] = ce.predict([(t, chunks[i]) for i in pool], batch_size=64)
        scores[name] = s
        del ce

    os.makedirs(f"{OUT}/runs", exist_ok=True)
    overall, by_type = [], []
    for name, s in scores.items():
        per_q = {}
        with open(f"{OUT}/runs/{name}.run", "w") as f:
            for qi, q in enumerate(qids):
                ranking = doc_ranking(s[qi], chunk_doc)
                f.writelines(f"{q} Q0 {d} {r + 1} {RUN_DEPTH - r} {name}\n" for r, d in enumerate(ranking))
                per_q[q] = metrics(ranking, qrels[q])
        keys = list(next(iter(per_q.values())))
        overall.append({"retriever": name, **{k: round(np.mean([m[k] for m in per_q.values()]), 4) for k in keys}})
        groups = defaultdict(list)
        for q, m in per_q.items():
            groups[qtype[q]].append(m)
        for t, ms in sorted(groups.items()):
            by_type.append({"retriever": name, "question_type": t, "n": len(ms),
                            "nDCG@10": round(np.mean([m["nDCG@10"] for m in ms]), 4),
                            "Recall@10": round(np.mean([m["Recall@10"] for m in ms]), 4)})

    for path, rows in ((f"{OUT}/overall.csv", overall), (f"{OUT}/by_type.csv", by_type)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{'retriever':22s} " + " ".join(f"{k:>9s}" for k in list(overall[0])[1:]))
    for r in overall:
        print(f"{r['retriever']:22s} " + " ".join(f"{v:9.3f}" for v in list(r.values())[1:]))
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
