"""
Hybrid Retrieval — BM25 + dense, fused by Reciprocal Rank Fusion (RRF).

References:
  BM25: Robertson & Zaragoza 2009, "The Probabilistic Relevance Framework"
  RRF:  Cormack, Clarke & Büttcher, SIGIR 2009,
        "Reciprocal Rank Fusion outperforms Condorcet and individual rank
        learning methods"
  Taxonomy: Gao et al. 2023 (arXiv:2312.10997) §3.2 "Advanced RAG"

Pipeline:
  query → BM25 top-N    ┐
                        ├─ RRF fuse → top-k → concat → generator
  query → dense top-N   ┘

RRF score for doc d: Σ_r 1 / (k + rank_r(d))     k = 60 (Cormack default)
"""

import time
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


RRF_K = 60       # canonical Cormack et al. 2009 value
RETRIEVE_N = 10  # per-retriever pool before fusion (top_k taken after)


GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an administrative assistant for CSIR/IACS/JU/UGC.
Answer the query using ONLY the provided context.
If the context does not contain the answer, say:
"I could not find sufficient information in the internal documents."

CONTEXT:
{context}

QUERY: {query}

ANSWER:"""),
])


def _doc_key(doc) -> str:
    """Identity key for a chunk — must be stable across retrievers."""
    return doc.page_content


def reciprocal_rank_fusion(rankings: list[list], k: int = RRF_K) -> list:
    """Cormack et al. 2009 RRF. Each `rankings[i]` is a ranked doc list."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, object] = {}
    for ranked in rankings:
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            scores[key]      = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs_by_key[key] = doc
    return [docs_by_key[k_] for k_, _ in
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


class HybridRAGBaseline(BaseBaseline):
    """Dense + BM25 fused via RRF."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        self.generator_chain = GENERATOR_PROMPT | self.llm

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            # Pull a wider pool from each retriever, then fuse.
            saved_topk = self.top_k
            self.top_k = RETRIEVE_N
            dense_docs = self.retrieve_vector(query)
            bm25_docs  = self.retrieve_bm25(query)
            self.top_k = saved_topk

            fused   = reciprocal_rank_fusion([dense_docs, bm25_docs])[:self.top_k]
            context = "\n\n".join(d.page_content for d in fused)

            result = self.generator_chain.invoke({"query": query, "context": context})
            answer = result.content.strip()

            return {
                "answer":          answer,
                "context":         context,
                "n_dense":         len(dense_docs),
                "n_bm25":          len(bm25_docs),
                "n_fused":         len(fused),
                "latency_seconds": round(time.time() - t0, 2),
                "error":           None,
            }
        except Exception as e:
            return {
                "answer":          "",
                "context":         "",
                "latency_seconds": round(time.time() - t0, 2),
                "error":           str(e),
            }
