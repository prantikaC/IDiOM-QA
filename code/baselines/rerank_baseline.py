"""
Rerank RAG — dense retrieve + cross-encoder reranking.

References:
  Cross-encoder rerank: Nogueira & Cho 2019 (arXiv:1901.04085),
    "Passage Re-ranking with BERT"
  Reranker model:       BAAI/bge-reranker-base (Xiao et al. 2023)
  Taxonomy:             Gao et al. 2023 (arXiv:2312.10997) §3.2 "Advanced RAG"

Pipeline:
  query → dense retrieve top-N (N=20) →
          cross-encoder score each (query, doc) pair →
          take top-k by reranker score → concat → generator

Dense retrieval gives recall (cast a wide net); cross-encoder gives
precision (re-score with full query-doc attention, which the bi-encoder
embedding can't see).
"""

import time
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

from baselines.base_baseline import BaseBaseline
from src.core.device import get_device


RERANK_MODEL = "BAAI/bge-reranker-base"
RETRIEVE_N   = 20  # wide dense pool before reranking


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


class RerankRAGBaseline(BaseBaseline):
    """Dense retrieve (wide) + cross-encoder rerank (precise)."""

    _shared_reranker = None  # class-level cache; loading is non-trivial

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        self.generator_chain = GENERATOR_PROMPT | self.llm

        if RerankRAGBaseline._shared_reranker is None:
            print(f"[RerankRAG] Loading reranker {RERANK_MODEL} on {get_device()}...")
            RerankRAGBaseline._shared_reranker = CrossEncoder(
                RERANK_MODEL, device=get_device()
            )
        self.reranker = RerankRAGBaseline._shared_reranker

    def _rerank(self, query: str, docs: list) -> list:
        if not docs:
            return docs
        pairs  = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        order  = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[:self.top_k]]

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            saved_topk = self.top_k
            self.top_k = RETRIEVE_N
            wide_docs = self.retrieve_vector(query)
            self.top_k = saved_topk

            top_docs = self._rerank(query, wide_docs)
            context  = "\n\n".join(d.page_content for d in top_docs)

            result = self.generator_chain.invoke({"query": query, "context": context})
            answer = result.content.strip()

            return {
                "answer":          answer,
                "context":         context,
                "n_retrieved":     len(wide_docs),
                "n_after_rerank":  len(top_docs),
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
