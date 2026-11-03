from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from src.core.state import MultiAgentRAGState
from src.core.recency import rank_by_recency_boost, proxy_relevance_score
import config

def get_vector_agent(vectorstore: VectorStore,
                     top_k: int = 5,
                     overfetch: int | None = None,
                     recency_lambda: float | None = None):
    """
    Factory function that returns the Vector Retrieval Agent.

    Over-fetches `top_k * overfetch` candidates, applies the recency boost
    and supersession penalty defined in src/core/recency.py, then trims back
    to `top_k`. This biases the candidate pool toward recent OMs without
    eliminating older relevant ones — they survive into the reranker via the
    cross-encoder score that still dominates the boost magnitude.
    """
    overfetch = overfetch if overfetch is not None else config.RETRIEVE_OVERFETCH
    lam = recency_lambda if recency_lambda is not None else config.RECENCY_LAMBDA_RETRIEVER
    fetch_k = max(top_k, top_k * overfetch)

    def vector_retrieval_agent(state: MultiAgentRAGState):
        current_task = state.get("current_task") or state.get("original_query")

        print(f"\n--- [Vector Agent] Searching dense space for: '{current_task}' "
              f"(fetch_k={fetch_k}, top_k={top_k}, λ={lam}) ---")

        # similarity_search_with_score returns (Document, distance) for FAISS,
        # where lower distance = more similar.  Convert to a similarity score
        # so the recency multiplier behaves intuitively (bigger = better).
        try:
            scored = vectorstore.similarity_search_with_score(current_task, k=fetch_k)
            docs = [d for d, _ in scored]
            # FAISS L2 distance is unbounded ≥ 0; flip to similarity = 1/(1+dist)
            base = [1.0 / (1.0 + float(s)) for _, s in scored]
        except (AttributeError, NotImplementedError):
            # Vector store doesn't expose scores — fall back to a ranked proxy
            docs = vectorstore.similarity_search(current_task, k=fetch_k)
            base = [proxy_relevance_score(i, len(docs)) for i in range(len(docs))]

        # Apply recency boost + supersession penalty, then trim
        top_docs = rank_by_recency_boost(docs, base, lam=lam, top_k=top_k)

        # Tag documents with their source agent
        for doc in top_docs:
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["agent_source"] = "vector_retriever"

        # Runs in parallel with the other retrievers; the state reducer merges
        # each retriever's docs, so return only our own.
        return {"gathered_context": top_docs}

    return vector_retrieval_agent
