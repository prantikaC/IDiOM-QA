from langchain_community.retrievers import BM25Retriever
from src.core.state import MultiAgentRAGState
from src.core.recency import rank_by_recency_boost, proxy_relevance_score
import config

def get_bm25_agent(bm25_retriever: BM25Retriever,
                   top_k: int = 5,
                   overfetch: int | None = None,
                   recency_lambda: float | None = None):
    """
    Factory function that returns the BM25 Lexical Retrieval Agent.

    Like the vector agent: over-fetch a wider candidate pool, apply the
    recency boost + supersession penalty, trim back to `top_k`. BM25Retriever
    does not expose per-document BM25 scores via .invoke(), so we use a
    rank-proxy score from src.core.recency.proxy_relevance_score().
    """
    overfetch = overfetch if overfetch is not None else config.RETRIEVE_OVERFETCH
    lam = recency_lambda if recency_lambda is not None else config.RECENCY_LAMBDA_RETRIEVER
    fetch_k = max(top_k, top_k * overfetch)

    # BM25Retriever's k is set at construction. Bump it so each call yields
    # the over-fetched pool we want to re-rank.
    bm25_retriever.k = fetch_k

    def bm25_retrieval_agent(state: MultiAgentRAGState):
        current_task = state.get("current_task") or state.get("original_query")

        print(f"\n--- [BM25 Agent] Performing lexical search for: '{current_task}' "
              f"(fetch_k={fetch_k}, top_k={top_k}, λ={lam}) ---")

        # Perform the actual keyword search using the focused task
        retrieved_docs = bm25_retriever.invoke(current_task)

        # BM25Retriever returns a ranked list without exposing raw scores;
        # use a monotonically decreasing rank proxy so the recency multiplier
        # has something meaningful to scale.
        base = [proxy_relevance_score(i, len(retrieved_docs))
                for i in range(len(retrieved_docs))]

        top_docs = rank_by_recency_boost(retrieved_docs, base, lam=lam, top_k=top_k)

        # Tag documents with their source agent
        for doc in top_docs:
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["agent_source"] = "bm25_retriever"

        # Runs in parallel with the other retrievers; the state reducer merges
        # each retriever's docs, so return only our own.
        return {"gathered_context": top_docs}

    return bm25_retrieval_agent
