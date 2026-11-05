import math
import os

from sentence_transformers import CrossEncoder
from src.core.state import MultiAgentRAGState
from src.core.recency import recency_weight, is_superseded
import config

def get_reranker_agent(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                       top_k: int = 5,
                       recency_lambda: float | None = None):
    """
    Factory function that returns the Reranker Agent.

    Combines three signals into the final ordering:
      1. Cross-encoder relevance (raw ms-marco logits, roughly [-11, +11]).
      2. Recency boost — newer OMs are preferred when relevance ties.
      3. Supersession penalty — OMs flagged as superseded by reconcile_supersession
         are heavily demoted (but kept in the pool for historical queries).

    Negative cross-encoder logits would invert multiplicative boosts, so the
    logit is first squashed through a sigmoid into [0, 1] before the boost
    and penalty are applied. The raw logit is still surfaced as
    `top_rerank_score` so the Critic's existing threshold semantics survive.
    """
    encoder = CrossEncoder(model_name)
    lam = recency_lambda if recency_lambda is not None else config.RECENCY_LAMBDA_RERANKER

    def reranker_agent(state: MultiAgentRAGState):
        print("--- [Reranker Agent] Evaluating and sorting gathered context ---")
        # Score against the focused sub-task, not the full original query.
        # Each retriever searched for current_task; scoring against the full
        # multi-hop query penalises docs that are highly relevant to the
        # sub-task but tangential to the overall question.
        query = state.get("current_task") or state["original_query"]
        raw_docs = state.get("gathered_context", [])

        if not raw_docs:
            print("--- [Reranker Agent] No context gathered by agents. ---")
            return {"final_context_string": "No relevant information found.",
                    "top_rerank_score": float("-inf"), "top_doc_sources": []}

        # ===========================
        # Deduplicate & Merge Sources
        # ===========================
        unique_docs = []
        seen_content = set()

        for doc in raw_docs:
            content_hash = hash(doc.page_content.strip())

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
            else:
                # If it's a duplicate, find the original and append the new source to its metadata!
                for u_doc in unique_docs:
                    if hash(u_doc.page_content.strip()) == content_hash:
                        existing_source = u_doc.metadata.get("agent_source", "unknown")
                        new_source = doc.metadata.get("agent_source", "unknown")
                        if new_source not in existing_source:
                            u_doc.metadata["agent_source"] = f"{existing_source}, {new_source}"
                        break

        print(f"--- [Reranker Agent] Deduplicated from {len(raw_docs)} down to {len(unique_docs)} unique chunks. ---")
        # ==========================================

        # Format inputs for the Cross-Encoder using the UNIQUE docs
        pairs = [[query, doc.page_content] for doc in unique_docs]

        # Predict relevance scores (raw ms-marco logits, may be negative)
        scores = encoder.predict(pairs)

        # Combine relevance × recency × supersession into a single ordering signal.
        # Critic still consumes `rerank_score` (raw logit) for its threshold gate.
        for doc, score in zip(unique_docs, scores):
            raw_logit = float(score)
            # Sigmoid → [0, 1] so multiplicative boosts behave correctly for
            # negative logits too. Probabilistic interpretation is convenient
            # but not required — we only need monotonicity.
            relevance_prob = 1.0 / (1.0 + math.exp(-raw_logit))
            r_weight = recency_weight(doc)
            boosted = relevance_prob * (1.0 + lam * r_weight)
            if is_superseded(doc):
                boosted *= config.SUPERSEDED_PENALTY

            doc.metadata["rerank_score"]         = raw_logit          # critic threshold
            doc.metadata["rerank_score_boosted"] = round(boosted, 4)
            doc.metadata["recency_weight"]       = round(r_weight, 3)

        # Sort by the boosted score so recency + supersession actually move docs
        unique_docs.sort(key=lambda x: x.metadata["rerank_score_boosted"], reverse=True)

        # Graph facts must clear a relevance floor and are capped, so generic
        # edge matches don't displace text chunks (see config.KG_MIN_RERANK_SCORE).
        kept, n_kg = [], 0
        for d in unique_docs:
            if d.metadata.get("agent_source") == "kg_retriever":
                if d.metadata["rerank_score"] < config.KG_MIN_RERANK_SCORE or n_kg >= config.KG_MAX_IN_CONTEXT:
                    continue
                n_kg += 1
            kept.append(d)
        unique_docs = kept

        # Keep only the top_k most relevant documents
        top_docs = unique_docs[:top_k]

        # Compile the final context string for the generation LLM.
        # Include the OM's issue_date in the per-chunk header so the generator's
        # "prefer the most recent OM" rule has a parseable signal even if a
        # given chunk's body happened to lose the prefix.
        def _hdr(d):
            src  = d.metadata.get("agent_source", "unknown")
            date = d.metadata.get("issue_date", "UNKNOWN")
            sup  = " [SUPERSEDED]" if is_superseded(d) else ""
            return f"Source ({src}) | Date: {date}{sup}: "
        formatted_context = "\n\n---\n\n".join(
            [_hdr(d) + d.page_content for d in top_docs]
        )

        # Surface the best raw cross-encoder logit so the Critic can short-circuit
        # the LLM call when retrieval is clearly strong — this is the value the
        # critic compares against CRITIC_ACCEPT_SCORE (raw logit scale).
        top_score = top_docs[0].metadata["rerank_score"] if top_docs else float("-inf")

        # OM of each kept chunk: text chunks carry source_file, KG facts carry source_om.
        def _om(d):
            om = d.metadata.get("source_om") or d.metadata.get("source_file") or ""
            return os.path.splitext(os.path.basename(om))[0]
        top_doc_sources = [{"om": _om(d), "text": d.page_content} for d in top_docs]

        return {"final_context_string": formatted_context, "top_rerank_score": top_score,
                "top_doc_sources": top_doc_sources}

    return reranker_agent
