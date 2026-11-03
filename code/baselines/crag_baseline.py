"""
Corrective RAG (CRAG) — closed-domain adaptation.

Original paper: Yan et al., 2024 (arXiv:2401.15884)
  (first author is Shi-Qi Yan — "Shi-Qi" is the given name, "Yan" the
  surname; confirmed via arXiv's own author-index link format, which
  lists "Yan, S". An earlier version of this docstring cited it as
  "Shi et al.", a name-parsing error.)
Key idea: a lightweight relevance evaluator judges retrieved docs as
  Correct / Incorrect / Ambiguous, then applies knowledge refinement
  (strip-then-search) before generation.

Closed-domain adaptation (no web search):
  - Evaluator: GPT-4o-mini with structured output (replaces fine-tuned T5)
  - Correct  → pass docs directly to generator
  - Ambiguous → knowledge refinement (extract key sentences) + re-retrieve
  - Incorrect → knowledge refinement on the full corpus (BM25 fallback)
  - No web search; "Incorrect" falls back to BM25 rather than Google

Pipeline:
  query → Vector retrieval → Evaluator → [Refine →] Generator
"""

import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


# ── Structured evaluator output ────────────────────────────────────────────────

class RelevanceDecision(BaseModel):
    verdict: str = Field(
        description="One of: 'Correct', 'Incorrect', 'Ambiguous'"
    )
    reasoning: str = Field(
        description="One sentence explaining the verdict."
    )


EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a relevance evaluator for an administrative document RAG system.

QUERY:
{query}

RETRIEVED CONTEXT:
{context}

Classify the retrieved context:
- "Correct"   — the context directly and sufficiently answers the query.
- "Ambiguous" — the context is partially relevant but incomplete or unclear.
- "Incorrect" — the context is irrelevant or off-topic.

Return ONLY the structured verdict."""),
])

REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a knowledge refiner. Extract ONLY the sentences from the passage
below that are directly relevant to answering the query.
Return them verbatim, separated by newlines.
If nothing is relevant, return an empty string.

QUERY: {query}
PASSAGE:
{passage}"""),
])

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an administrative assistant for CSIR/IACS/JU.
Answer the query using ONLY the provided context.
If the context does not contain the answer, say:
"I could not find sufficient information in the internal documents."

CONTEXT:
{context}

QUERY: {query}

ANSWER:"""),
])


class CRAGBaseline(BaseBaseline):
    """Corrective RAG adapted for our closed-domain corpus."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        structured_llm = self.llm.with_structured_output(RelevanceDecision)
        self.evaluator_chain = EVALUATOR_PROMPT | structured_llm
        self.refinement_chain = REFINEMENT_PROMPT | self.llm
        self.generator_chain  = GENERATOR_PROMPT | self.llm

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _evaluate(self, query: str, context: str) -> RelevanceDecision:
        return self.evaluator_chain.invoke({"query": query, "context": context})

    def _refine(self, query: str, docs: list) -> str:
        """Extract relevant sentences from each doc, then concatenate."""
        refined_parts = []
        for doc in docs:
            result = self.refinement_chain.invoke(
                {"query": query, "passage": doc.page_content}
            )
            text = result.content.strip()
            if text:
                refined_parts.append(text)
        return "\n\n".join(refined_parts)

    def _generate(self, query: str, context: str) -> str:
        result = self.generator_chain.invoke({"query": query, "context": context})
        return result.content.strip()

    # ── Main pipeline ───────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            # Step 1 — initial retrieval
            docs = self.retrieve_vector(query)
            context = "\n\n".join(d.page_content for d in docs)

            # Step 2 — relevance evaluation
            decision: RelevanceDecision = self._evaluate(query, context)
            verdict = decision.verdict.strip()

            if verdict == "Correct":
                final_context = context

            elif verdict == "Ambiguous":
                # Refine vector docs + supplement with BM25
                refined = self._refine(query, docs)
                bm25_docs = self.retrieve_bm25(query)
                bm25_ctx  = "\n\n".join(d.page_content for d in bm25_docs)
                final_context = refined + "\n\n" + bm25_ctx if refined else bm25_ctx

            else:  # "Incorrect" — skip vector results, use BM25 only
                bm25_docs = self.retrieve_bm25(query)
                if bm25_docs:
                    final_context = self._refine(query, bm25_docs)
                else:
                    final_context = ""

            # Step 3 — generation
            answer = self._generate(query, final_context)

            return {
                "answer":           answer,
                "context":          final_context,
                "verdict":          verdict,
                "latency_seconds":  round(time.time() - t0, 2),
                "error":            None,
            }

        except Exception as e:
            return {
                "answer":          "",
                "context":         "",
                "verdict":         "error",
                "latency_seconds": round(time.time() - t0, 2),
                "error":           str(e),
            }
