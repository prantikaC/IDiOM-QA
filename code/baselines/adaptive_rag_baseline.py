"""
Adaptive-RAG — closed-domain adaptation.

Original paper: Jeong et al., 2024 (arXiv:2403.14403)
Key idea: a lightweight complexity classifier routes each query to one of:
  A — No retrieval (simple factoid answerable by LLM alone)
  B — Single-step retrieval
  C — Multi-step iterative retrieval

Closed-domain adaptation:
  - Classifier: fast_llm few-shot (replaces trained T5-Large) — matches
    IDiOM-RAG's own planner complexity classifier, which also stays on
    fast_llm rather than reasoning_llm.
  - Strategy A: direct generation (no retrieval)
  - Strategy B: single vector+BM25 retrieval → generate
  - Strategy C: up to 3 iterative retrieval rounds with follow-up queries.
    The follow-up-query generator is the step that judges what's missing
    and drives the next retrieval round — the same role IDiOM-RAG's
    rewriter plays — so it runs on reasoning_llm, not fast_llm.

Pipeline:
  query → Classifier → [A | B | C] → Generator
"""

import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


# ── Complexity classifier ───────────────────────────────────────────────────────

class ComplexityDecision(BaseModel):
    strategy: str = Field(
        description="One of: 'A', 'B', 'C'"
    )
    reasoning: str = Field(
        description="One sentence justifying the strategy choice."
    )


CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a query complexity classifier for an administrative document Q&A system
covering office memoranda from CSIR, IACS, and Jadavpur University.

Classify the user query into exactly one retrieval strategy:

A — No retrieval needed. The query asks for a well-known fact that a capable
    LLM can answer directly (e.g., "What does OM stand for?").
B — Single-step retrieval. One targeted search is enough to answer the query
    (e.g., "What is the LTC entitlement for a Group-B officer?").
C — Multi-step retrieval. The query requires combining information from
    multiple documents or sub-questions
    (e.g., "How did the leave policy for contract employees change from 2018 to 2023?").

Examples:
  Query: "What is the full form of CSIR?"          → A
  Query: "What is the medical advance limit?"       → B
  Query: "Compare HRA rates before and after 2020." → C

Now classify:
QUERY: {query}"""),
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

FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are helping with a multi-step research task.
Based on the original query and the context gathered so far, generate ONE
focused follow-up search query that would retrieve the MISSING information
needed to fully answer the original question.
Return ONLY the follow-up query string, nothing else.

ORIGINAL QUERY: {query}
CONTEXT SO FAR:
{context}"""),
])


class AdaptiveRAGBaseline(BaseBaseline):
    """Adaptive-RAG adapted for our closed-domain corpus."""

    MAX_ITERATIONS = 3  # Strategy C cap

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        structured_llm = self.llm.with_structured_output(ComplexityDecision)
        self.classifier_chain = CLASSIFIER_PROMPT | structured_llm
        self.generator_chain  = GENERATOR_PROMPT  | self.llm
        # Follow-up generator drives Strategy C's iterative loop — the
        # rewriter-analogous judgment step, promoted to reasoning_llm to
        # match IDiOM-RAG's tier for the same role.
        self.followup_chain   = FOLLOWUP_PROMPT   | self.reasoning_llm

    # ── Strategy implementations ────────────────────────────────────────────────

    def _strategy_a(self, query: str) -> tuple[str, str]:
        """Direct generation with no retrieval."""
        result = self.generator_chain.invoke({"query": query, "context": ""})
        return result.content.strip(), ""

    def _strategy_b(self, query: str) -> tuple[str, str]:
        """Single-step: combine vector + BM25, dedup, generate."""
        vec_docs  = self.retrieve_vector(query)
        bm25_docs = self.retrieve_bm25(query)
        seen, merged = set(), []
        for doc in vec_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        context = "\n\n".join(d.page_content for d in merged[: self.top_k * 2])
        result  = self.generator_chain.invoke({"query": query, "context": context})
        return result.content.strip(), context

    def _strategy_c(self, query: str) -> tuple[str, str]:
        """
        Multi-step: iteratively retrieve and generate follow-up queries
        until MAX_ITERATIONS is exhausted.
        """
        accumulated_context = ""
        current_query = query

        for _ in range(self.MAX_ITERATIONS):
            vec_docs  = self.retrieve_vector(current_query)
            bm25_docs = self.retrieve_bm25(current_query)
            seen, merged = set(), []
            for doc in vec_docs + bm25_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)

            new_context = "\n\n".join(d.page_content for d in merged[: self.top_k])
            accumulated_context = (accumulated_context + "\n\n" + new_context).strip()

            # Generate a follow-up query for the next iteration
            followup = self.followup_chain.invoke(
                {"query": query, "context": accumulated_context}
            )
            current_query = followup.content.strip()

            # Stop early if follow-up looks like "nothing more needed"
            if not current_query or len(current_query) < 5:
                break

        result = self.generator_chain.invoke(
            {"query": query, "context": accumulated_context}
        )
        return result.content.strip(), accumulated_context

    # ── Main pipeline ───────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            decision: ComplexityDecision = self.classifier_chain.invoke(
                {"query": query}
            )
            strategy = decision.strategy.strip().upper()

            if strategy == "A":
                answer, context = self._strategy_a(query)
            elif strategy == "C":
                answer, context = self._strategy_c(query)
            else:  # default to B for any unexpected value
                strategy = "B"
                answer, context = self._strategy_b(query)

            return {
                "answer":          answer,
                "context":         context,
                "strategy":        strategy,
                "latency_seconds": round(time.time() - t0, 2),
                "error":           None,
            }

        except Exception as e:
            return {
                "answer":          "",
                "context":         "",
                "strategy":        "error",
                "latency_seconds": round(time.time() - t0, 2),
                "error":           str(e),
            }
