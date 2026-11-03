"""
R3-RAG — interleaved reasoning and retrieval, RL-trained end-to-end.

Original paper: Li et al., arXiv:2505.23794 (May 2025)
                 "R3-RAG: Learning Step-by-Step Reasoning and Retrieval for
                  LLMs via Reinforcement Learning"
Code:   https://github.com/Yuan-Li-FNLP/R3-RAG
Key idea: cold-start SFT teaches the model to interleave analysis/search
  steps with retrieval, then RL refines the policy with TWO rewards —
  (1) outcome reward: does the trajectory reach the correct final answer,
  (2) process reward: is each retrieved document actually relevant to the
  question. Both rewards shape TRAINING; neither is a separate step the
  deployed model runs at inference time.

Implementation notes
---------------------
The paper releases a fine-tuned checkpoint (R3-RAG-Llama, Llama-3.1-8B),
but using it would put this baseline on a different base LLM than the
rest of the suite — the project chose single-backend fairness over
checkpoint authenticity here (same reasoning that led to dropping
Self-RAG rather than keeping it as an off-backend outlier; Self-RAG's
reflection tokens are a trained vocabulary that genuinely can't be
prompted, whereas R3-RAG's interleaved-reasoning PROCEDURE can be). This
is a PROMPTED reproduction of that procedure on our own backend LLM, not
the RL-trained policy — footnote in the paper accordingly.

Verified against the actual inference code (startup/R3-RAG.py's
`split_response`, which parses "The problem analysis:" / "The retrieval
query:" / "The final answer:" segments out of the model's own free-form
generation — see `benchmark/R3-RAG/src/inference.py` for the
non-visualization version of the same loop). Two corrections from an
earlier draft of this file, which invented mechanisms not present in the
real repo:
  1. There is NO separate relevance-verification call at inference time.
     The "process reward" is a TRAINING-time signal (shapes the RL
     policy to prefer useful searches); the deployed model just retrieves
     and keeps going. An earlier version of this file added an explicit
     post-retrieval verification step that doesn't exist in the actual
     system — removed.
  2. The step vocabulary is analysis/retrieval-query/final-answer, not
     generic think/search/answer — aligned to match.
Net effect: without the RL training this repo actually runs, a faithful
PROMPTED reproduction of R3-RAG's inference control flow is structurally
very close to search_r1_baseline.py's loop (see that file — Search-R1's
real inference format is also a bare think/search/answer loop, verified
against its own infer.py). The two papers' real difference is in HOW the
policy was trained (SFT cold-start + dual-reward RL vs. pure outcome-
reward RL from a base model), not in the inference-time control flow —
a prompted reproduction with no training step cannot express that
difference, and pretending otherwise (as the earlier draft did with the
invented verification gate) would be less honest than just saying so.

Pipeline (matches the real step vocabulary):
  query → [analysis → retrieval-query-or-final-answer] → (if query:
  retrieve → inject as "retrieval documents") → repeat → final answer
"""

import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline
from baselines.naive_rag_baseline import GENERATOR_PROMPT as EVIDENCE_ANSWER_PROMPT
from langchain_core.exceptions import OutputParserException


MAX_STEPS = 5  # Raised from 2 (2026-10-28) to match IDiOM-RAG's own cap of 5
# sub-tasks per question (config.MAX_PLAN_STEPS). At 2 steps this baseline refused
# answerable questions whenever it ran out of steps, which handicapped it relative
# to IDiOM-RAG. The 2-step run is archived in data/eval/full475_matched_agentic_maxsteps2/.


class R3Step(BaseModel):
    analysis: str = Field(description="One sentence of problem analysis: what's needed next.")
    action: str = Field(description="One of: 'retrieve', 'answer'.")
    retrieval_query: str = Field(description="Query to retrieve, if action='retrieve'. Empty otherwise.")
    final_answer: str = Field(description="Final answer, if action='answer'. Empty otherwise.")


STEP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are answering a question about CSIR/IACS/JU office memoranda by \
interleaving problem analysis with retrieval (R3-RAG-style).

THE QUESTION: {query}

TRAJECTORY SO FAR:
{trajectory}

Give the problem analysis for the next step, then decide: issue ONE \
retrieval query for evidence you still need ('retrieve'), or, if the \
trajectory already has enough evidence, give the final answer \
('answer'). Prefer retrieving for anything specific (rates, dates, \
entitlements, procedures) before answering."""),
])


class R3RAGBaseline(BaseBaseline):
    """R3-RAG adapted for our closed-domain corpus (prompted reproduction)."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        # Retrieve-vs-answer decision is critic-analogous → reasoning_llm,
        # matching IDiOM-RAG's own tier allocation for this class of decision.
        self.step_chain = STEP_PROMPT | self.reasoning_llm.with_structured_output(R3Step)

    def _retrieve_merged(self, query: str) -> str:
        vec_docs  = self.retrieve_vector(query)
        bm25_docs = self.retrieve_bm25(query)
        seen, merged = set(), []
        for doc in vec_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return "\n\n".join(d.page_content for d in merged[: self.top_k])

    @staticmethod
    def _format_trajectory(steps: list[dict]) -> str:
        if not steps:
            return "(nothing yet)"
        lines = []
        for i, s in enumerate(steps):
            lines.append(f"The problem analysis {i+1}: {s['analysis']}")
            lines.append(f"The retrieval query {i+1}: {s['retrieval_query']}")
            lines.append(f"The retrieval documents {i+1}: {s['documents']}")
        return "\n".join(lines)

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            steps: list[dict] = []
            retrieved_context_parts: list[str] = []
            parse_failed = False

            for i in range(MAX_STEPS):
                trajectory = self._format_trajectory(steps)
                try:
                    step: R3Step = self.step_chain.invoke(
                        {"query": query, "trajectory": trajectory}
                    )
                except OutputParserException:
                    # Qwen occasionally returns {} for the structured step on
                    # the same question every time; stop stepping and answer
                    # from the evidence gathered so far (fallback below).
                    parse_failed = True
                    break

                if step.action.strip().lower() == "answer" and step.final_answer.strip():
                    return {
                        "answer":          step.final_answer.strip(),
                        "context":         "\n\n".join(retrieved_context_parts),
                        "iterations":      i,
                        "latency_seconds": round(time.time() - t0, 2),
                        "error":           None,
                    }

                retrieval_query = step.retrieval_query.strip() or query
                documents = self._retrieve_merged(retrieval_query)
                if documents:
                    retrieved_context_parts.append(documents)
                else:
                    documents = "No documents found for this query."

                steps.append({
                    "analysis":        step.analysis,
                    "retrieval_query": retrieval_query,
                    "documents":       documents,
                })

            # Step budget exhausted — force a final answer from the trajectory so far
            trajectory = self._format_trajectory(steps)
            answer = ""
            if not parse_failed:
                try:
                    final_step: R3Step = self.step_chain.invoke(
                        {"query": query,
                         "trajectory": trajectory + "\n\n(Step budget exhausted — answer now with what you have.)"}
                    )
                    answer = final_step.final_answer.strip()
                except OutputParserException:
                    parse_failed = True
            if not answer:
                # The model still wanted to search. Answer from everything gathered
                # so far (Naive RAG's prompt), so running out of steps never turns
                # into an automatic refusal; it refuses only if the evidence lacks it.
                evidence = "\n\n".join(retrieved_context_parts) or self._retrieve_merged(query)
                retrieved_context_parts = [evidence]  # report what the answer used
                answer = (EVIDENCE_ANSWER_PROMPT | self.llm).invoke(
                    {"query": query, "context": evidence}
                ).content.strip()

            return {
                "answer":          answer,
                "context":         "\n\n".join(retrieved_context_parts),
                "iterations":      len(steps),
                "parse_failed":    parse_failed,
                "latency_seconds": round(time.time() - t0, 2),
                "error":           None,
            }

        except Exception as e:
            return {
                "answer":          "",
                "context":         "",
                "iterations":      0,
                "latency_seconds": round(time.time() - t0, 2),
                "error":           str(e),
            }
