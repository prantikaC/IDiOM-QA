"""
Search-R1 — RL-trained interleaved reasoning and search-engine calling.

Original paper: Jin et al., arXiv:2503.09516 (Mar 2025)
                 "Search-R1: Training LLMs to Reason and Leverage Search
                  Engines with Reinforcement Learning"
Code:   https://github.com/PeterGriffinJin/Search-R1
Key idea: an RL-trained (DeepSeek-R1-style) policy that autonomously
  interleaves <think>/<search>/<information>/<answer> steps, learning
  purely from an OUTCOME-based reward (final answer correctness) with
  retrieved-token masking for stable training — no explicit process
  reward or intermediate verification step.

Implementation notes
---------------------
The paper releases fine-tuned checkpoints (Llama-3.2-3B, Qwen2.5-7B),
but as with R3-RAG, deploying a foreign checkpoint would break the
single-backend comparison the project chose for baseline fairness (see
r3_rag_baseline.py and the note on dropping Self-RAG for the same
reason). This is a PROMPTED reproduction of the published
think/search/answer loop on our own backend LLM, not the RL-trained
policy — footnote in the paper accordingly.

Verified against the actual `infer.py` in the repo: the real inference
prompt is close to verbatim what's used here — "conduct reasoning inside
<think> and </think>... call a search engine by <search> query
</search>... results between <information> and </information>...
directly provide the answer inside <answer> and </answer>." No
relevance-verification or other intermediate step is present in the real
script — it's a bare think/search/answer loop, full stop.

Note this makes it structurally close to r3_rag_baseline.py's loop once
both are reduced to a prompted reproduction — R3-RAG's real inference
code (verified against its own repo) is also a bare analysis/retrieve/
answer loop with no runtime verification step; an earlier draft of
r3_rag_baseline.py had invented one and was corrected. The two papers'
actual difference is in the TRAINING procedure (SFT cold-start +
dual-reward RL for R3-RAG vs. pure outcome-reward RL for Search-R1), not
the inference-time control flow — a prompted reproduction with no
training step can't express that difference, so don't expect these two
baselines to diverge much in behavior; that's an honest property of this
reproduction method, not a bug.

"Search engine" is adapted to mean the local corpus retrievers
(vector + BM25), consistent with every other baseline in this suite —
there is no live web search over this closed administrative corpus.

Pipeline:
  query → [think → search-or-answer] → (if search: retrieve → inject as
  observation, no verification) → repeat → answer
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


class SearchR1Action(BaseModel):
    thought: str = Field(description="One sentence of reasoning about what's needed next.")
    action: str = Field(description="One of: 'search', 'answer'.")
    search_query: str = Field(description="Query to search, if action='search'. Empty otherwise.")
    answer: str = Field(description="Final answer, if action='answer'. Empty otherwise.")


ACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are answering a question about CSIR/IACS/JU office memoranda using an \
interleaved think-then-act loop (Search-R1-style): reason, optionally \
issue a search query, observe the result, and repeat until you can answer.

ORIGINAL QUERY: {query}

TRAJECTORY SO FAR:
{trajectory}

Decide the next step: either issue ONE focused search query ('search'), \
or, if the trajectory already has enough evidence, give the final answer \
('answer'). Prefer searching for anything specific (rates, dates, \
entitlements, procedures) before answering."""),
])


class SearchR1Baseline(BaseBaseline):
    """Search-R1 adapted for our closed-domain corpus (prompted reproduction)."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        # Search-or-answer decision is critic-analogous → reasoning_llm,
        # matching IDiOM-RAG's own tier allocation for this class of decision.
        self.action_chain = ACTION_PROMPT | self.reasoning_llm.with_structured_output(SearchR1Action)

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
            lines.append(f"Thought {i+1}: {s['thought']}")
            lines.append(f"Search {i+1}: {s['search_query']}")
            lines.append(f"Information {i+1}: {s['information']}")
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
                    decision: SearchR1Action = self.action_chain.invoke(
                        {"query": query, "trajectory": trajectory}
                    )
                except OutputParserException:
                    # Qwen occasionally returns {} for the structured step on
                    # the same question every time; stop stepping and answer
                    # from the evidence gathered so far (fallback below).
                    parse_failed = True
                    break

                if decision.action.strip().lower() == "answer" and decision.answer.strip():
                    return {
                        "answer":          decision.answer.strip(),
                        "context":         "\n\n".join(retrieved_context_parts),
                        "iterations":      i,
                        "latency_seconds": round(time.time() - t0, 2),
                        "error":           None,
                    }

                search_query = decision.search_query.strip() or query
                context = self._retrieve_merged(search_query)
                information = context if context else "No documents found for this search query."
                if context:
                    retrieved_context_parts.append(context)

                steps.append({
                    "thought":      decision.thought,
                    "search_query": search_query,
                    "information":  information,
                })

            # Step budget exhausted — force a final answer from the trajectory so far
            trajectory = self._format_trajectory(steps)
            answer = ""
            if not parse_failed:
                try:
                    final_decision: SearchR1Action = self.action_chain.invoke(
                        {"query": query,
                         "trajectory": trajectory + "\n\n(Step budget exhausted — answer now with what you have.)"}
                    )
                    answer = final_decision.answer.strip()
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
