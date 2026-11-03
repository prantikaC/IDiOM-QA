"""
DeepRAG — retrieval-augmented reasoning as a step-by-step decision process.

Original paper: Guan et al., arXiv:2502.01142 (Feb 2025)
                 "DeepRAG: Thinking to Retrieval Step by Step for Large
                  Language Models"
Key idea: model retrieval-augmented reasoning as an MDP. The query is
  iteratively decomposed into atomic subqueries; for EACH subquery the
  model makes a binary decision — retrieve external documents, or answer
  from parametric knowledge alone — before moving to the next subquery.
  Stops once enough subqueries have been resolved to answer the original
  question, then synthesizes the final answer from the subquery ledger.

Implementation notes
---------------------
DeepRAG releases fine-tuned checkpoints (DeepRAG-7b on Qwen2.5, DeepRAG-8b
on Llama-3.1 — https://github.com/gxy-gxy/DeepRAG), but as with R3-RAG and
Search-R1, deploying any of them would put this baseline on a different
base LLM than the rest of the suite. This is a PROMPTED reproduction of
the published retrieve-vs-parametric decision procedure on our own
backend LLM, not the original trained policy — same class of caveat as
CRAG/Adaptive-RAG in this suite, footnote in the paper accordingly. Runs
on the same LLM pair as every other prompted baseline (fair,
single-variable comparison — see project decision to drop Self-RAG
rather than keep an off-backend outlier).

Verified against the actual repo (`ICL-prompt/instruction.txt`,
`question_decompose.txt`, `intermediate_answer.txt`, and the real
decision loop in `src/algorithms/sragsamplev2.py`). The authentic
mechanism is Self-Ask-style free-text generation: the fine-tuned model
emits "Follow up: <subquery>" then either "Intermediate answer: <answer>"
directly, or the literal trigger phrase "Let's search the question in
Wikipedia." before retrieving, repeating until "So the final answer is:
<answer long>...</answer long><answer short>...</answer short>". We use
structured output (a Pydantic decision schema) instead of parsing that
exact trigger phrase out of free text, because — unlike DeepRAG's
checkpoint — our backend LLM was never fine-tuned to reliably emit it;
structured output is a more robust substitute for an off-the-shelf
instruct model, not a claim that this is the original prompting format.
"Wikipedia" is replaced with the internal document corpus throughout
(closed-domain adaptation, same as every other baseline here). The
Follow-up/Intermediate-answer vocabulary from the real ICL prompts is
kept in the trajectory formatting shown back to the model each step,
since that's cheap to match exactly and keeps the trace closer to their
actual few-shot format.

The retrieve-vs-parametric decision is a critic-analogous judgment (does
this subquery need external evidence or not?), so it runs on
reasoning_llm, matching IDiOM-RAG's own tier allocation for that class
of decision. Subquery generation and final synthesis are cheap
formatting/generation steps and stay on fast_llm.

Pipeline:
  query → [Follow up: subquery → retrieve-or-parametric → Intermediate
  answer]* → So the final answer is: synthesize
"""

import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


MAX_STEPS = 5  # Raised from 2 (2026-10-28) to match IDiOM-RAG's own cap of 5
# sub-tasks per question (config.MAX_PLAN_STEPS). At 2 steps this baseline refused
# answerable questions whenever it ran out of steps, which handicapped it relative
# to IDiOM-RAG. The 2-step run is archived in data/eval/full475_matched_agentic_maxsteps2/.


# ── Structured step decision ────────────────────────────────────────────────

class DeepRAGStep(BaseModel):
    done: bool = Field(
        description="True if enough subqueries have been resolved to fully "
                     "answer the ORIGINAL query."
    )
    subquery: str = Field(
        description="If not done: the next atomic subquery to resolve. "
                     "Empty string if done=True."
    )
    action: str = Field(
        description="If not done: 'retrieve' if the subquery needs external "
                     "documents to answer, 'parametric' if it can be answered "
                     "directly from general/well-known knowledge without "
                     "looking anything up. Ignored if done=True."
    )


STEP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a helpful Retrieve-Augmented Generation model answering a question \
about CSIR/IACS/JU office memoranda. Decompose the question into clear, \
logical sub-questions and iteratively answer it, one "Follow up" at a \
time (DeepRAG-style). For each Follow up, decide whether you can give a \
direct Intermediate answer, or whether you need to search the internal \
document corpus first.

Question: {query}

{history}

Decide the NEXT step:
- If the Follow ups resolved so far are enough to fully answer the \
Question, set done=True.
- Otherwise, propose ONE new Follow up (not a repeat of an earlier one) \
and decide whether it needs document retrieval ('retrieve') or can be \
answered from general/well-known knowledge alone ('parametric'). \
Administrative specifics (rates, dates, entitlements, procedures) almost \
always need 'retrieve'; only use 'parametric' for genuinely general \
knowledge (e.g. acronym expansions everyone would know)."""),
])

SUBANSWER_RETRIEVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
Answer the subquery using ONLY the provided context. Be concise — one to \
two sentences. If the context doesn't contain the answer, say so plainly.

CONTEXT:
{context}

SUBQUERY: {subquery}

ANSWER:"""),
])

SUBANSWER_PARAMETRIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
Answer this subquery concisely from your own general knowledge, without \
looking anything up.

SUBQUERY: {subquery}

ANSWER:"""),
])

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an administrative assistant for CSIR/IACS/JU. Using the resolved \
Follow ups below, give the final answer to the Question — this is the \
"So the final answer is:" step. If the trajectory doesn't contain enough \
information, say:
"I could not find sufficient information in the internal documents."

Question: {query}

{history}

So the final answer is:"""),
])


class DeepRAGBaseline(BaseBaseline):
    """DeepRAG adapted for our closed-domain corpus (prompted reproduction)."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        # Retrieve-vs-parametric decision is critic-analogous → reasoning_llm
        self.step_chain = STEP_PROMPT | self.reasoning_llm.with_structured_output(DeepRAGStep)
        self.retrieve_answer_chain   = SUBANSWER_RETRIEVE_PROMPT   | self.llm
        self.parametric_answer_chain = SUBANSWER_PARAMETRIC_PROMPT | self.llm
        self.synthesize_chain        = SYNTHESIZE_PROMPT           | self.llm

    def _retrieve_merged(self, query: str) -> tuple[list, str]:
        vec_docs  = self.retrieve_vector(query)
        bm25_docs = self.retrieve_bm25(query)
        seen, merged = set(), []
        for doc in vec_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        context = "\n\n".join(d.page_content for d in merged[: self.top_k])
        return merged, context

    @staticmethod
    def _format_history(ledger: list[dict]) -> str:
        if not ledger:
            return "(no Follow ups yet)"
        return "\n".join(
            f"Follow up: {step['subquery']}\nIntermediate answer: {step['answer']}"
            for step in ledger
        )

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            ledger: list[dict] = []
            retrieved_context_parts: list[str] = []

            for _ in range(MAX_STEPS):
                history_text = self._format_history(ledger)
                step: DeepRAGStep = self.step_chain.invoke(
                    {"query": query, "history": history_text}
                )

                if step.done or not step.subquery.strip():
                    break

                action = step.action.strip().lower()
                if action == "retrieve":
                    _, sub_context = self._retrieve_merged(step.subquery)
                    sub_answer = self.retrieve_answer_chain.invoke(
                        {"subquery": step.subquery, "context": sub_context}
                    ).content.strip()
                    if sub_context:
                        retrieved_context_parts.append(sub_context)
                else:
                    action = "parametric"
                    sub_answer = self.parametric_answer_chain.invoke(
                        {"subquery": step.subquery}
                    ).content.strip()

                ledger.append({
                    "subquery": step.subquery.strip(),
                    "action":   action,
                    "answer":   sub_answer,
                })

            history_text = self._format_history(ledger)
            final_answer = self.synthesize_chain.invoke(
                {"query": query, "history": history_text}
            ).content.strip()
            full_context = "\n\n".join(retrieved_context_parts)

            return {
                "answer":          final_answer,
                "context":         full_context,
                "iterations":      len(ledger),
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
