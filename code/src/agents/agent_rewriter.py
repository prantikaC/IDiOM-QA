from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from src.core.state import MultiAgentRAGState
from src.prompts.rewriter_prompt import REWRITER_SYSTEM_PROMPT


class RewriteDecision(BaseModel):
    """Corrective query rewrite for a sub-task that retrieval failed to resolve.

    The rewriter is responsible only for producing a better-worded retrieval
    target. Deciding whether an answer is genuinely absent from the corpus
    is the critic's and the generator's job — those stages have a wider
    view (final_context_string, past_steps ledger, partial-evidence policy)
    than the rewriter, which only sees one failed retrieval at a time.
    """
    rewritten_task: str = Field(
        description="A rephrased version of the failed sub-task that targets "
                    "the SAME underlying fact but uses different anchor "
                    "entities / vocabulary / scope so similarity-based "
                    "retrievers land in a different index region. Must be "
                    "non-empty — always produce a concrete rephrasing."
    )
    diagnosis: str = Field(
        description="One short sentence diagnosing why the previous retrieval "
                    "failed (e.g., wrong anchor, formal-register mismatch, "
                    "scope too broad, missing date/institute qualifier)."
    )


def get_rewriter_agent(llm: BaseChatModel, override_llm: BaseChatModel = None):
    """
    Factory for the Corrective Query-Rewriter Agent.

    Fires after MAX_CRITIC_RETRIES same-query retries fail. Reads the failed
    sub-task, the last failed retrieval's reranked context, and the
    past_steps ledger; emits a rephrased sub-task and increments
    rewrite_count. The orchestrator picks up the rewritten current_task
    on the next pass with loop_count reset to 0.

    override_llm (config.ENABLE_COMPLEX_CRITIC_ROUTING): when provided and
    the query was flagged complex (state["use_stronger_critic"]), the
    rewrite call uses override_llm instead of llm. Mirrors the critic's
    routing — see agent_critic.py.
    """
    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITER_SYSTEM_PROMPT)
    ])
    rewriter_chain = rewriter_prompt | llm.with_structured_output(RewriteDecision)
    override_chain = (rewriter_prompt | override_llm.with_structured_output(RewriteDecision)
                       if override_llm is not None else None)

    def rewriter_agent(state: MultiAgentRAGState):
        original_query   = state.get("original_query", "")
        current_task     = state.get("current_task", "")
        past_steps       = state.get("past_steps", [])
        last_context     = state.get("final_context_string", "")
        rewrite_count    = state.get("rewrite_count", 0)
        critic_reasoning = state.get("critic_reasoning", "")
        task_variants    = state.get("task_variants", [])

        # Render the ledger compactly, skipping sentinel entries so the prompt
        # only shows genuinely resolved facts.
        resolved = [(t, r) for t, r in past_steps
                    if r not in ("[OUT_OF_SCOPE]", "[RETRIEVAL_FAILED]")]
        ledger_text = (
            "\n".join(f"- {t}\n    → {r}" for t, r in resolved)
            if resolved else "(no prior sub-tasks resolved)"
        )

        variants_text = (
            "\n".join(f"  - {v}" for v in task_variants)
            if task_variants else "(none tried yet)"
        )

        use_override = override_chain is not None and state.get("use_stronger_critic", False)
        active_chain = override_chain if use_override else rewriter_chain

        print(f"\n--- [Rewriter] Same-query retries exhausted; rephrasing sub-task ---")
        print(f"    Failed sub-task: '{current_task}'")
        if critic_reasoning:
            print(f"    Critic diagnosis: {critic_reasoning}")
        if use_override:
            print(f"    Query flagged complex; routing rewrite to override model.")

        try:
            decision: RewriteDecision = active_chain.invoke({
                "original_query":       original_query,
                "current_task":         current_task,
                "past_steps":           ledger_text,
                "final_context_string": last_context or "(empty)",
                "critic_reasoning":     critic_reasoning or "(not available)",
                "task_variants":        variants_text,
            })
        except Exception as e:
            # Fail open: if the rewriter LLM errors, log it and hand back to
            # the orchestrator with the ORIGINAL sub-task. We still consume
            # one rewrite slot so the budget is honest. On the next pass the
            # critic will see the same wording, exhaust its same-query budget
            # again, and force-progress through tier 3.
            print(f"    [Rewriter] Exception: {e}; reusing original sub-task.")
            return {
                "rewrite_count":    rewrite_count + 1,
                "task_variants":    [f"[rewriter_error] {current_task}"],
                "current_task":     current_task,
                "loop_count":       0,
                "gathered_context": None,
                "next":             ["orchestrator"],
            }

        rewritten = (decision.rewritten_task or "").strip()
        if not rewritten:
            # Defensive: structured-output may occasionally return an empty
            # field despite the schema. Treat as an LLM-side failure and
            # fall through with the original sub-task wording, same as
            # the exception branch above.
            print(f"    [Rewriter] Empty rewrite returned; reusing original sub-task.")
            return {
                "rewrite_count":    rewrite_count + 1,
                "task_variants":    [f"[empty_rewrite] {current_task}"],
                "current_task":     current_task,
                "loop_count":       0,
                "gathered_context": None,
                "next":             ["orchestrator"],
            }

        print(f"    [Rewriter] Diagnosis: {decision.diagnosis}")
        print(f"    [Rewriter] Rewritten: '{rewritten}'")

        return {
            "current_task":     rewritten,
            "task_variants":    [rewritten],
            "rewrite_count":    rewrite_count + 1,
            "loop_count":       0,
            "gathered_context": None,        # reset reducer for the next pass
            "next":             ["orchestrator"],
        }

    return rewriter_agent
