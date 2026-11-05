from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from src.core.state import MultiAgentRAGState
from src.prompts.critic_prompt import CRITIC_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT_ENTITY_CHECK
import re
import config


def attribute_fact(fact: str, docs: list) -> list:
    """OMs of the kept chunks the critic's accepted fact came from, best match first.
    Word overlap, with numbers and dates (amounts, order numbers) counted three times,
    since they identify the right notice among near-identical siblings. A fact that
    merges values from several notices also gets the OM of each value the best
    match does not contain (bare years and short numbers are too common to count)."""
    words = lambda s: set(re.findall(r"[a-z]{3,}|\d[\d,./-]*", s.lower()))
    fw = words(fact)
    scored = []
    for d in docs:
        if not d.get("om"):
            continue
        dw = words(d.get("text", ""))
        shared = fw & dw
        scored.append((sum(3 if w[0].isdigit() else 1 for w in shared), d["om"], dw))
    scored = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])
    if not scored:
        return []
    oms, covered = [scored[0][1]], set(scored[0][2])
    values = [w for w in fw if w[0].isdigit() and len(w.strip(",./-")) >= 4
              and not re.fullmatch(r"(19|20)\d\d", w)]
    for v in values:
        if v in covered:
            continue
        for _, om, dw in scored:
            if v in dw:
                if om not in oms:
                    oms.append(om)
                covered |= dw
                break
    return oms

# 1. Define the Structured Output Schema
class CriticDecision(BaseModel):
    """Evaluation of whether the current task has been successfully answered."""
    is_resolved: bool = Field(
        description="True if the gathered context definitively answers the current task. False otherwise."
    )
    reasoning: str = Field(
        description="Explanation of why the context does or does not answer the task."
    )
    extracted_fact: str = Field(
        description="If is_resolved is True, provide the concise factual answer extracted from the context. If False, output 'Failed to resolve'."
    )
    out_of_scope: bool = Field(
        default=False,
        description=(
            "True if the context does not concern the entity asked about — either it is about a "
            "different institution/scheme/position/document, or the asked entity does not appear in "
            "the context at all (so the corpus does not cover it). When True, retrying retrieval "
            "cannot help, so the sub-task should short-circuit to a refusal rather than retry/rewrite."
        ),
    )


class CriticDecisionEntityCheck(CriticDecision):
    """CriticDecision + an explicit entity-disambiguation field.

    Used instead of CriticDecision when config.ENABLE_CRITIC_ENTITY_CHECK is
    True — see CRITIC_ENTITY_CHECK_ADDENDUM in critic_prompt.py for what
    this is guarding against (near-duplicate template documents, e.g.
    recurring recruitment ads that differ only in the named person/dept).
    """
    query_entities: list[str] = Field(
        default_factory=list,
        description=(
            "Specific identifying details implied by the sub-task (named "
            "person, department/school, advertisement/reference number, "
            "date) that must match exactly, not just topically."
        ),
    )
    context_entity_match: bool = Field(
        default=True,
        description=(
            "True only if the context's specific named entities match "
            "query_entities exactly — not just topical/document-type "
            "similarity. False if this is a same-template-different-"
            "instance mismatch (right kind of document, wrong specific "
            "person/department/number)."
        ),
    )


def get_critic_agent(fast_llm: BaseChatModel, reasoning_llm: BaseChatModel,
                      override_llm: BaseChatModel = None):
    """
    Factory function for the Critic (Reflexion) Agent.
    Evaluates retriever outputs and manages the plan progression.

    LLM cascade (config.ENABLE_CRITIC_CASCADE): fast_llm judges first; only
    sub-tasks it flags unresolved escalate to reasoning_llm for the
    authoritative verdict. See config.py for the trade-off this accepts.

    override_llm (config.ENABLE_COMPLEX_CRITIC_ROUTING): when provided, and
    the current query was flagged by the planner (state["use_stronger_critic"]),
    every LLM-judged pass on that query goes straight to override_llm instead
    of the fast_llm/reasoning_llm cascade — same rationale as
    ablation/stronger_critic_spike.py: a sub-task worth escalating to a
    stronger model shouldn't first pay for a weak-model opinion we're going
    to override anyway. The cheap gate (CRITIC_ACCEPT_SCORE) still applies
    first regardless — a strong-enough raw rerank score is accepted for free
    no matter which model would have judged it; override_llm only changes
    which model runs on the passes the gate does NOT accept.
    """
    entity_check = config.ENABLE_CRITIC_ENTITY_CHECK
    decision_schema = CriticDecisionEntityCheck if entity_check else CriticDecision
    prompt_text = CRITIC_SYSTEM_PROMPT_ENTITY_CHECK if entity_check else CRITIC_SYSTEM_PROMPT

    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text)
    ])

    fast_chain = critic_prompt | fast_llm.with_structured_output(decision_schema)
    reasoning_chain = critic_prompt | reasoning_llm.with_structured_output(decision_schema)
    override_chain = (critic_prompt | override_llm.with_structured_output(decision_schema)
                       if override_llm is not None else None)

    def critic_agent(state: MultiAgentRAGState):
        query = state["original_query"]
        current_task = state.get("current_task", "")
        context_docs = state.get("gathered_context", [])
        loop_count    = state.get("loop_count", 0)
        rewrite_count = state.get("rewrite_count", 0)
        top_score = state.get("top_rerank_score", float("-inf"))
        gate_fired = top_score >= config.CRITIC_ACCEPT_SCORE
        use_override = override_chain is not None and state.get("use_stronger_critic", False)

        # Telemetry appended to every return branch so each critic pass logs one
        # (score, gate_fired) pair for post-hoc CRITIC_ACCEPT_SCORE calibration.
        # escalated is filled in below once the cascade (or lack thereof) runs;
        # False here covers the gate-fired early-return path where no LLM ran.
        # override_used mirrors use_override but is forced False on the
        # gate-fired path below, since no model (override or otherwise) ran.
        telemetry = {"rerank_score_log": [top_score], "gate_fired_log": [gate_fired],
                     "critic_escalated_log": [False], "override_used_log": [False]}

        print(f"\n--- [Critic] Evaluating Task: '{current_task}' ---")

        # Use the reranker's curated output; fall back to raw docs if not available
        context_text = state.get("final_context_string") or "\n\n".join([d.page_content for d in context_docs])

        plan = state.get("plan", []).copy()

        # Capture agent attribution before wiping gathered_context
        agent_sources = list({
            doc.metadata.get("agent_source", "unknown")
            for doc in context_docs
            if hasattr(doc, "metadata") and doc.metadata
        })

        def _progress(fact: str):
            """Pop the resolved task, record its fact, and route onward.

            Resets BOTH loop_count and rewrite_count so the next sub-task
            starts with a fresh same-query-retry budget AND a fresh
            corrective-rewrite budget.
            """
            next_plan = plan.copy()
            if next_plan:
                next_plan.pop(0)
            kept = state.get("top_doc_sources") or []
            answer_oms = attribute_fact(fact, kept)
            source = {"task": current_task, "fact": fact,
                      "answer_om": answer_oms[0] if answer_oms else "",
                      "answer_oms": answer_oms,
                      "context_oms": list(dict.fromkeys(d["om"] for d in kept if d.get("om")))}
            return {
                "plan": next_plan,
                "past_steps": [(current_task, fact)],
                "subtask_sources": [source],
                "subtask_contexts": [context_text],
                "loop_count":    0,
                "rewrite_count": 0,
                "gathered_context": None,  # reset signal for the reducer
                "all_agent_sources": agent_sources,
                "next": ["planner" if next_plan else "generator"],
                **telemetry,
            }

        # ── Cheap gate: strong retrieval → accept without an LLM call ──────────
        if gate_fired:
            print(f"--- [Critic] Rerank score {top_score:.2f} ≥ {config.CRITIC_ACCEPT_SCORE}; "
                  f"accepting without LLM call. ---")
            return _progress(context_text)

        # ── Otherwise invoke the LLM judge ────────────────────────────────────
        chain_input = {
            "original_query": query,
            "current_task": current_task,
            "context": context_text,
        }

        if use_override:
            print("--- [Critic] Query flagged complex; routing directly to override model "
                  "(cascade skipped). ---")
            decision: CriticDecision = override_chain.invoke(chain_input)
            telemetry["override_used_log"] = [True]
        elif config.ENABLE_CRITIC_CASCADE:
            decision: CriticDecision = fast_chain.invoke(chain_input)
            print(f"--- [Critic] (fast) Resolved? {decision.is_resolved} ---")
            if not decision.is_resolved:
                print("--- [Critic] fast_llm flagged unresolved; escalating to reasoning_llm. ---")
                decision = reasoning_chain.invoke(chain_input)
                telemetry["critic_escalated_log"] = [True]
        else:
            decision: CriticDecision = reasoning_chain.invoke(chain_input)

        if entity_check and not getattr(decision, "context_entity_match", True):
            print(f"--- [Critic] Entity mismatch (query_entities={getattr(decision, 'query_entities', [])}); "
                  f"overriding is_resolved=True -> False. ---")
            decision.is_resolved = False

        print(f"--- [Critic] Resolved? {decision.is_resolved} ---")
        print(f"--- [Critic] Reasoning: {decision.reasoning} ---")

        if decision.is_resolved:
            return _progress(decision.extracted_fact)

        # ── Unresolved: 3-way branch ──────────────────────────────────────────
        # Tier 1: same-query retry while same-query budget remains.
        #
        # NOTE: out_of_scope is intentionally NOT short-circuited here. On the
        # first critic pass we typically hold a single retrieved chunk, and a
        # genuine in-corpus entity may simply be absent from THAT chunk while a
        # later retry surfaces the answer-bearing one. Refusing on attempt 1
        # caused false refusals on legitimate questions (Relational/Multi-part/
        # CrossHop regressions). So out_of_scope only earns a short-circuit AFTER
        # the same-query retry budget is spent — see below.
        if loop_count < config.MAX_CRITIC_RETRIES:
            print(f"--- [Critic] Task unresolved (attempt {loop_count + 1}/{config.MAX_CRITIC_RETRIES}). "
                  f"Retrying via Orchestrator. ---")
            return {
                "loop_count": loop_count + 1,
                "total_retries": 1,
                "gathered_context": None,
                "all_agent_sources": agent_sources,
                "next": ["orchestrator"],
                **telemetry,
            }

        # ── Out of scope (post-retry): the asked entity stayed absent from the
        # context across every same-query retry, so the corpus does not cover it
        # and a corrective rewrite would only re-fetch the same off-entity
        # context. Short-circuit to a clean refusal marker, skipping the rewrite
        # tier. Write to the structured out_of_scope_tasks list so the generator
        # doesn't need to parse string sentinels.
        if decision.out_of_scope:
            print("--- [Critic] Out of scope after retries (asked entity absent "
                  "from context). Short-circuiting to refusal. ---")
            next_plan = plan.copy()
            if next_plan:
                next_plan.pop(0)
            return {
                "plan":             next_plan,
                "past_steps":       [(current_task, "[OUT_OF_SCOPE]")],
                "subtask_contexts": [context_text],
                "out_of_scope_tasks": [current_task],
                "loop_count":       0,
                "rewrite_count":    0,
                "gathered_context": None,
                "all_agent_sources": agent_sources,
                "next": ["planner" if next_plan else "generator"],
                **telemetry,
            }

        # Tier 2: same-query budget spent → corrective rewrite if budget remains.
        # Pass the critic's reasoning to the rewriter so it has a precise
        # diagnosis rather than re-inferring from raw context alone.
        if rewrite_count < config.MAX_REWRITES:
            print(f"--- [Critic] Same-query retries exhausted "
                  f"(rewrite {rewrite_count + 1}/{config.MAX_REWRITES}). "
                  f"Routing to Rewriter. ---")
            return {
                "total_retries":   1,
                "gathered_context": None,
                "all_agent_sources": agent_sources,
                "critic_reasoning": decision.reasoning,
                "next": ["rewriter"],
                **telemetry,
            }

        # Tier 3: both budgets spent → force plan progression with a structured
        # sentinel. Generator reads [RETRIEVAL_FAILED] to distinguish from
        # genuine out-of-scope vs hard-to-retrieve content.
        print("--- [Critic] Same-query and rewrite budgets both exhausted. "
              "Forcing plan progression. ---")
        return _progress("[RETRIEVAL_FAILED]")

    return critic_agent