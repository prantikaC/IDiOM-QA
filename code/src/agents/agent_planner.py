from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from src.core.state import MultiAgentRAGState
from src.prompts.planner_prompt import (
    PLANNER_SYSTEM_PROMPT, COMPLEXITY_CLASSIFIER_PROMPT, REPLAN_SYSTEM_PROMPT
)
import config

class Plan(BaseModel):
    """The step-by-step breakdown of the complex query."""
    steps: List[str] = Field(
        description="A sequential list of specific, actionable sub-tasks. Each step should be solvable by a single database query or search."
    )

class ComplexityDecision(BaseModel):
    """Whether the query needs multi-part decomposition or a single direct lookup,
    and separately, whether judging its retrieved context is hard enough to need
    a stronger critic/rewriter model (see COMPLEXITY_CLASSIFIER_PROMPT)."""
    is_multi_hop: bool = Field(
        description="True if answering requires multiple distinct retrievals or a dependency chain. False for a single direct lookup."
    )
    needs_stronger_critic: bool = Field(
        default=False,
        description="True if the query is Aggregation-shaped (needs a completeness "
                    "judgment across multiple entities), Temporal-shaped (needs a "
                    "date-range-coverage judgment), or Cross-document-shaped (needs "
                    "correct entity attribution across near-duplicate template "
                    "documents) — the three patterns where a weaker critic has been "
                    "observed accepting incomplete or wrong context. False otherwise."
    )

class PlanRevision(BaseModel):
    """Adaptive revision of the REMAINING plan, given evidence found so far."""
    steps: List[str] = Field(
        min_length=1,
        description="The revised remaining plan — may be unchanged, split, merged, or "
                    "reordered relative to the input remaining plan. Must be non-empty."
    )
    reasoning: str = Field(
        description="One short sentence explaining what changed and why, or 'no change needed'."
    )

def get_planner_agent(reasoning_llm: BaseChatModel):
    """
    Factory function for the Planner Agent.

    On a fresh query it first runs a complexity classifier (reasoning_llm):
      - single-hop  → skip decomposition, plan = [original_query] (fast path)
      - multi-part   → decompose with the reasoning LLM
    The fast path is gated by config.ENABLE_SINGLE_HOP_FASTPATH.
    """
    structured_llm = reasoning_llm.with_structured_output(Plan)

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", "Break down this query: {query}")
    ])
    planner_chain = planner_prompt | structured_llm

    # Classifier promoted to reasoning_llm (2026-08-18): empirically verified
    # fast_llm (7B) called MULTI-HOP on every one of 6 test questions,
    # including clearly single-hop ones -- not a borderline miscalibration,
    # it wasn't following "default to single-hop" at all. reasoning_llm (32B)
    # got all 6 right on the identical prompt, confirming a model capability
    # gap rather than a prompt problem. This was costing ~89% of questions an
    # unneeded decompose call plus every extra sub-task's full retrieve→
    # rerank→critic cycle -- far more than the classify call itself costs.
    classifier_prompt = ChatPromptTemplate.from_template(COMPLEXITY_CLASSIFIER_PROMPT)
    classifier_chain = classifier_prompt | reasoning_llm.with_structured_output(ComplexityDecision)

    # Adaptive re-planning: re-examines the remaining plan against actual
    # evidence each time control returns from the critic. Runs on
    # reasoning_llm — same tier as the initial decomposition, since revising
    # a plan given evidence is the same kind of judgment call.
    replan_structured_llm = reasoning_llm.with_structured_output(PlanRevision)
    replan_prompt = ChatPromptTemplate.from_messages([
        ("system", REPLAN_SYSTEM_PROMPT)
    ])
    replan_chain = replan_prompt | replan_structured_llm

    def planner_agent(state: MultiAgentRAGState):
        query = state["original_query"]
        plan = state.get("plan", [])

        # Scenario 1: First time running — classify, then plan (or fast-path)
        if not plan:
            # Classifier now always runs on the first pass (previously gated
            # behind ENABLE_SINGLE_HOP_FASTPATH) because it also drives
            # use_stronger_critic, which is independent of the fast-path
            # decision. When ENABLE_SINGLE_HOP_FASTPATH is False this adds one
            # extra classify call per query vs the old behavior — acceptable,
            # since that flag is only ever disabled for ablations, not
            # production (see config.py).
            verdict: ComplexityDecision = classifier_chain.invoke({"query": query})
            use_stronger_critic = bool(config.ENABLE_COMPLEX_CRITIC_ROUTING and verdict.needs_stronger_critic)
            if use_stronger_critic:
                print(f"--- [Planner] Query flagged COMPLEX (needs_stronger_critic) → "
                      f"critic/rewriter will route to CRITIC_REWRITER_LLM_OVERRIDE. ---")

            if config.ENABLE_SINGLE_HOP_FASTPATH and not verdict.is_multi_hop:
                print(f"\n--- [Planner] Single-hop query → fast path (1 step, no decomposition). ---")
                return {"plan": [query], "current_task": query, "use_stronger_critic": use_stronger_critic}

            if config.ENABLE_SINGLE_HOP_FASTPATH:
                print(f"\n--- [Planner] Multi-part query → decomposing. ---")

            plan_decision: Plan = planner_chain.invoke({"query": query})
            steps = plan_decision.steps[:config.MAX_PLAN_STEPS]  # cap over-decomposition
            if len(plan_decision.steps) > config.MAX_PLAN_STEPS:
                print(f"--- [Planner] Generated {len(plan_decision.steps)} steps, "
                      f"capped to {config.MAX_PLAN_STEPS}. ---")
            else:
                print(f"--- [Planner] Generated {len(steps)} steps ---")

            # Save the plan AND set the very first task to active
            return {
                "plan": steps,
                "current_task": steps[0] if steps else query,
                "use_stronger_critic": use_stronger_critic,
            }

        # Scenario 2: Looping back from the Critic. Plan already exists —
        # `plan` here is the REMAINING plan (the critic already popped the
        # just-resolved step before routing back here).
        else:
            if not config.ENABLE_ADAPTIVE_REPLANNING:
                print("--- [Planner] Plan already exists, advancing to next task. ---")
                return {"current_task": plan[0]}

            past_steps = state.get("past_steps", [])
            resolved = [(t, r) for t, r in past_steps
                        if r not in ("[OUT_OF_SCOPE]", "[RETRIEVAL_FAILED]")]
            ledger_text = (
                "\n".join(f"- {t}\n    → {r}" for t, r in resolved)
                if resolved else "(none yet)"
            )
            last_context = state.get("final_context_string") or "(not available)"
            remaining_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
            budget = max(config.MAX_TOTAL_PLAN_STEPS - len(resolved), 1)

            print("--- [Planner] Re-examining remaining plan against evidence found so far. ---")
            try:
                revision: PlanRevision = replan_chain.invoke({
                    "original_query": query,
                    "past_steps":     ledger_text,
                    "last_context":   last_context,
                    "remaining_plan": remaining_text,
                    "budget":         budget,
                })
                revised_steps = (revision.steps or plan)[:budget]
                if revised_steps != plan:
                    print(f"--- [Planner] Plan revised: {revision.reasoning} ---")
                    print(f"--- [Planner] New remaining plan ({len(revised_steps)} steps): {revised_steps} ---")
                else:
                    print("--- [Planner] Plan unchanged. ---")
            except Exception as e:
                # Fail open: keep the original remaining plan rather than stall the run.
                print(f"--- [Planner] Re-plan failed ({e}); keeping original plan. ---")
                revised_steps = plan

            return {
                "plan":         revised_steps,
                "current_task": revised_steps[0],
            }

    return planner_agent