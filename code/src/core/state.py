import operator
from typing import TypedDict, List, Annotated, Tuple, Optional
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


def merge_gathered_context(
    existing: List[Document], update: Optional[List[Document]]
) -> List[Document]:
    """Reducer for gathered_context.

    The three retrievers now run in parallel and each return ONLY their own
    docs in the same superstep, so a reducer is required to merge concurrent
    writes. `None` is the explicit reset signal (sent by the Critic between
    sub-tasks and before a retry) — it wipes the blackboard clean.
    """
    if update is None:
        return []
    return (existing or []) + update


class MultiAgentRAGState(TypedDict):
    # Standard Fields
    original_query: str
    messages: Annotated[List[BaseMessage], operator.add]
    next: List[str]

    # Retrievers fan out in parallel and append via the reducer; the Critic
    # resets it (returns None) between sub-tasks so each pass sees fresh data.
    gathered_context: Annotated[List[Document], merge_gathered_context]
    final_context_string: str  # To store the post-reranked string
    top_rerank_score: float    # Top cross-encoder score from the last rerank pass
    
    final_answer: str
    loop_count: int # Removed operator.add so we can explicitly reset it to 0
    web_search_needed: bool
    is_web_searchable: bool

    # Sequential Reasoning Fields
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add] # Keeps a running ledger
    current_task: str

    # The reranked context each sub-task ended on, one entry per sub-task
    # (written by the critic alongside past_steps). final_context_string only
    # holds the last sub-task's, so evaluation scores the answer against these.
    subtask_contexts: Annotated[List[str], operator.add]

    # KG audit trail — accumulates one entry per sub-task that hits the KG agent
    cypher_queries: Annotated[List[str], operator.add]

    # Retry counter — sums across all sub-task retry passes
    total_retries: Annotated[int, operator.add]

    # Agent attribution — accumulates across all retrieval passes (gathered_context is wiped per sub-task)
    all_agent_sources: Annotated[List[str], operator.add]

    # Corrective query-rewriting — fires when same-query retries are exhausted.
    # rewrite_count is reset to 0 between sub-tasks (alongside loop_count) and
    # capped by config.MAX_REWRITES. task_variants accumulates every rewritten
    # form tried across the whole query so the eval log preserves the rewrite
    # audit trail.
    rewrite_count: int
    task_variants: Annotated[List[str], operator.add]

    # Critic → Rewriter handoff: the critic's LLM reasoning for why retrieval
    # failed, written when routing to the rewriter. Simple str (overwritten each
    # critic pass) so the rewriter always sees the most recent diagnosis.
    critic_reasoning: str

    # Structured refusal ledger: sub-tasks for which the critic confirmed the
    # corpus has no coverage (out_of_scope=True after retries). Accumulates
    # across sub-tasks. Generator reads this directly instead of parsing string
    # sentinels in past_steps.
    out_of_scope_tasks: Annotated[List[str], operator.add]

    # Orchestrator routing decision: which retrieval agents to activate for the
    # current sub-task. Set by the orchestrator's keyword classifier before each
    # fan-out. Vector and BM25 always run; KG is skipped for non-relational
    # sub-tasks to save latency.
    active_retrievers: List[str]

    # Critic fast-gate telemetry — one entry appended per critic pass, used for
    # post-hoc CRITIC_ACCEPT_SCORE re-calibration (the threshold was tuned on
    # original_query scoring; the reranker now scores current_task). Every run
    # therefore doubles as calibration data without a dedicated instrumented run.
    #   rerank_score_log      — the top_rerank_score the gate compared this pass.
    #   gate_fired_log        — True if the cheap gate accepted without an LLM
    #                           call, False if a critic LLM call ran.
    #   critic_escalated_log  — True if fast_llm flagged unresolved and the
    #                           call escalated to reasoning_llm (see
    #                           config.ENABLE_CRITIC_CASCADE). Always False
    #                           when the cascade is off or the gate fired.
    #   override_used_log     — True if this pass's critic judgment actually
    #                           ran on CRITIC_REWRITER_LLM_OVERRIDE (either
    #                           global spike mode or complex-routing mode),
    #                           False if it used the normal reasoning_llm or
    #                           was gate-skipped entirely. Added 2026-09-04
    #                           after manually cross-referencing gate_fired_log
    #                           against config to answer "was the override
    #                           always fired" — this makes that answerable
    #                           directly from a single run's output.
    rerank_score_log: Annotated[List[float], operator.add]
    gate_fired_log: Annotated[List[bool], operator.add]
    critic_escalated_log: Annotated[List[bool], operator.add]
    override_used_log: Annotated[List[bool], operator.add]

    # Planner's complexity classifier (see COMPLEXITY_CLASSIFIER_PROMPT):
    # True if the query was flagged as needing the stronger critic/rewriter
    # model (Aggregation/Temporal/CrossHop-shaped judgment calls). Set once
    # per query at first planner pass, read by critic/rewriter on every
    # sub-task of that query. Only consulted when
    # config.ENABLE_COMPLEX_CRITIC_ROUTING is True.
    use_stronger_critic: bool