from langgraph.graph import StateGraph, START, END

from src.core.state import MultiAgentRAGState
from src.core.llm_factory import get_llms, get_llm_by_spec
from src.tools.local_tools import get_local_vector_store, get_local_bm25_retriever, get_kg_chain

from src.agents.agent_planner import get_planner_agent
from src.agents.agent_orchestrator import get_orchestrator_agent
from src.agents.agent_vector import get_vector_agent
from src.agents.agent_bm25 import get_bm25_agent
from src.agents.agent_kg import get_kg_agent
from src.agents.agent_reranker import get_reranker_agent
from src.agents.agent_critic import get_critic_agent
from src.agents.agent_generator import get_generator_agent
from src.agents.agent_rewriter import get_rewriter_agent

import config

def build_sequential_graph(backend: str = "openai", neo4j_database: str | None = None,
                            kg_cypher_prompt=None):
    """
    Constructs the Plan-and-Execute sequential StateGraph.

    Args:
        backend: "openai" | "ollama" | "qwen"
        neo4j_database: forwarded to get_kg_chain(). None (default) keeps the
            existing production behavior (config.NEO4J_DATABASE, "neo4j").
            Pass a distinct database name to point the KG agent at an
            isolated graph instead -- e.g. a benchmark pilot's own KG, so it
            can never read/write the production OM graph.
        kg_cypher_prompt: forwarded to get_kg_chain(); overrides the
            OM-specific CYPHER_GENERATION_PROMPT for a different domain.

    LLM allocation:
      fast_llm      — generator,
                      critic's first-pass judgment (see cascade note below)
      reasoning_llm — planner (classifier + decomposition), rewriter, KG Cypher
                      chain (moved from fast_llm 2026-10-28: 7B wrote invalid or
                      empty Cypher on all 20 Relational probe questions),
                      critic's escalated (authoritative) judgment

    Complexity classifier promoted to reasoning_llm (2026-08-18): fast_llm
    (7B) empirically called every test question MULTI-HOP regardless of
    actual need — not a borderline miscalibration, a model capability gap.
    See agent_planner.py for the verification.

    Critic cascade (config.ENABLE_CRITIC_CASCADE): fast_llm judges each
    sub-task first; only ones it flags unresolved escalate to reasoning_llm.
    Rewriter is unconditionally on reasoning_llm — it's already only invoked
    on failure paths (same-query retries exhausted), so there's no cheap
    case to cascade away.
    """
    print("--- [Graph Builder] Initializing Sequential Reasoning Engine ---")

    fast_llm, reasoning_llm = get_llms(backend)

    # Critic/rewriter model override (config.CRITIC_REWRITER_LLM_OVERRIDE),
    # leaving the planner (and fast_llm) on their normal assignments either
    # way. Two modes, selected by config.ENABLE_COMPLEX_CRITIC_ROUTING:
    #
    #   False (default) — legacy/spike behavior (see
    #   ablation/stronger_critic_spike.py): the override applies globally,
    #   to EVERY sub-task, by baking it directly into the "reasoning_llm"
    #   slot the critic/rewriter are built with.
    #
    #   True — full feature: the override is passed as a SEPARATE
    #   override_llm, used only for sub-tasks of queries the planner's
    #   complexity classifier flagged (state["use_stronger_critic"]); every
    #   other sub-task keeps using the normal reasoning_llm/cascade.
    override_llm = None
    if config.CRITIC_REWRITER_LLM_OVERRIDE:
        override_llm = get_llm_by_spec(config.CRITIC_REWRITER_LLM_OVERRIDE)

    if config.ENABLE_COMPLEX_CRITIC_ROUTING:
        critic_reasoning_llm = reasoning_llm
        critic_override_llm = override_llm
        if override_llm is not None:
            print(f"--- [Graph Builder] ENABLE_COMPLEX_CRITIC_ROUTING active: complex-flagged "
                  f"sub-tasks -> {config.CRITIC_REWRITER_LLM_OVERRIDE}, others stay on "
                  f"reasoning_llm (planner unaffected either way) ---")
    else:
        critic_reasoning_llm = override_llm if override_llm is not None else reasoning_llm
        critic_override_llm = None
        if override_llm is not None:
            print(f"--- [Graph Builder] CRITIC_REWRITER_LLM_OVERRIDE active (global, legacy mode): "
                  f"critic + rewriter -> {config.CRITIC_REWRITER_LLM_OVERRIDE} "
                  f"(planner stays on reasoning_llm) ---")

    # Initialize databases
    vectorstore = get_local_vector_store()
    bm25_retriever = get_local_bm25_retriever()
    cypher_chain = get_kg_chain(reasoning_llm, database=neo4j_database, cypher_prompt=kg_cypher_prompt)
    if cypher_chain is not None:
        # Hand the KG agent the raw rows (not a summarising LLM's answer, which
        # mostly said "the information does not specify" even when the rows
        # held the fact), and keep the generated Cypher for the audit log.
        cypher_chain.return_direct = True
        cypher_chain.return_intermediate_steps = True
        cypher_chain.top_k = 25

    # Instantiate agents
    planner_agent = get_planner_agent(reasoning_llm=reasoning_llm)
    orchestrator_agent = get_orchestrator_agent()
    vector_agent = get_vector_agent(
        vectorstore=vectorstore,
        top_k=config.TOP_K_RETRIEVE,
        overfetch=config.RETRIEVE_OVERFETCH,
        recency_lambda=config.RECENCY_LAMBDA_RETRIEVER,
    )
    bm25_agent = get_bm25_agent(
        bm25_retriever=bm25_retriever,
        top_k=config.TOP_K_RETRIEVE,
        overfetch=config.RETRIEVE_OVERFETCH,
        recency_lambda=config.RECENCY_LAMBDA_RETRIEVER,
    )
    kg_agent = get_kg_agent(cypher_chain=cypher_chain)
    critic_agent = get_critic_agent(fast_llm=fast_llm, reasoning_llm=critic_reasoning_llm,
                                     override_llm=critic_override_llm)
    generator_agent = get_generator_agent(llm=fast_llm)
    rewriter_agent = get_rewriter_agent(llm=critic_reasoning_llm, override_llm=critic_override_llm)
    reranker_agent = get_reranker_agent(
        model_name=config.RERANKER_MODEL_NAME,
        top_k=config.TOP_K_RERANK,
        recency_lambda=config.RECENCY_LAMBDA_RERANKER,
    )

    # ==========================================
    # Custom Graph Routing Functions
    # ==========================================
    
    def route_from_planner(state: MultiAgentRAGState):
        """
        Decides if we have tasks left to execute, or if we are ready to generate.
        """
        plan = state.get("plan", [])
        if not plan:
            print("--- [Router] Plan is empty. Routing to Generator. ---")
            return "generator"
            
        print(f"--- [Router] Tasks remaining: {len(plan)}. Routing to Orchestrator. ---")
        return "orchestrator"

    def route_from_critic(state: MultiAgentRAGState):
        next_nodes = state.get("next", ["planner"])
        return next_nodes[0]

    # ==========================================
    # Build the Workflow
    # ==========================================
    workflow = StateGraph(MultiAgentRAGState)

    # Add Nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("vector_agent", vector_agent)
    workflow.add_node("bm25_agent", bm25_agent)
    workflow.add_node("kg_agent", kg_agent)
    workflow.add_node("reranker", reranker_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("generator", generator_agent)
    workflow.add_node("rewriter", rewriter_agent)

    # Add Edges
    workflow.add_edge(START, "planner")

    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "orchestrator": "orchestrator",
            "generator": "generator"
        }
    )

    # Fan-out: run all 3 retrievers in PARALLEL (same superstep), then fan back
    # into the reranker, which runs once after all three complete. The state
    # reducer on gathered_context merges their concurrent writes.
    workflow.add_edge("orchestrator", "vector_agent")
    workflow.add_edge("orchestrator", "bm25_agent")
    workflow.add_edge("orchestrator", "kg_agent")
    workflow.add_edge("vector_agent", "reranker")
    workflow.add_edge("bm25_agent", "reranker")
    workflow.add_edge("kg_agent", "reranker")
    workflow.add_edge("reranker", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "planner":      "planner",
            "generator":    "generator",
            "orchestrator": "orchestrator",
            "rewriter":     "rewriter",
        }
    )

    # Corrective rewrite hop: rewriter rephrases the failed sub-task and hands
    # control back to the orchestrator, which fans out to all three retrievers
    # with the new current_task and a fresh loop_count.
    workflow.add_edge("rewriter", "orchestrator")

    workflow.add_edge("generator", END)

    app = workflow.compile()
    print("--- [Graph Builder] Sequential Graph compilation complete ---")
    
    return app