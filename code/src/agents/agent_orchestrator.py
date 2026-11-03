from src.core.state import MultiAgentRAGState
import config

# Keywords derived from KG_ALLOWED_RELATIONS semantics (the real Neo4j
# relationship types — see config.py's note on this list's history). Only
# used when config.ENABLE_UNCONDITIONAL_KG is False. If ANY appear in the
# sub-task (case-insensitive), the KG agent is activated; otherwise it is
# skipped to avoid a wasted Cypher-translation LLM call + Neo4j round-trip.
# Vector and BM25 always run regardless.
#
# The trailing group (supersede/applies to/budget head/reports to/
# compliance) doesn't match a real relationship type 1:1, but each is
# plausibly answerable via property filters or an indirect relation chain
# (e.g. "reports to" via APPOINTED_AS + ASSOCIATED_WITH), so they're kept —
# a false-positive KG attempt only costs one wasted Cypher call; dropping a
# keyword that's still occasionally answerable would silently lose recall.
_KG_KEYWORDS = {
    "issued by", "who issued", "issuing authority",                    # ISSUED_BY
    "directive", "decision",                                            # CONTAINS_DIRECTIVE
    "affects", "affected by", "impact on",                              # AFFECTS
    "appointed", "appointment", "appointed as",                         # APPOINTED_AS
    "associated with",                                                  # ASSOCIATED_WITH
    "transfer", "transferred", "transferred from", "transferred to",    # TRANSFERRED_FROM/TO
    "modifies", "clarifies", "amendment", "amends",                     # MODIFIES_OR_CLARIFIES
    "committee member", "committee", "member of",                       # MEMBER_OF
    "position", "vacancy", "fellowship", "recruitment", "opening",      # ANNOUNCES_POSITION
    "funds", "funded by", "funding", "sponsor",                         # FUNDS_PROJECT
    "leads", "project lead", "principal investigator",                  # LEADS_PROJECT
    "supersede", "supersedes", "superseded", "replaces", "replaced by",
    "applies to", "applicable to", "which institute",
    "budget head",
    "reports to", "reporting to", "reporting structure",
    "compliance", "mandates", "mandated by",
}


def get_orchestrator_agent():
    """
    Routing node — activates retrievers for the current sub-task.

    config.ENABLE_UNCONDITIONAL_KG controls the tradeoff:
      True  — vector, BM25, and KG all run on every sub-task. Never silently
              misses a KG-answerable sub-task, at the cost of an extra
              Cypher-translation LLM call + Neo4j round-trip on sub-tasks
              that don't need it.
      False — KG only runs if _KG_KEYWORDS matches the sub-task text.
              Cheaper, but only as complete as that keyword list.
    """
    def orchestrator_agent(state: MultiAgentRAGState):
        current_task = state.get("current_task") or state.get("original_query")

        if config.ENABLE_UNCONDITIONAL_KG:
            active = ["vector", "bm25", "kg"]
        else:
            task_lower = current_task.lower()
            use_kg = any(kw in task_lower for kw in _KG_KEYWORDS)
            active = ["vector", "bm25"] + (["kg"] if use_kg else [])

        print(f"\n--- [Orchestrator] Dispatching {active} for: '{current_task}' ---")
        return {"active_retrievers": active}

    return orchestrator_agent
