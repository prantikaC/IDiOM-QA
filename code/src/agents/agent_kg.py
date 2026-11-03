from langchain_neo4j import GraphCypherQAChain
from langchain_core.documents import Document
from src.core.state import MultiAgentRAGState

def _row_text(row: dict) -> str:
    """One graph row as a readable fact ("_" in node IDs read as spaces)."""
    clean = lambda v: str(v).replace("_", " ")
    if {"subject", "relation", "object"} <= row.keys():
        text = f"{clean(row['subject'])} -[{row['relation']}]-> {clean(row['object'])}"
        if row.get("neighbour_description"):
            text += f" ({row['neighbour_description']})"
        if row.get("source_om"):
            text += f" [source OM: {row['source_om']}]"
        return text
    return "; ".join(f"{k}: {clean(v)}" for k, v in row.items() if v not in (None, ""))


def get_kg_agent(cypher_chain: GraphCypherQAChain):
    """
    Factory function that returns the Knowledge Graph Retrieval Agent.
    """
    def kg_retrieval_agent(state: MultiAgentRAGState):
        # Skip if the orchestrator's keyword classifier determined KG is not
        # useful for this sub-task (saves Cypher LLM call + Neo4j round-trip).
        active = state.get("active_retrievers", ["vector", "bm25", "kg"])
        if "kg" not in active:
            print("\n--- [KG Agent] Skipped (no relational keywords in sub-task) ---")
            return {"gathered_context": [], "cypher_queries": []}

        current_task = state.get("current_task") or state.get("original_query")
        print(f"\n--- [KG Agent] Translating and querying Graph DB for: '{current_task}' ---")
        
        docs = []
        cypher_log = []
        try:
            # Run the Text-to-Cypher QA Chain on the specific task
            response = cypher_chain.invoke({"query": current_task})

            # Extract the generated Cypher from intermediate_steps for audit logging
            for step in response.get("intermediate_steps", []):
                if isinstance(step, dict) and "query" in step:
                    cypher_log.append(step["query"])

            result = response.get("result", "")
            if isinstance(result, list):
                # return_direct: raw rows, one fact per Document, so the
                # cross-encoder ranks each fact against the text chunks and the
                # reranker's recency weight sees the fact's own issue_date.
                for row in result:
                    docs.append(Document(page_content=_row_text(row), metadata={
                        "agent_source": "kg_retriever", "type": "structured_data",
                        "issue_date": row.get("issue_date") or "UNKNOWN",
                        "source_om": row.get("source_om") or ""}))
            elif result:
                # Chain built without return_direct: a summarising LLM's answer.
                docs.append(Document(
                    page_content=result,
                    metadata={"agent_source": "kg_retriever", "type": "structured_data"}
                ))

        except Exception as e:
            print(f"--- [KG Agent Error]: Failed to execute Cypher query: {e} ---")

        # Runs in parallel with the other retrievers; the state reducer merges
        # each retriever's docs, so return only our own.
        return {"gathered_context": docs, "cypher_queries": cypher_log}
        
    return kg_retrieval_agent