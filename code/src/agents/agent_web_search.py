from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from src.core.state import MultiAgentRAGState

def get_web_search_agent():
    """
    Factory function for the Web Search Agent.
    Triggered only when local retrieval fails to find highly relevant context.
    """
    # Initialize the search tool
    web_search_tool = DuckDuckGoSearchRun()
    
    def web_search_agent(state: MultiAgentRAGState):
        current_task = state.get("current_task") or state.get("original_query")
        
        print(f"\n--- [Web Search Agent] Local context insufficient. Searching web for: '{current_task}' ---")
        
        docs = []
        try:
            # Execute the web search using the atomic task
            search_result = web_search_tool.invoke(current_task)
            
            # Wrap the result in a Document
            doc = Document(
                page_content=search_result,
                metadata={"agent_source": "web_search"} 
            )
            docs.append(doc)
            print("--- [Web Search Agent] Successfully retrieved web context. ---")
            
        except Exception as e:
            print(f"--- [Web Search Error] Search failed: {e} ---")
            
        # Overwrite the gathered_context so the Critic can evaluate the web findings.
        # We also reset web_search_needed so it doesn't get stuck in a loop.
        return {
            "gathered_context": docs, 
            "web_search_needed": False
        }
        
    return web_search_agent