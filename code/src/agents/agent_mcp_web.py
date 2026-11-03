import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent  # <-- NEW v1.0 IMPORT
from langchain_core.documents import Document

from src.core.state import MultiAgentRAGState
from src.tools.mcp_client import get_mcp_browser_tools

def get_mcp_web_agent(llm: ChatOpenAI):
    """
    Factory function for the Live Browser Agent.
    """
    def mcp_web_agent(state: MultiAgentRAGState):
        current_task = state.get("current_task") or state.get("original_query")
        print(f"\n--- [MCP Web Agent] Driving live browser for task: '{current_task}' ---")
        
        # 1. Fetch the tools dynamically using asyncio
        loop = asyncio.get_event_loop()
        tools, mcp_client = loop.run_until_complete(get_mcp_browser_tools())
        
        if not tools:
            print("--- [MCP Web Agent] Browser unavailable. Aborting web search. ---")
            return {"gathered_context": state.get("gathered_context", [])}

        # 2. Spin up a sub-agent specifically designed to use tools
        sys_prompt = (
            "You are a live web research agent. "
            "You have access to a Puppeteer browser. "
            "To answer the user's task, navigate to the required URL, read the page content, "
            "and extract ONLY the exact facts requested. Be highly concise."
        )
        
        # 3. Initialize with the new LangChain v1.0 API
        browser_agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)
        
        # 4. Execute the browser agent
        print("--- [MCP Web Agent] Executing browser commands... ---")
        result = browser_agent.invoke({"messages": [("user", current_task)]})
        
        # Extract the final answer from the sub-agent
        final_answer = result["messages"][-1].content
        
        # 5. Package it into a Document for the Reranker to grade
        live_doc = Document(
            page_content=final_answer,
            metadata={"agent_source": "mcp_live_browser", "url": "dynamic_fetch"}
        )
        
        print("--- [MCP Web Agent] Live extraction complete. Adding to blackboard. ---")
        
        # Merge the new live data with whatever the local databases already found
        existing_context = state.get("gathered_context", [])
        existing_context.append(live_doc)
        
        return {"gathered_context": existing_context}
        
    return mcp_web_agent