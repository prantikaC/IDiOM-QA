import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_mcp_browser_tools():
    """
    Connects to the official Puppeteer MCP Server via stdio and 
    loads the browser automation tools (navigate, read, click) as LangChain tools.
    """
    print("--- [MCP Client] Booting Headless Browser Server ---")
    
    try:
        # We use MultiServerMCPClient to spawn the Node subprocess securely
        client = MultiServerMCPClient({
            "puppeteer": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
            }
        })
        
        # This reaches out to the server and converts MCP tools to LangChain tools
        tools = await client.get_tools()
        
        print(f"--- [MCP Client] Successfully loaded {len(tools)} live browser tools ---")
        return tools, client
    
    except Exception as e:
        print(f"--- [MCP Error] Failed to connect to browser server: {e} ---")
        return [], None