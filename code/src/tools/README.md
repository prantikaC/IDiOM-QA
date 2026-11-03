# 🔧 Tools Directory

## Purpose
This directory contains the specific utilities, wrappers, and clients that allow our LangGraph agents to interact with the outside world. If an agent needs to access a local database, run an embedding model, or drive a live web browser, the underlying execution code lives here.

## Key Files

### 1. `local_tools.py` (The Database Connectors)
Contains the wrapper functions that securely initialize and connect LangChain to our Hybrid Retrieval databases.
* **`get_local_vector_store()`:** Loads the FAISS index from disk. It automatically initializes the `BAAI/bge-small-en-v1.5` HuggingFace embedding model on the GPU (CUDA). 
* **`get_local_bm25_retriever()`:** Reads the `bm25_data.json` file and reconstructs the lexical index, ensuring the Time-Aware metadata is properly attached to each document.
* **`get_kg_chain()`:** Connects to Neo4j. Crucially, it runs `graph.refresh_schema()` on boot so the LLM always has the most up-to-date map of the institutional entities before writing Cypher queries.
* **`clean_and_chunk_text()`:** A utility function for splitting raw strings into manageable `Document` chunks.

### 2. `mcp_client.py` (The Browser Subprocess)
The connector for the Model Context Protocol (MCP).
* **`get_mcp_browser_tools()`:** Instead of relying on static APIs, this async function uses `npx` to spawn a live, headless Puppeteer browser running in a Node.js subprocess. It translates the browser's capabilities (navigate, read, click) into standard LangChain tools that the `agent_mcp_web.py` can use dynamically.

## For Future Developers 🛠️

**1. The Fallback Shields (Error Handling):**
Look at the `try/except` blocks inside `local_tools.py`. If a database fails to load (e.g., a missing path or expired Neo4j credentials), these tools intentionally catch the error, print a warning, and return a "dummy" or empty initialization. This ensures that a single broken database doesn't crash the entire LangGraph Orchestrator. 

**2. FAISS Deserialization Security:**
To load the local FAISS `.pkl` files, the `allow_dangerous_deserialization=True` flag is mandatory in LangChain. Do not remove this, or the Vector Agent will fail to boot.

**3. Node.js Dependency for MCP:**
Because `mcp_client.py` uses `MultiServerMCPClient` to execute the `@modelcontextprotocol/server-puppeteer` package via `npx`, **the host machine must have Node.js installed.** If the MCP Web Agent is failing immediately, verify that `npm` and `npx` are in your system's PATH.