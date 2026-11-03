import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel

import config
from src.prompts.kg_prompt import CYPHER_GENERATION_PROMPT
from src.core.device import get_device

def get_local_vector_store():
    """Initializes the local FAISS Vector Store."""
    model_kwargs = {'device': get_device()}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    try:
        # We drop the os.path.exists check and let FAISS natively load it
        return FAISS.load_local(
            folder_path=config.VECTOR_STORE_PATH, 
            embeddings=embeddings,
            allow_dangerous_deserialization=True # Required for FAISS .pkl security
        )
    except Exception as e:
        print(f"--- [Local Tools Error] Failed to load FAISS from {config.VECTOR_STORE_PATH}: {e} ---")
        print("--- [Local Tools] Creating empty fallback Vector Store ---")
        return FAISS.from_texts(["Initialization document."], embedding=embeddings)

def get_local_bm25_retriever():
    """Initializes the in-memory BM25 Retriever."""
    # Resolve the exact path to the new bm25_data.json
    local_db_path = os.path.dirname(config.BM25_CORPUS_PATH) if hasattr(config, 'BM25_CORPUS_PATH') else os.path.join("data", "local_db")
    data_path = os.path.join(local_db_path, "bm25_data.json")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            bm25_data = json.load(f)
            
        # Parse the new Time-Aware Dictionary format
        docs = []
        corpus = bm25_data.get("corpus", [])
        metadata = bm25_data.get("metadata", [])
        
        for i in range(len(corpus)):
            docs.append(Document(
                page_content=corpus[i], 
                metadata=metadata[i] if i < len(metadata) else {}
            ))
            
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = config.TOP_K_RETRIEVE
        return retriever
        
    except Exception as e:
        print(f"--- [Local Tools Error] Failed to load BM25 from {data_path}: {e} ---")
        print("--- [Local Tools] Creating empty fallback BM25 Retriever ---")
        retriever = BM25Retriever.from_texts(["Initialization document."])
        retriever.k = config.TOP_K_RETRIEVE
        return retriever

def clean_and_chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[Document]:
    """Utility for chunking raw text before inserting into local tools."""
    cleaned = text.replace("\n\n", " ").replace("\t", " ").strip()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [Document(page_content=chunk) for chunk in splitter.split_text(cleaned)]

def get_kg_chain(llm: BaseChatModel, database: str | None = None,
                  cypher_prompt: PromptTemplate | None = None) -> GraphCypherQAChain:
    """
    Initializes the Neo4j Graph connection and the Text-to-Cypher QA Chain.
    Requires Neo4j credentials in your config.py.

    Args:
        database: Neo4j database name to connect to. Defaults to
            config.NEO4J_DATABASE ("neo4j", the production OM graph) when not
            given -- existing call sites are unaffected. Pass a distinct name
            (e.g. "wiki2pilot") to isolate a benchmark pilot's KG from the OM
            production graph on the same Neo4j server.
        cypher_prompt: override for the OM-specific CYPHER_GENERATION_PROMPT,
            e.g. for a different ontology/domain (see wikidata/wiki2 pilots).
    """
    try:
        # 1. Connect to the Database
        graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            database=database or config.NEO4J_DATABASE
        )
        
        # 2. Force a schema refresh so the LLM knows exactly what nodes exist today
        graph.refresh_schema()
        
        # 3. Build the translation chain using our imported custom prompt
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            cypher_prompt=cypher_prompt or CYPHER_GENERATION_PROMPT,
            validate_cypher=True,
            allow_dangerous_requests=True
        )

        print(f"--- [Local Tools] Neo4j KG Chain initialized successfully (database={database or config.NEO4J_DATABASE}) ---")
        return chain
        
    except Exception as e:
        print(f"--- [Local Tools Error] Failed to connect to Neo4j or build chain: {e} ---")
        return None