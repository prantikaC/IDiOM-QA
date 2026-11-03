"""
Shared infrastructure for all baselines.

Every baseline inherits BaseBaseline and calls self.vectorstore,
self.bm25, and self.neo4j_graph for retrieval. Both self.llm (fast_llm)
and self.reasoning_llm are exposed; most single-pass baselines only ever
touch self.llm. The iterative/agentic baselines (Adaptive-RAG's
Strategy-C follow-up generator; DeepRAG's retrieve-vs-parametric step;
R3-RAG's and Search-R1's search-or-answer decision loops) use
self.reasoning_llm for that judgment step specifically, mirroring
IDiOM-RAG's own critic/rewriter allocation — otherwise those baselines
would be judged on a weaker model than the system they're benchmarked
against for structurally the same kind of decision.

All baselines run on the same backend LLM pair (see --backend), by
design — this keeps the RAG strategy as the only varying factor across
the comparison. Self-RAG was dropped from this suite specifically
because its entire contribution (the [ISREL]/[ISSUP] reflection-token
vocabulary) is trained into a specific Llama2-7B checkpoint and can't be
prompted onto a different backend without losing the thing being tested,
which would have made it the one baseline running on a different LLM
than everything else.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

import config
from src.core.llm_factory import get_llms
from src.tools.local_tools import (
    get_local_vector_store,
    get_local_bm25_retriever,
)


class BaseBaseline:
    """
    Lazily initialises shared retrieval infrastructure once per process.
    Subclasses call self.retrieve_vector(), self.retrieve_bm25(), and
    optionally self.neo4j_graph directly.

    Args:
        top_k:   number of docs per retriever
        backend: "openai" or "ollama" — selects which LLM pair to use
    """

    _shared_vectorstore = None
    _shared_bm25 = None

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        self.top_k = top_k
        self.backend = backend
        # Most baselines are single-pass and only use the fast LLM. A few
        # (Agent-G, Adaptive-RAG) also use reasoning_llm for their
        # critic-analogous judgment step — see module docstring.
        self.llm, self.reasoning_llm = get_llms(backend)

        if BaseBaseline._shared_vectorstore is None:
            print("[BaseBaseline] Loading FAISS vectorstore...")
            BaseBaseline._shared_vectorstore = get_local_vector_store()

        if BaseBaseline._shared_bm25 is None:
            print("[BaseBaseline] Loading BM25 retriever...")
            BaseBaseline._shared_bm25 = get_local_bm25_retriever()

        self.vectorstore = BaseBaseline._shared_vectorstore
        self.bm25 = BaseBaseline._shared_bm25

        # Neo4j: initialise lazily in subclasses that need it
        self.neo4j_graph = None

    def retrieve_vector(self, query: str) -> list[Document]:
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        return retriever.invoke(query)

    def retrieve_bm25(self, query: str) -> list[Document]:
        self.bm25.k = self.top_k
        return self.bm25.invoke(query)

    def run(self, query: str) -> dict:
        """
        Override in each subclass. Must return a dict with at least:
          {"answer": str, "latency_seconds": float, "error": str | None}
        """
        raise NotImplementedError
