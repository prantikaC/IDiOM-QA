"""
Closed-book reference point (no retrieval) — not a RAG system.

The same generator LLM as every baseline answers from its own parametric
knowledge, with no documents. A low score shows IDiOM-QA cannot be answered
from what the model already knows, i.e. the benchmark actually requires
retrieval over the OM corpus.

Prompt and refusal wording mirror naive_rag_baseline.py so the only
difference from Naive RAG is the missing context.
"""

import time
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an administrative assistant for CSIR/IACS/JU/UGC.
Answer the query from your own knowledge.
If you do not know the answer, say:
"I could not find sufficient information in the internal documents."

QUERY: {query}

ANSWER:"""),
])


class ClosedBookBaseline(BaseBaseline):
    """Generator only — no retrieval."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        self.generator_chain = GENERATOR_PROMPT | self.llm

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            result = self.generator_chain.invoke({"query": query})
            return {
                "answer":          result.content.strip(),
                "context":         "",
                "latency_seconds": round(time.time() - t0, 2),
                "error":           None,
            }
        except Exception as e:
            return {
                "answer":          "",
                "context":         "",
                "latency_seconds": round(time.time() - t0, 2),
                "error":           str(e),
            }
