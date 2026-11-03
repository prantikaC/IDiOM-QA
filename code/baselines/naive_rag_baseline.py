"""
Naive RAG — the textbook retrieve-then-generate baseline.

Conceptual ancestor: Lewis et al. NeurIPS 2020 (arXiv:2005.11401)
  "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
Taxonomy entry:      Gao et al. 2023 (arXiv:2312.10997) §3.1 "Naive RAG"

Pipeline (single pass, no rerank / rewrite / critique):
  query → dense retrieve top-k → concat → generator

Used in this paper as the lower-bound reference for all other systems.
"""

import time
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an administrative assistant for CSIR/IACS/JU/UGC.
Answer the query using ONLY the provided context.
If the context does not contain the answer, say:
"I could not find sufficient information in the internal documents."

CONTEXT:
{context}

QUERY: {query}

ANSWER:"""),
])


class NaiveRAGBaseline(BaseBaseline):
    """Single-pass dense retrieve-then-generate."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        self.generator_chain = GENERATOR_PROMPT | self.llm

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            docs    = self.retrieve_vector(query)
            context = "\n\n".join(d.page_content for d in docs)
            result  = self.generator_chain.invoke({"query": query, "context": context})
            answer  = result.content.strip()

            return {
                "answer":          answer,
                "context":         context,
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
