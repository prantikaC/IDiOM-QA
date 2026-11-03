"""
Gold-context reference point (oracle retrieval) — not a RAG system.

The same generator LLM as every baseline answers from the question's gold
evidence (`context_used` in the dataset) instead of retrieved passages. This
is an upper bound on what better retrieval alone could achieve: the gap
between it and a RAG system is retrieval loss, and a low score here means the
category is hard to answer even with the right evidence in hand.

Prompt and refusal wording mirror naive_rag_baseline.py so the only
difference from Naive RAG is where the context comes from.
"""

import json
import time
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


GOLD_DATASET = "data/eval/v4/idiom_qa_v4.json"

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


class GoldContextBaseline(BaseBaseline):
    """Generator over the gold evidence — no retrieval."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        self.generator_chain = GENERATOR_PROMPT | self.llm
        with open(GOLD_DATASET, "r", encoding="utf-8") as f:
            self.gold = {r["input"]: str(r.get("context_used", "")) for r in json.load(f)}

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            if query not in self.gold:
                raise KeyError(f"question not in {GOLD_DATASET}")
            context = self.gold[query]
            result = self.generator_chain.invoke({"query": query, "context": context})
            return {
                "answer":          result.content.strip(),
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
