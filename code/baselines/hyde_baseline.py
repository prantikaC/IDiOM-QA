"""
HyDE — Hypothetical Document Embeddings.

Original paper: Gao et al., ACL 2023 (arXiv:2212.10496)
  "Precise Zero-Shot Dense Retrieval without Relevance Labels"

Key idea: instead of embedding the user's question and retrieving similar
documents, ask the LLM to *write* a hypothetical passage that would answer
the question, then embed *that* passage and retrieve against it. The
synthetic passage tends to live closer to the corpus distribution than the
query does, which improves dense retrieval on out-of-vocabulary or
domain-shifted queries.

Implementation notes — divergence from the reference implementation
--------------------------------------------------------------------
Verified against the authors' actual repo (texttron/hyde,
src/hyde/generator.py + hyde.py). The real default samples n=8
hypothetical passages per query (temperature 0.7), then embeds and
AVERAGES all 9 vectors together — the original query's embedding PLUS
all 8 passage embeddings — into one search vector, which damps the noise
from any single hallucinated passage.

This implementation is a SIMPLIFIED single-sample variant: it generates
ONE passage and searches with that passage's text directly (no
query-embedding blending). This was an undisclosed simplification in an
earlier version of this file — flagging it now. Trade-off: 8x fewer LLM
calls per question (meaningful at 475 questions × several baselines),
at the cost of the variance-reduction the paper's multi-sample averaging
provides. If this baseline's quality looks like a weak point in results,
this is the first thing to reconsider — see run() below for where to
plug in multi-sample averaging if it's worth the added cost.

Pipeline:
  query → LLM hypothetical-passage generator (n=1, not paper's n=8) →
          embed passage → dense retrieval → generator
"""

import time
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline


HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
Write a brief passage (3-5 sentences) that could plausibly appear in an
official Office Memorandum issued by an Indian university, research
institute, or governmental body (e.g., CSIR, IACS, JU, UGC). The passage
should directly answer the question below, in the formal administrative
register typical of such documents (specific designations, dates,
amounts, scheme names, directive numbers where appropriate).

Output ONLY the passage. No preamble, no explanation, no quotes around it.
If the question cannot be answered by an administrative document, write a
short passage stating the relevant policy is not specified.

QUESTION: {query}

PASSAGE:"""),
])

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


class HyDEBaseline(BaseBaseline):
    """HyDE adapted for the OM corpus."""

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        self.hyde_chain      = HYDE_PROMPT | self.llm
        self.generator_chain = GENERATOR_PROMPT | self.llm

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _hypothetical_passage(self, query: str) -> str:
        result = self.hyde_chain.invoke({"query": query})
        return result.content.strip()

    def _retrieve_with_passage(self, passage: str) -> list:
        # Re-use the vector store but search with the hypothetical passage
        # instead of the raw query — this is the entire HyDE trick.
        return self.vectorstore.similarity_search(passage, k=self.top_k)

    def _generate(self, query: str, context: str) -> str:
        result = self.generator_chain.invoke({"query": query, "context": context})
        return result.content.strip()

    # ── Main pipeline ───────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            passage = self._hypothetical_passage(query)
            docs    = self._retrieve_with_passage(passage)
            context = "\n\n".join(d.page_content for d in docs)
            answer  = self._generate(query, context)

            return {
                "answer":              answer,
                "context":             context,
                "hypothetical_passage": passage,
                "latency_seconds":     round(time.time() - t0, 2),
                "error":               None,
            }

        except Exception as e:
            return {
                "answer":              "",
                "context":             "",
                "hypothetical_passage": "",
                "latency_seconds":     round(time.time() - t0, 2),
                "error":               str(e),
            }
