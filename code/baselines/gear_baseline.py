"""
GEAR — closed-domain adaptation of GeAR: Graph-enhanced Agent for
Retrieval-augmented Generation (Shen et al. 2024, arXiv:2412.18431, Huawei).
https://gear-rag.github.io/

Key idea (per the paper): an iterative agentic loop, up to ~4 steps:
  1. BM25 base retrieval
  2. Triple extraction from retrieved passages via LLM
  3. tripleLink — link extracted triples to real triples in a knowledge-graph
     index
  4. DTBS (Diverse Triple Beam Search) — beam search over the graph from the
     linked triples, with a diversity penalty to avoid redundant paths
  5. RRF fusion of BM25 passages + graph-derived passages
  6. Reasoning: does accumulated "gist memory" (working memory of evidence
     found so far) sufficiently answer the query?
  7. If not, rewrite the query and go to step 1; otherwise generate

Closed-domain adaptation (grounded against this project's live Neo4j graph —
labels/relations verified with `CALL db.labels()` / `db.relationshipTypes()`,
matching data/om_ontology.json's schema, NOT config.py's KG_ALLOWED_RELATIONS,
which is a stale/unused list):
  - Base retriever: BM25 only, matching the paper (no dense-retrieval
    addition — an earlier version of this file silently added a FAISS call
    mislabeled as part of "SyncGE"; that's removed here).
  - tripleLink: Cypher match on `id`/`description` substring against the
    extracted triple's subject/object (nodes here carry `id`, not `name` —
    confirmed against the live graph). This links by entity-mention overlap,
    not by aligning the extracted free-text predicate to the graph's actual
    relation types — predicate alignment isn't attempted, since it would
    need either a fixed extraction vocabulary matched to the ontology's 12
    relation types or a fine-tuned linker, neither of which the paper's
    description (as available to us) specifies precisely enough to
    reproduce exactly.
  - DTBS: real beam search — each hop expands every node in the current beam
    by one Cypher hop, then diversifies by round-robining across distinct
    relationship types before capping at BEAM_WIDTH, so no single relation
    type can dominate the beam. This is our own reading of "diversity
    penalty," not a transcription of the paper's exact penalty formula
    (unavailable to us) — flagged so it isn't mistaken for a verified match.
  - Gist memory: a real accumulator across iterations (list of passages
    found so far, carried into the next round's reasoning step) — the prior
    version of this file claimed this but didn't implement any iteration at
    all.
  - Reasoning + query rewriting: the paper's judgment-and-follow-up step,
    structurally the same role as IDiOM-RAG's critic + rewriter, so both run
    on reasoning_llm rather than fast_llm (see base_baseline.py docstring).
  - No web search (closed domain).

Pipeline:
  query --loop up to MAX_ITER times--> BM25 retrieve -> extract triples ->
    tripleLink -> DTBS expand -> RRF(BM25, graph) -> append to gist memory ->
    reasoning: sufficient? -> [done | rewrite query, loop]
  -> final RRF over every ranked list gathered across all iterations -> Generate
"""

import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from baselines.base_baseline import BaseBaseline
import config


# ── Schemas ─────────────────────────────────────────────────────────────────────

class Triple(BaseModel):
    subject: str = Field(description="Subject entity mention, verbatim from the passage.")
    predicate: str = Field(description="Short free-text relation phrase, e.g. 'appointed as'.")
    object: str = Field(description="Object entity mention, verbatim from the passage.")


class TripleList(BaseModel):
    triples: list[Triple] = Field(
        description="Proximal (subject, predicate, object) triples found in the passages. "
                    "Empty list if none found."
    )


class SufficiencyDecision(BaseModel):
    is_sufficient: bool = Field(
        description="True if the accumulated gist memory fully and directly answers the original query."
    )
    missing: str = Field(
        description="If not sufficient, one sentence describing what information is still missing. "
                    "Otherwise empty string."
    )


class FollowUpQuery(BaseModel):
    query: str = Field(
        description="A single targeted follow-up retrieval query for the missing information."
    )


# ── Prompts ─────────────────────────────────────────────────────────────────────

TRIPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are extracting proximal triples from administrative document passages
(CSIR/IACS/JU office memoranda) to seed a knowledge-graph search.

QUERY: {query}

PASSAGES:
{passages}

Extract (subject, predicate, object) triples that are relevant to answering
the query — e.g. (Person, "appointed as", Designation), (Document, "affects",
Scheme_or_Project). Use entity mentions verbatim from the passages so they
can be matched against a knowledge graph. Return an empty list if none found.
"""),
])

REASONING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are judging whether enough evidence has been gathered to answer a query
about administrative documents (CSIR/IACS/JU).

ORIGINAL QUERY: {query}

GIST MEMORY (evidence accumulated so far):
{gist_memory}

Does the gist memory fully and directly answer the original query?
- If yes: is_sufficient=True, missing=""
- If no : is_sufficient=False, missing="<what is missing, one sentence>"
"""),
])

FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
Based on what is missing, generate ONE precise follow-up retrieval query
that will fetch the missing information from an administrative document
corpus.

ORIGINAL QUERY: {query}
MISSING INFORMATION: {missing}

Return only the follow-up query string."""),
])

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an administrative assistant for CSIR/IACS/JU.
Answer the query using ONLY the provided context (text passages + graph triples).
If the context does not contain the answer, say:
"I could not find sufficient information in the internal documents."

CONTEXT:
{context}

QUERY: {query}

ANSWER:"""),
])


class GEARBaseline(BaseBaseline):
    """
    GEAR: iterative BM25 + triple-linked graph beam search + RRF fusion,
    gated by a reasoning/rewrite loop. See module docstring for exactly
    which parts are faithful to the paper vs. our documented approximation.
    """

    MAX_ITER   = 4  # paper: "up to 4 steps"
    HOP_DEPTH  = 2
    BEAM_WIDTH = 4

    def __init__(self, top_k: int = 5, backend: str = "openai"):
        super().__init__(top_k=top_k, backend=backend)
        self._init_neo4j()

        structured_triples = self.llm.with_structured_output(TripleList)
        structured_sufficiency = self.reasoning_llm.with_structured_output(SufficiencyDecision)
        structured_followup = self.reasoning_llm.with_structured_output(FollowUpQuery)

        self.triple_chain     = TRIPLE_PROMPT     | structured_triples
        self.reasoning_chain  = REASONING_PROMPT  | structured_sufficiency
        self.followup_chain   = FOLLOWUP_PROMPT   | structured_followup
        self.generator_chain  = GENERATOR_PROMPT  | self.llm

    def _init_neo4j(self):
        try:
            from langchain_neo4j import Neo4jGraph
            self.neo4j_graph = Neo4jGraph(
                url=config.NEO4J_URI,
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
            )
            self.neo4j_graph.refresh_schema()
            print("[GEAR] Neo4j connected.")
        except Exception as e:
            print(f"[GEAR] Neo4j unavailable: {e}. Graph step will be skipped.")
            self.neo4j_graph = None

    # ── Step: triple extraction ─────────────────────────────────────────────────

    def _extract_triples(self, query: str, passages: list[str]) -> list[Triple]:
        if not passages:
            return []
        try:
            result: TripleList = self.triple_chain.invoke({
                "query": query,
                "passages": "\n\n".join(passages[: self.top_k]),
            })
            return result.triples
        except Exception:
            return []

    # ── Step: tripleLink — match extracted triples to real graph triples ───────

    def _triple_link(self, triples: list[Triple]) -> tuple[list[str], list[str]]:
        """
        For each extracted triple, find real (a)-[r]-(b) triples in the graph
        whose endpoint matches the triple's subject or object by substring.
        Returns (passages, seed_node_ids) — passages formatted for context /
        RRF, seed_node_ids to start DTBS expansion from.
        """
        if self.neo4j_graph is None or not triples:
            return [], []

        passages: list[str] = []
        seeds: set[str] = set()

        for t in triples:
            for term in {t.subject.strip(), t.object.strip()}:
                if not term:
                    continue
                try:
                    rows = self.neo4j_graph.query(
                        """
MATCH (a)-[r]-(b)
WHERE replace(toLower(a.id), '_', ' ') CONTAINS $term
   OR any(x IN coalesce(a.aliases, []) WHERE replace(toLower(x), '_', ' ') CONTAINS $term)
   OR toLower(coalesce(a.description, '')) CONTAINS $term
RETURN a.id AS a_id, type(r) AS rel, b.id AS b_id,
       coalesce(b.description, '') AS b_desc
LIMIT $limit
""",
                        # Graph IDs use underscores for spaces (Jadavpur_University),
                        # so compare both sides lower-cased with "_" read as " ".
                        params={"term": term.lower().replace("_", " "), "limit": self.BEAM_WIDTH * 2},
                    )
                except Exception:
                    continue

                for row in rows:
                    a_id, rel, b_id, b_desc = row.get("a_id"), row.get("rel"), row.get("b_id"), row.get("b_desc")
                    if not a_id or not b_id:
                        continue
                    passage = f"{a_id} -[{rel}]-> {b_id}" + (f" ({b_desc})" if b_desc else "")
                    passages.append(passage)
                    seeds.add(a_id)
                    seeds.add(b_id)

        return passages, list(seeds)

    # ── Step: DTBS — diverse beam search expansion from seed nodes ─────────────

    def _dtbs_expand(self, seed_ids: list[str]) -> list[str]:
        """
        Beam search from `seed_ids` out to HOP_DEPTH hops. At each hop,
        candidates are grouped by relationship type and round-robined across
        types before capping at BEAM_WIDTH, so the beam can't be dominated by
        one relation type — our approximation of the paper's diversity
        penalty (see module docstring).
        """
        if self.neo4j_graph is None or not seed_ids:
            return []

        passages: list[str] = []
        beam = seed_ids[: self.BEAM_WIDTH]
        visited: set[str] = set(beam)

        for _ in range(self.HOP_DEPTH):
            by_rel: dict[str, list[tuple[str, str, str]]] = {}
            for node_id in beam:
                try:
                    rows = self.neo4j_graph.query(
                        """
MATCH (s)-[r]-(t)
WHERE s.id = $node_id
RETURN type(r) AS rel, t.id AS t_id, coalesce(t.description, '') AS t_desc
LIMIT $limit
""",
                        params={"node_id": node_id, "limit": self.BEAM_WIDTH * 3},
                    )
                except Exception:
                    continue

                for row in rows:
                    t_id = row.get("t_id")
                    if not t_id or t_id in visited:
                        continue
                    rel = row.get("rel", "RELATED")
                    by_rel.setdefault(rel, []).append((node_id, rel, t_id))

            if not by_rel:
                break

            # Round-robin across relation types so no single type dominates.
            next_beam: list[str] = []
            rel_types = list(by_rel.keys())
            idx = 0
            while len(next_beam) < self.BEAM_WIDTH and any(by_rel.values()):
                rel = rel_types[idx % len(rel_types)]
                idx += 1
                candidates = by_rel[rel]
                if not candidates:
                    continue
                s_id, r, t_id = candidates.pop(0)
                if t_id in visited:
                    continue
                visited.add(t_id)
                next_beam.append(t_id)
                passages.append(f"{s_id} -[{r}]-> {t_id}")

            if not next_beam:
                break
            beam = next_beam

        return passages

    # ── RRF fusion (unchanged — Cormack et al. 2009 standard formula) ──────────

    @staticmethod
    def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, passage in enumerate(ranked, start=1):
                scores[passage] = scores.get(passage, 0.0) + 1.0 / (k + rank)
        return [p for p, _ in sorted(scores.items(), key=lambda x: -x[1])]

    # ── Reasoning + rewrite (rewrite-analogous step -> reasoning_llm) ──────────

    def _sufficient(self, query: str, gist_memory: list[str]) -> SufficiencyDecision:
        return self.reasoning_chain.invoke({
            "query": query,
            "gist_memory": "\n\n".join(gist_memory[-self.top_k * 3:]) or "(none yet)",
        })

    def _rewrite(self, query: str, missing: str) -> str:
        result: FollowUpQuery = self.followup_chain.invoke({"query": query, "missing": missing})
        return result.query.strip()

    # ── Main pipeline ───────────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        t0 = time.time()
        try:
            gist_memory: list[str] = []
            all_ranked_lists: list[list[str]] = []
            current_query = query
            iterations_used = 0

            for i in range(self.MAX_ITER):
                iterations_used = i + 1

                # Step 1 — BM25 base retrieval
                bm25_docs = self.retrieve_bm25(current_query)
                bm25_passages = [d.page_content for d in bm25_docs]
                all_ranked_lists.append(bm25_passages)

                # Step 2 — triple extraction from this round's passages
                triples = self._extract_triples(current_query, bm25_passages)

                # Step 3 — tripleLink
                link_passages, seed_ids = self._triple_link(triples)

                # Step 4 — DTBS expansion from linked seeds
                expand_passages = self._dtbs_expand(seed_ids)
                graph_passages = link_passages + expand_passages
                if graph_passages:
                    all_ranked_lists.append(graph_passages)

                # Step 5 — this round's RRF(BM25, graph) feeds gist memory
                step_fused = self._rrf([bm25_passages, graph_passages]) if graph_passages else bm25_passages
                for p in step_fused[: self.top_k]:
                    if p not in gist_memory:
                        gist_memory.append(p)

                # Step 6 — reasoning: is gist memory sufficient?
                decision = self._sufficient(query, gist_memory)
                if decision.is_sufficient or not decision.missing.strip():
                    break

                # Step 7 — rewrite query and loop
                next_query = self._rewrite(query, decision.missing)
                if not next_query or len(next_query) < 5:
                    break
                current_query = next_query

            # Final RRF over every ranked list gathered across all iterations
            fused_final = self._rrf(all_ranked_lists) if all_ranked_lists else gist_memory
            final_passages = fused_final[: self.top_k * 3]
            context = "\n\n".join(final_passages)

            result = self.generator_chain.invoke({"query": query, "context": context})
            answer = result.content.strip()

            return {
                "answer":           answer,
                "context":          context,
                "iterations":       iterations_used,
                "gist_memory_size": len(gist_memory),
                "latency_seconds":  round(time.time() - t0, 2),
                "error":            None,
            }

        except Exception as e:
            return {
                "answer":          "",
                "context":         "",
                "iterations":      0,
                "gist_memory_size": 0,
                "latency_seconds": round(time.time() - t0, 2),
                "error":           str(e),
            }
