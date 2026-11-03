# Agents Directory

## Purpose
This directory contains the individual cognitive nodes (Agents) that make up the Multi-Agent RAG system. Each file represents a specialized agent with a single, distinct responsibility.

The system relies on LangGraph to pass the `MultiAgentRAGState` between these agents, allowing them to collaborate, route tasks, retrieve data, and critique findings.

## Key Agents

### 1. Planning & Routing

**`agent_planner.py`** — The first node. Runs a cheap complexity classifier (fast_llm) to decide single-hop vs multi-hop. Single-hop queries skip decomposition entirely (`plan = [query]`); multi-hop queries are decomposed by the reasoning_llm into a `Plan` of sequential sub-tasks (capped at `MAX_PLAN_STEPS`).

**`agent_orchestrator.py`** — Routing node. Uses a keyword heuristic (`_KG_KEYWORDS`) to decide whether the KG agent is likely to help for the current sub-task. Sets `active_retrievers` in state (`["vector", "bm25"]` or `["vector", "bm25", "kg"]`). Vector and BM25 always run; KG is skipped when no relational keywords are present in the sub-task, saving a Cypher LLM call + Neo4j round-trip.

### 2. Internal Retrievers (The Local "Brain")

**`agent_vector.py`** — Semantic searcher. Connects to FAISS with recency boost and over-fetching to pull conceptually similar context.

**`agent_bm25.py`** — Lexical searcher. Connects to the BM25 index for exact keyword, date, and document ID matches.

**`agent_kg.py`** — Relational searcher. Checks `active_retrievers` first; if `"kg"` is not in the list, returns empty immediately. Otherwise uses a Text-to-Cypher chain to query Neo4j for structural relationships between institutional entities.

All three run in **parallel** (same LangGraph superstep) and write to `gathered_context` via a merge reducer. The reranker runs once after all three complete.

### 3. Quality Control & Synthesis

**`agent_reranker.py`** — Deduplicates chunks from all retrievers and scores them with a CrossEncoder (`ms-marco-MiniLM-L-6-v2`). Scoring is done against `current_task` (the focused sub-task), not the full original query, so documents relevant to the specific atomic question score correctly in multi-hop settings. Applies recency boost and supersession penalty. Exposes `top_rerank_score` (raw logit) and `final_context_string` (formatted ranked context) to the critic.

**`agent_critic.py`** — LLM-as-a-Judge (reasoning_llm for `qwen` backend; fast_llm otherwise). Evaluates whether the reranked context resolves the current sub-task. Four branches:
- **Resolved** → extracts fact, appends `(task, fact)` to `past_steps`, routes to planner (next task) or generator (done).
- **Retry** (loop_count < MAX_CRITIC_RETRIES) → increments `loop_count`, routes back to orchestrator.
- **Out of scope** (post-retry, entity absent from corpus) → appends task to `out_of_scope_tasks`, writes `[OUT_OF_SCOPE]` sentinel to `past_steps`, short-circuits to planner/generator.
- **Rewrite** (retries exhausted, in-corpus) → writes `critic_reasoning` to state, routes to rewriter.
- **Force-progress** (both budgets spent) → writes `[RETRIEVAL_FAILED]` sentinel to `past_steps`, routes to planner/generator.

**`agent_rewriter.py`** — Corrective query rewriter (reasoning_llm for `qwen` backend; fast_llm otherwise). Fires after `MAX_CRITIC_RETRIES` same-query retries fail. Reads `critic_reasoning` (the critic's diagnosis of why retrieval failed) and `task_variants` (all phrasings already tried), and produces a rephrased sub-task that targets the same fact via different vocabulary or scope.

**`agent_generator.py`** — Final synthesizer (fast_llm). Reads the `past_steps` ledger, separating resolved facts from `[OUT_OF_SCOPE]` and `[RETRIEVAL_FAILED]` sentinels. Also reads `out_of_scope_tasks` (structured refusal list) and `final_context_string` (raw context as a supplementary fallback). Contains Python-level intercepts that produce refusal responses without an LLM call when all sub-tasks failed.

### 4. External Fallbacks (The "Internet")

**`agent_web_search.py`** — Standard fallback via DuckDuckGo (not active in the current evaluation pipeline).

**`agent_mcp_web.py`** — Live browser driver via Puppeteer/MCP (not active in the current evaluation pipeline).

---

## State Sentinels in `past_steps`

The critic writes structured sentinel strings to `past_steps` so the generator can distinguish failure types without string parsing:

| Sentinel | Meaning | Source |
|---|---|---|
| `[OUT_OF_SCOPE]` | Corpus confirmed has no coverage for this entity | Critic: `out_of_scope=True` after retries |
| `[RETRIEVAL_FAILED]` | Retrieval exhausted all budgets, answer may exist but couldn't be found | Critic: Tier 3 force-progress |

Both are filtered out of the ledger the generator shows the LLM; `out_of_scope_tasks` provides them as a clean list.

---

## LLM Allocation by Backend

| Agent | openai / ollama | qwen |
|---|---|---|
| Planner (decompose) | reasoning_llm | reasoning_llm |
| Planner (classify) | fast_llm | fast_llm |
| KG Cypher chain | fast_llm | fast_llm |
| **Critic** | fast_llm | **reasoning_llm** |
| **Rewriter** | fast_llm | **reasoning_llm** |
| Generator | fast_llm | fast_llm |

For the Qwen backend the 7B↔32B quality gap is large enough that the critic and rewriter are promoted to the reasoning model. For openai/ollama the fast model is capable enough and saves cost/latency.

---

## For Future Developers

**1. State Management is King:**
`gathered_context` is wiped between sub-tasks (critic returns `None` as the reset signal; the reducer treats `None` as a full clear). If you add a new retriever, ensure it appends to this field (not overwrites) and that it checks `active_retrievers` if selective routing applies.

**2. Orchestrator KG routing:**
The keyword list `_KG_KEYWORDS` in `agent_orchestrator.py` is derived from `config.KG_ALLOWED_RELATIONS` semantics. If you extend the knowledge graph with new relationship types, add corresponding keywords there.

**3. Web Searchable Flag (`is_web_searchable`):**
The orchestrator flags whether a query is safe for public web search. The generator uses this flag to block the web search agent from sending internal keywords to DuckDuckGo.

**4. `CRITIC_ACCEPT_SCORE` fast-gate:**
When `top_rerank_score ≥ CRITIC_ACCEPT_SCORE`, the critic accepts context without an LLM call. This threshold was calibrated against `original_query`-based scoring. After the reranker fix (now scores against `current_task`), the logit distribution will differ — re-validate the threshold before publishing results.
