# Core Directory

## Purpose
This directory is the central nervous system of the application. It defines the strict rules of the environment, manages the shared memory, and compiles the individual agents into a functional LangGraph state machine.

## Key Files

### 1. `state.py` (The Shared Memory)

Defines `MultiAgentRAGState` as a Python `TypedDict`. Every field is either a plain value (overwritten each write) or annotated with a reducer (accumulated across writes).

**Accumulator fields** (`Annotated[..., operator.add]`):
| Field | What accumulates |
|---|---|
| `messages` | Full message history |
| `past_steps` | Verified fact ledger — `(task, fact)` tuples for each resolved sub-task |
| `cypher_queries` | All Cypher strings executed against Neo4j |
| `total_retries` | Running retry count across all sub-tasks |
| `all_agent_sources` | Attribution of which retrievers contributed across the run |
| `task_variants` | Every rewritten sub-task form tried (audit trail) |
| `out_of_scope_tasks` | Sub-tasks confirmed absent from corpus (critic's structured refusal list) |

**Reset/overwrite fields** (plain, no reducer):
| Field | Role |
|---|---|
| `gathered_context` | Uses a custom `merge_gathered_context` reducer: **concurrent writes merge** (three parallel retrievers), **`None` is a reset signal** (sent by the critic between sub-tasks) |
| `final_context_string` | Formatted reranked context for the current sub-task |
| `top_rerank_score` | Best raw cross-encoder logit from the latest rerank pass |
| `current_task` | The sub-task currently being resolved (rewritten by the rewriter on retry) |
| `loop_count` | Same-query retry counter; reset to 0 when a sub-task resolves or a rewrite fires |
| `rewrite_count` | Corrective rewrite counter; reset to 0 when a sub-task resolves |
| `critic_reasoning` | Critic's LLM explanation of why retrieval failed; written when routing to the rewriter so the rewriter has a precise diagnosis rather than re-inferring from raw context |
| `active_retrievers` | List set by the orchestrator each sub-task: `["vector","bm25"]` or `["vector","bm25","kg"]` |

### 2. `graph.py` (The Compiler)

Imports all agents, defines them as LangGraph nodes, wires conditional routing, and compiles the application.

**LLM allocation:**
```
fast_llm      → generator, KG Cypher chain, complexity classifier in planner
reasoning_llm → planner (decomposition), critic*, rewriter*

* For the qwen backend only: the 7B↔32B quality gap is large, so critic and
  rewriter are promoted to reasoning_llm. For openai/ollama the fast model
  is used for critic and rewriter (cost/latency saving).
```

**Key routing:**

```
START → Planner
Planner → Orchestrator  (tasks remain)
        → Generator     (plan exhausted)

Orchestrator → Vector ∥ BM25 ∥ KG  (parallel fan-out)
[all three]  → Reranker              (fan-in after all complete)
Reranker     → Critic

Critic → Orchestrator  (retry, loop_count < MAX_CRITIC_RETRIES)
       → Rewriter      (retries exhausted, rewrite budget remains)
       → Planner       (sub-task resolved or out-of-scope, tasks remain)
       → Generator     (sub-task resolved or out-of-scope, plan done)

Rewriter → Orchestrator  (fresh retrieval pass with rewritten query)
Generator → END
```

**Graph construction:** `build_sequential_graph(backend)` takes a backend string (`"openai"`, `"ollama"`, `"qwen"`), initialises LLMs via `get_llms(backend)`, selects `_critic_llm` based on backend, and wires all edges.

### 3. `llm_factory.py` (The LLM Factory)

Returns `(fast_llm, reasoning_llm)` for a given backend. Three backends supported:

| Backend | fast_llm | reasoning_llm |
|---|---|---|
| `openai` | gpt-4o-mini | gpt-5-mini |
| `ollama` | qwen2.5:7b | qwen2.5:32b |
| `qwen` | Qwen2.5-7B-Instruct (port 8000) | Qwen2.5-32B-Instruct-AWQ (port 8001) |

For the `qwen` backend, each model runs on its own vLLM server (OpenAI-compatible API). Configure via env vars:
```
QWEN_FAST_BASE_URL       (default: http://localhost:8000/v1)
QWEN_FAST_MODEL          (default: Qwen/Qwen2.5-7B-Instruct)
QWEN_REASONING_BASE_URL  (default: http://localhost:8001/v1)
QWEN_REASONING_MODEL     (default: Qwen/Qwen2.5-32B-Instruct-AWQ)
```

### 4. `workflow_plot.py` (The Visualizer)

Utility that draws the compiled LangGraph state machine and saves `workflow_architecture.png`. Re-run after any routing or edge changes.

---

## For Future Developers

**1. Understanding State Mutability:**
When an agent returns a dict, LangGraph updates state based on the field's annotation:
- `past_steps` has `operator.add` → `{"past_steps": [(task, fact)]}` **appends**.
- `gathered_context` has `merge_gathered_context` → concurrent writes **merge**; `None` **clears**.
- Plain fields → `{"current_task": "..."}` **overwrites**.

**2. Adding a New Agent:**
Open `graph.py` and:
1. Import the factory function.
2. Initialise it: `my_agent = get_my_agent(...)`
3. Add as a node: `workflow.add_node("my_agent", my_agent)`
4. Wire edges to/from it (e.g., `workflow.add_edge("orchestrator", "my_agent")` and `workflow.add_edge("my_agent", "reranker")`)
5. If it's a conditional retriever, add its name to the orchestrator's `active_retrievers` logic and add an `active_retrievers` gate inside the agent.

**3. `CRITIC_ACCEPT_SCORE` re-validation:**
The fast-gate threshold (`config.CRITIC_ACCEPT_SCORE = 7.0`) was calibrated when the reranker scored against `original_query`. It now scores against `current_task`. Re-validate this threshold on the evaluation set before publishing results — the logit distribution under sub-task scoring will differ from full-query scoring.

**4. Graph Visualisation:**
Run `python src/core/workflow_plot.py` after any routing changes to regenerate the diagram.
