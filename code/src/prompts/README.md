# Prompts Directory

## Purpose
This directory centralizes all system prompts and prompt templates used by LLMs across the multi-agent system.

By keeping prompts separated from the Python execution logic, we make it easier to tune agent behaviors, update institutional rules, and manage context windows without breaking the LangGraph architecture.

## Key Files

**`planner_prompt.py`** — Two prompts:
- `COMPLEXITY_CLASSIFIER_PROMPT`: Used by the fast_llm to decide single-hop vs multi-hop. Defaults to single-hop unless the query clearly requires multiple distinct retrievals or a dependency chain.
- `PLANNER_SYSTEM_PROMPT`: Used by the reasoning_llm to decompose a multi-hop query into sequential sub-tasks. Enforces minimal decomposition (1–3 steps for most queries) and per-entity retrieval for comparisons.

**`critic_prompt.py`** — `CRITIC_SYSTEM_PROMPT`: Strict QA rules for the LLM judge. Template variables: `{original_query}`, `{current_task}`, `{context}`. Forces the critic to:
- Extract a concise fact if the context definitively answers the sub-task.
- Return `is_resolved=False` if context is only tangentially related.
- Set `out_of_scope=True` if the entity asked about is entirely absent from context (corpus has no coverage), after the retry budget is spent.
Returns a structured `CriticDecision` (`is_resolved`, `reasoning`, `extracted_fact`, `out_of_scope`).

**`rewriter_prompt.py`** — `REWRITER_SYSTEM_PROMPT`: Corrective query rewriting. Template variables: `{original_query}`, `{current_task}`, `{critic_reasoning}`, `{past_steps}`, `{task_variants}`, `{final_context_string}`.
- Uses the **critic's diagnosis** (`critic_reasoning`) as the primary rewriting signal.
- Receives `task_variants` (all phrasings already tried) to avoid regenerating the same rewrite.
- Instructs the rewriter to target the same underlying fact with different vocabulary or scope.

**`generator_prompt.py`** — `GENERATOR_SYSTEM_PROMPT`: Final synthesis. Template variables: `{original_query}`, `{past_steps_ledger}`, `{out_of_scope_list}`, `{raw_context}`.
- `past_steps_ledger`: Only resolved facts (sentinels filtered out by the agent before this call).
- `out_of_scope_list`: Structured list of sub-tasks confirmed absent from corpus; generator explicitly acknowledges these rather than fabricating answers.
- `raw_context`: Last retrieved context as a supplementary fallback for partial-evidence cases; overridden by ledger facts when both are available.
Enforces: direct answers (no question restating), recency preference, supersession chain citation, date/ID format preservation, entity faithfulness, and anti-hallucination refusal.

**`kg_prompt.py`** — `CYPHER_GENERATION_PROMPT`: Forces the LLM to write valid Neo4j Cypher using generic node variables `(s)` and `(t)` and rely on relationship types, not node labels. Required because the KG was extracted dynamically and cannot use strict entity label constraints.

---

## For Future Developers

**1. Treat Prompts like Code:**
If the critic is rejecting good context, or the rewriter keeps generating the same phrasing, the issue is almost always in the prompt. Check the relevant prompt file before touching Python logic.

**2. Template Variable Consistency:**
Every `{variable}` in a prompt must be matched by a key in the corresponding `chain.invoke({...})` call in the agent file. Adding a variable to the prompt without updating the agent (or vice versa) will raise a `KeyError` at runtime.

**3. Critic → Rewriter handoff:**
`{critic_reasoning}` in the rewriter prompt is populated from `state["critic_reasoning"]`, which the critic writes when routing to the rewriter. If you modify what the critic writes to state, update both the agent and this prompt.

**4. Generator sentinel handling:**
The generator agent filters `[OUT_OF_SCOPE]` and `[RETRIEVAL_FAILED]` sentinels from `past_steps` before building `past_steps_ledger`. The prompt receives clean resolved facts only. The sentinel tasks are surfaced separately via `{out_of_scope_list}`. Do not add logic to the prompt to detect sentinels — keep that in the agent.

**5. Regression Testing:**
Prompts are deeply interconnected. When updating any prompt, run the evaluation suite on a sample of questions to check for regressions in faithfulness and refusal behaviour.
