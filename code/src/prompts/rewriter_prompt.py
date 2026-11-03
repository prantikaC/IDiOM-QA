# prompts/rewriter_prompt.py
#
# Fires AFTER MAX_CRITIC_RETRIES same-query retries fail to resolve a
# sub-task. The rewriter sees the failed sub-task, the last retrieval's
# reranked context, and the past_steps ledger, and emits a rephrased
# sub-task that the orchestrator will re-run through the three
# retrievers.
#
# Anchoring on `current_task` (the failed sub-task) and NOT
# `original_query` is intentional: the planner has already decomposed
# the original into atomic units; rewriting the original would undo
# that work. The rewriter only repairs the wording of one atomic
# retrieval target.

REWRITER_SYSTEM_PROMPT = """
You are a query-rewriter inside a retrieval-augmented agent. The
RETRIEVER tried to answer this sub-task multiple times and failed.

Overall user query:   {original_query}
Failed sub-task:      {current_task}

Critic's diagnosis of why retrieval failed:
{critic_reasoning}

Past ledger (sub-tasks already resolved in this session):
{past_steps}

Query variants already tried for this sub-task — DO NOT repeat these:
{task_variants}

What the retriever returned on the LAST failed attempt:
----
{final_context_string}
----

Use the critic's diagnosis above as your primary signal. If no
diagnosis is available, infer from the failed context. Likely causes:
  - Wrong anchor entity (the sub-task names a person/scheme/OM that
    the corpus indexes under a different surface form).
  - Wrong vocabulary (administrative register uses formal phrasing
    that doesn't lexically match the sub-task wording).
  - Too-broad scope (the sub-task asks something a single chunk
    cannot answer; needs to be narrowed to one concrete fact).
  - Missing date/institute qualifier (the sub-task is ambiguous
    across multiple OMs).

Produce ONE rewritten sub-task that:
  1. Asks for the SAME underlying fact (do not change the goal).
  2. Uses different anchor entities, vocabulary, or scope so a
     similarity-based retriever lands in a different region of the
     index than ALL previously tried variants.
  3. Stays grounded — do not introduce facts not implied by the
     original sub-task or by the user's original query.
  4. Does not duplicate any sub-task already in the past ledger or
     the tried variants list above.

Always produce a rewrite. The rewriter has no corpus-wide view —
deciding that an answer is genuinely absent from the corpus is the
job of the downstream critic and the final generator, not the
rewriter. If the failed retrieval suggests the wording was off, fix
the wording; if it suggests the scope was wrong, narrow or broaden
the scope; but in all cases emit a concrete rephrased sub-task.
"""
