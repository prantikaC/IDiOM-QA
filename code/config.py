import math
import os
from dotenv import load_dotenv

# Load environment variables from a .env file located in the same directory
load_dotenv()

# ==========================================
# API Keys
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Add others as needed (e.g., ANTHROPIC_API_KEY, HUGGINGFACEHUB_API_TOKEN)

# ==========================================
# Model Configurations
# ==========================================
# The main LLM for the Orchestrator, Generator, and KG Cypher translation
MAIN_LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0

# The Cross-Encoder model used by the Reranker agent
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ==========================================
# Knowledge Graph Strict Ontology
# ==========================================
# The strict list of structural relationships allowed in the system.
# Must match the live Neo4j graph's actual relationship types (verified via
# `CALL db.relationshipTypes()`), which mirrors data/om_ontology.json — NOT
# an independently-invented list. This previously drifted from the real
# schema (had ISSUED_BY/SUPERSEDES/APPLIES_TO_INSTITUTE/HAS_BUDGET_HEAD/
# REPORTS_TO/HAS_COMMITTEE_MEMBER/MANDATES_COMPLIANCE_FROM — only ISSUED_BY
# actually exists as a relationship type), which fed a false-restrictive
# vocabulary into PLANNER_SYSTEM_PROMPT and silently under-routed KG-worthy
# sub-tasks about appointments/transfers/project funding/leadership.
# Supersession is NOT its own relationship type here — it's the `superseded`/
# `superseded_by` properties on Document nodes and every relationship (see
# kg_prompt.py's TEMPORAL AWARENESS rules), so it isn't listed as a relation.
KG_ALLOWED_RELATIONS = [
    "ISSUED_BY",
    "CONTAINS_DIRECTIVE",
    "AFFECTS",
    "APPOINTED_AS",
    "ASSOCIATED_WITH",
    "TRANSFERRED_FROM",
    "TRANSFERRED_TO",
    "MODIFIES_OR_CLARIFIES",
    "MEMBER_OF",
    "ANNOUNCES_POSITION",
    "FUNDS_PROJECT",
    "LEADS_PROJECT",
]

# ==========================================
# Database Configurations
# ==========================================
# Neo4j (Knowledge Graph) Settings
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")
# Default Neo4j database name. Left as "neo4j" (Neo4j's own default database)
# so this is a no-op for every existing OM call site. A separate benchmark
# pilot (e.g. 2WikiMultihopQA, see scripts/build_kg_wiki2.py) should point at
# a distinct database name via get_kg_chain(..., database=...) instead of
# ever touching this one -- the production OM knowledge graph must never be
# wiped/mixed with pilot data.
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Vector Store Settings
# E.g., Path to your local FAISS index or ChromaDB directory
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/local_db/faiss_index")

# BM25 Settings
# E.g., Path to your saved BM25 parameters or raw text corpus
BM25_CORPUS_PATH = os.getenv("BM25_CORPUS_PATH", "data/local_db/bm25_corpus.json")

# ==========================================
# Retrieval & Reranking Parameters
# ==========================================
# How many documents each individual retriever should ultimately return
TOP_K_RETRIEVE = 5

# Over-fetch factor: each retriever pulls TOP_K_RETRIEVE * RETRIEVE_OVERFETCH
# candidates internally, then applies the recency boost and trims back to
# TOP_K_RETRIEVE. >1 lets the recency tiebreaker meaningfully reorder the
# pool; =1 disables over-fetching.
RETRIEVE_OVERFETCH = 3

# How many top documents the Reranker should keep after evaluating the combined pool
# Calibrated (2026-09-06) via ablation/top_k_rerank.py: 3/5/7 sweep found 7 beating
# 5 and 3 on every surface metric, flat latency. Re-swept (2026-09-09) after the
# adaptive_rag context-volume-fairness discussion -- extended to 10 and 15 to check
# whether quality keeps climbing with volume the way adaptive_rag's did once its own
# top_k was uncapped. It doesn't: 3->5->7->10 keeps improving on every metric
# (BERTScore F1 0.8943/0.8951/0.8963/0.8971), but 10->15 is flat/noise-level
# (0.8971 vs 0.8972, rerank_10 actually edges out rerank_15 on ROUGE-1/METEOR) --
# a genuine diminishing-returns plateau around 10, not unbounded volume-scaling.
# 10 chosen over 15 at matched quality since it's the smaller prompt. Raised 7->10.
TOP_K_RERANK = 10

# Graph facts in the reranked context. A fact is kept only if the cross-encoder
# scores it at least KG_MIN_RERANK_SCORE (raw logit) against the sub-task, and
# at most KG_MAX_IN_CONTEXT facts are kept. Without this, generic matches
# (ISSUED_BY / CONTAINS_DIRECTIVE edges of unrelated OMs, median logit ~-3.5)
# took ~2 of the 10 slots in 58% of contexts and lowered context precision
# within every question type (RAGAS full-475 run, 2026-10-30). At 0.0, 35% of
# those graph facts pass, against 51% of text chunks.
KG_MIN_RERANK_SCORE = 0.0
KG_MAX_IN_CONTEXT = 3

# ==========================================
# Recency & Supersession Bias
# ==========================================
# Multiplicative boost applied to retriever candidates by issue_date.
# Score for the newest OM in the corpus is multiplied by (1 + LAMBDA);
# oldest by (1 + 0); linear in between. Keep small at the retriever stage
# so a highly relevant old OM still survives into the reranker pool.
# Calibrated (2026-09-06) via ablation/retriever_recency.py on the stratified
# 140-question validation set, CRITIC_ACCEPT_SCORE held at inf. baseline (0.15)
# vs retriever_off (0.0) vs retriever_double (0.30) came back flat and mixed-sign
# on both latency (69.8-74.1s across arms) and BERTScore F1 (0.8960/0.8954/0.8950
# -- a 0.001 spread, noise-level) with no question type showing a consistent
# winner either. No evidence to move off the existing default. Left unchanged.
RECENCY_LAMBDA_RETRIEVER = 0.15

# Stronger boost in the reranker, where the cross-encoder has already
# established topical relevance and we want recency to be a real tiebreaker.
# Calibrated (2026-09-08) via ablation/reranker_recency.py on the stratified
# 140-question validation set (RECENCY_LAMBDA_RETRIEVER held at 0.15,
# CRITIC_ACCEPT_SCORE at inf). reranker_boost (=0.45) beat baseline (=0.30),
# reranker_half (=0.15), and reranker_off (=0.0) on BERTScore F1 (0.8979 vs
# 0.8955) and on ROUGE/BLEU/METEOR too; latency was flat across all four arms
# (59.9-62.2s mean), so no cost tradeoff. Win is broad (Conceptual, CrossHop,
# Factoid, Temporal) but not universal -- baseline still edges it out on
# Relational and Multi-part. Raised from 0.30 to 0.45.
RECENCY_LAMBDA_RERANKER = 0.45

# Multiplier applied to a chunk's rerank score when its metadata reports
# the OM has been superseded by a later one. <1.0 demotes; 0 removes.
# Calibrated (2026-09-06) via ablation/superseded_penalty.py on the stratified
# 140-question validation set, CRITIC_ACCEPT_SCORE held at inf. hard_penalty
# (=0.02) beat baseline (=0.1) and no_penalty (=1.0) on every surface metric
# (BERTScore F1 0.8967 vs 0.8948 vs 0.8946; consistent margin on ROUGE/BLEU/
# METEOR), flat latency across arms. Contrary to the original design
# rationale, the gain is NOT concentrated on Temporal (hard_penalty actually
# scored lowest of the three there, 0.8737 vs baseline's 0.8768) -- it shows up
# broadly instead (Aggregation, CrossHop, Relational), consistent with a
# stale-but-superficially-relevant chunk occasionally winning a slot at 0.1
# and diluting the answer even outside date-sensitive questions. Lowered from
# 0.1 to 0.02.
# NOTE (2026-10-28): this calibration is INVALID. No OM in the corpus has
# superseded = 1 (ingest only detects explicit "in supersession of" phrases,
# and none point to an in-corpus OM), so the penalty never fired and the
# three arms above were the same system; their differences are run-to-run
# noise. The penalty is inert, and the paper no longer claims supersession
# handling. Do not cite these numbers.
SUPERSEDED_PENALTY = 0.02

# Corpus year bounds used to normalise the recency boost to [0, 1].
# Set automatically from the ingest log if reconcile_supersession.py is run,
# but kept here as overridable defaults.
CORPUS_OLDEST_YEAR = int(os.getenv("CORPUS_OLDEST_YEAR", "1989"))
CORPUS_NEWEST_YEAR = int(os.getenv("CORPUS_NEWEST_YEAR", "2025"))

# ==========================================
# Critic / Retry Control (latency tuning)
# ==========================================
# Max critic retry passes per sub-task before escalating to corrective rewrite.
# Empirically verified (2026-08-18, ollama backend, temperature=0 throughout):
# 3 previously-failing sub-tasks re-run 3x each with an unchanged current_task
# produced byte-identical rerank scores, doc counts, and critic verdicts every
# time — vector/bm25/kg/reranker/critic reference no retry-attempt counter, so
# a same-query retry cannot see new evidence. Set to 0: skip straight to the
# rewriter (see MAX_REWRITES) after the first failure, since the rewriter is
# the only tier that actually changes the input text.
MAX_CRITIC_RETRIES = 0

# Max corrective query-rewriting attempts per sub-task, fired AFTER
# MAX_CRITIC_RETRIES same-query retries are exhausted. The rewriter rephrases
# the failed sub-task using the last retrieval's context as evidence and hands
# control back to the orchestrator with loop_count reset. Each rewrite adds
# one LLM call (rewriter) + one orchestrator fan-out + one reranker + one
# critic call; budget ~4-6 seconds per fire on gpt-4o-mini. Set to 0 to
# disable corrective rewriting entirely (ablation).
# Re-validated (2026-09-06) via ablation/max_rewrites.py on the stratified
# 140-question validation set. Aggregate BERTScore F1 was flat/noise-level
# across 0/1/2 (0.8932 / 0.8946 / 0.8952) while latency scaled monotonically
# with the budget (43.8s / 70.9s / 94.7s mean) -- no clean aggregate case for
# either raising or cutting it. One small-n exception: Aggregation (n=11)
# favored rewrites_2 clearly (0.8316 vs rewrites_0's 0.8031), too small a
# sample to trust on its own. Kept at 1 (conservative -- unchanged from
# production) rather than acting on a flat/ambiguous aggregate result.
MAX_REWRITES = 1

# If the top cross-encoder rerank score clears this threshold, the Critic accepts
# the context immediately and SKIPS its LLM call (uses the reranked context as the
# fact). Saves an LLM round-trip + any retries on high-confidence retrievals.
# Scores are unbounded ms-marco-MiniLM logits (~ -11..+11).
#
# CRITICAL: the cross-encoder measures TOPICAL RELEVANCE, not answerhood. An
# out-of-scope ("Adversarial") question can land a topically-similar chunk with a
# high logit even though the corpus does NOT contain its answer. If the gate fires
# there, the sub-task is marked resolved and the non-answering context is forwarded
# as if it were the answer, bypassing the critic's refusal signal → hallucination
# risk. On the original v4 set, 5 Adversarial questions scored ≥5.0 (max 6.10),
# which motivated raising the threshold from 5.0 to 7.0.
#
# Disabled entirely (2026-09-05) after re-validating on the stratified 140-
# question validation set (scripts/build_validation_set.py — all 8 question
# types, not just the ones the original calibration checked):
# ablation/gate_stats.py measured CRITIC_ACCEPT_SCORE=7.0's actual Adversarial
# gate-fire rate at 23.5% (8/20 Adversarial questions cleared 7.0, scores
# 7.15-7.70) -- the "zero false-accepts at 7.0" property did NOT hold on this
# set, and 2 of those 8 produced confirmed hallucinations reaching the user
# (e.g. answering a question about IIT-Kharagpur, an out-of-domain
# institution, with fabricated specifics). Separately, the reranker_model
# ablation's RAGAS comparison found gate-off beating gate-on-7.0 on 5/6
# metrics. 9.0 was tested as a middle ground but measured at ~0% fire rate on
# this corpus (both the 184Q and 140Q sets) -- behaviorally indistinguishable
# from disabling it outright, so there is no real "cheap gate" latency win
# available at any threshold that's actually safe here. See
# ablation/critic_accept_score.py for the full 5-arm sweep this is based on.
CRITIC_ACCEPT_SCORE = math.inf

# Critic LLM cascade: critic/rewriter were promoted to always use reasoning_llm
# (see graph.py) because 7B-tier judgment was too weak for nuanced answerhood
# calls. But that means every sub-task that misses the cheap gate above pays
# the reasoning-model cost even when the context turns out fine on the first
# look. This cascades the LLM call itself: fast_llm judges first, and only
# sub-tasks it flags as unresolved escalate to reasoning_llm for the
# authoritative call that actually drives retry/rewrite/out-of-scope.
# Trade-off: this trusts fast_llm's POSITIVE verdicts with no second opinion,
# so a fast_llm false-accept (weak context waved through as resolved) is never
# caught by the stronger model — that's the exact failure mode the reasoning
# promotion was meant to prevent. Set False to keep every non-gated sub-task
# on reasoning_llm (safer, costlier — pre-cascade behavior).
ENABLE_CRITIC_CASCADE = True

# Explicit entity-disambiguation check (ablation, 2026-08-28). Prompted rule 5
# (ENTITY GROUNDING) already asks the critic to check the context concerns the
# same real-world entity as the query, but a per-row audit on Relational/
# Multi-part questions found it failing anyway on near-duplicate TEMPLATE
# documents (e.g. this corpus's recurring RA-I recruitment ads, structurally
# identical across different departments/rounds with a different named
# professor/contact each time) — 7 of 8 audited failures had an LLM critic
# actually run and still accept the wrong instance; 6 of those 7 even had
# reasoning_llm (not just fast_llm) look at it and still get fooled. Soft
# prose grounding isn't enough for that specific failure mode. This flag adds
# a second, more explicit structured field (query_entities /
# context_entity_match — see CriticDecisionEntityCheck in agent_critic.py)
# forcing the critic to name the specific identifying details (person,
# department, ad/reference number, date) implied by the sub-task and verify
# they match, not just the topic. Default False — test-only lever, not yet
# validated against a full run; see ablation/critic_entity_grounding.py.
ENABLE_CRITIC_ENTITY_CHECK = False

# Stronger-critic spike test (2026-09-04). Swaps ONLY the critic + rewriter's
# LLM to a different model, independent of the planner (which stays on the
# normal reasoning_llm) and independent of fast_llm. This is plumbing for a
# manual spike test on known-wrong Aggregation/CrossHop/Temporal questions
# BEFORE building the full feature (a planner few-shot classifier that
# flags "complex" questions and conditionally routes them here) — per the
# user's explicit sequencing: validate the model swap helps before
# investing in the classifier. None = disabled, normal behavior (critic +
# rewriter both use reasoning_llm, as always).
#
# Spec format: "<family>:<model>", e.g. "ollama:deepseek-r1:32b",
# "ollama:llama3.1:70b", "openai:gpt-4o-mini", "openai:gpt-5-mini" — see
# src/core/llm_factory.py's get_llm_by_spec(). See ablation/stronger_critic_spike.py.
CRITIC_REWRITER_LLM_OVERRIDE = None

# Full feature (2026-09-04), built after the spike above showed a real but
# modest, inconsistent signal (2 clean wins, ~9 no-change, 2 new
# hallucination-regressions across 15 known-wrong Aggregation/CrossHop/
# Temporal questions — see ablation/stronger_critic_spike.py's docstring).
# Rather than apply CRITIC_REWRITER_LLM_OVERRIDE to every sub-task in a run
# (the spike's blunt approach), the planner's existing complexity classifier
# (see COMPLEXITY_CLASSIFIER_PROMPT, agent_planner.py) now also emits
# needs_stronger_critic per query, using few-shot examples for the
# Aggregation/Temporal/CrossHop patterns that motivated this. When this flag
# is True, only queries the planner actually flags complex route their
# critic + rewriter calls to CRITIC_REWRITER_LLM_OVERRIDE (skipping the
# fast_llm cascade for those calls, same rationale as the spike); everything
# else keeps normal reasoning_llm behavior, unchanged.
#
# False (default): legacy/spike behavior — if CRITIC_REWRITER_LLM_OVERRIDE is
# set, it applies globally to EVERY sub-task regardless of complexity,
# exactly as ablation/stronger_critic_spike.py used it. Kept as the default
# so that script's existing runs are unaffected.
ENABLE_COMPLEX_CRITIC_ROUTING = False

# Single-hop fast path: a cheap fast_llm classifier runs before the planner. If the
# query is judged single-hop, decomposition is skipped (plan = [query]) and the
# expensive reasoning planner is never called — big latency/cost win for simple
# lookups, which dominate. Set False to always decompose (planner ablation).
ENABLE_SINGLE_HOP_FASTPATH = True

# KG routing: True runs the KG agent on every sub-task unconditionally (never
# silently misses a KG-answerable sub-task, at the cost of an extra Cypher
# LLM call + Neo4j round-trip on sub-tasks that don't need it). False
# restores the keyword-gated heuristic (_KG_KEYWORDS in agent_orchestrator.py)
# — cheaper, but only as complete as that keyword list. Toggle this to
# ablate the accuracy-vs-latency tradeoff between the two before deciding
# which ships in the final paper runs.
ENABLE_UNCONDITIONAL_KG = False

# Hard cap on planner sub-tasks. Each step is its own retrieve→rerank→critic pass
# (plus retries), so latency scales with step count. The planner tends to
# over-decompose (15+ steps for a simple comparison); this truncates the plan to a
# sane ceiling. Multi-hop queries rarely need more than a handful of steps.
MAX_PLAN_STEPS = 5

# Adaptive re-planning: after each sub-task resolves, the planner re-examines
# the remaining plan against what was actually found (final_context_string +
# the past_steps ledger), not just the original text-only estimate, and may
# split, merge, or reorder the remaining steps. Costs one extra reasoning_llm
# call per sub-task transition. Set False to keep the original fixed-plan
# behavior, where the plan's step count never changes after the first pass
# (ablation / cost-sensitive runs).
ENABLE_ADAPTIVE_REPLANNING = False

# Ceiling on TOTAL sub-tasks across a run once adaptive re-planning can grow
# the plan beyond its initial MAX_PLAN_STEPS estimate. Higher than
# MAX_PLAN_STEPS so a genuinely under-decomposed question has room to grow
# when the evidence justifies it, while still bounding runaway cost.
# Enforced as len(past_steps) + len(revised_plan) <= MAX_TOTAL_PLAN_STEPS on
# every re-plan. Unused if ENABLE_ADAPTIVE_REPLANNING is False.
MAX_TOTAL_PLAN_STEPS = 8

# LangGraph counts every superstep toward its recursion_limit (default 25) and
# aborts the run when it's hit. Our plan-execute loop legitimately needs far more
# than 25: a single unresolved sub-task runs up to
#   (MAX_REWRITES + 1) × (MAX_CRITIC_RETRIES + 1)
# retrieve→rerank→critic cycles — the rewriter resets the same-query budget, so a
# rewrite buys a whole fresh round of retries — and the plan can hold up to
# MAX_TOTAL_PLAN_STEPS such sub-tasks (adaptive re-planning's ceiling, not just
# the initial MAX_PLAN_STEPS estimate — re-planning can grow the plan past its
# first-pass size). With the rewriter in place the worst-case path exceeds 25
# even for ONE sub-task, tripping LangGraph one superstep before the critic's
# Tier-3 give-up valve can fire (surfaces as "Recursion limit of 25 reached").
# Derive the limit from the budgets so it stays correct when they're tuned,
# with a safety margin on top.
_CYCLES_PER_SUBTASK = (MAX_REWRITES + 1) * (MAX_CRITIC_RETRIES + 1)
_TICKS_PER_CYCLE    = 4   # orchestrator → 3 parallel retrievers (1 superstep) → reranker → critic
_TICKS_PER_SUBTASK  = _CYCLES_PER_SUBTASK * _TICKS_PER_CYCLE + MAX_REWRITES + 1  # + rewriter hops + planner
GRAPH_RECURSION_LIMIT = int((_TICKS_PER_SUBTASK * MAX_TOTAL_PLAN_STEPS + 5) * 1.3)

# ==========================================
# SQLite Database
# ==========================================
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/local_db/om_rag.db")

# ==========================================
# Qwen / vLLM Backend
# ==========================================
# These mirror the env vars read by llm_factory.py. Kept here for reference;
# the factory reads os.getenv() directly and does not import config.
#
# Default two-tier layout (recommended):
#   fast_llm      → Qwen2.5-7B-Instruct       port 8000  (~14 GB VRAM)
#   reasoning_llm → Qwen2.5-32B-Instruct-AWQ  port 8001  (~18 GB VRAM)
#
# Start vLLM:
#   vllm serve Qwen/Qwen2.5-7B-Instruct \
#       --port 8000 --gpu-memory-utilization 0.85 --dtype auto
#   vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
#       --port 8001 --gpu-memory-utilization 0.85 --dtype auto
#
# Single-server fallback (point both URLs at the same port):
#   QWEN_REASONING_BASE_URL=http://localhost:8000/v1
#   QWEN_REASONING_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
QWEN_FAST_BASE_URL      = os.getenv("QWEN_FAST_BASE_URL",      "http://localhost:8000/v1")
QWEN_FAST_MODEL         = os.getenv("QWEN_FAST_MODEL",         "Qwen/Qwen2.5-7B-Instruct")
QWEN_REASONING_BASE_URL = os.getenv("QWEN_REASONING_BASE_URL", "http://localhost:8001/v1")
QWEN_REASONING_MODEL    = os.getenv("QWEN_REASONING_MODEL",    "Qwen/Qwen2.5-32B-Instruct-AWQ")

# ==========================================
# GPU / Device
# ==========================================
# Default GPU index to use (0-based). Override with CUDA_DEVICE env var.
# On a multi-GPU machine (e.g. L40S with 3 devices) set CUDA_DEVICE=1 or =2.
# On CPU-only machines this is ignored — get_device() in src/core/device.py
# automatically falls back to "cpu" when CUDA is unavailable.
DEFAULT_CUDA_DEVICE = os.getenv("CUDA_DEVICE")  # must be set explicitly; no fallback