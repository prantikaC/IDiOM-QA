
"""
IDiOM-QA v2 question generator (unified, decision-logged).

Pipeline:
  Stage 1  Feasibility gate  ── per OM, per type    (single-doc types only)
  Stage 2  Generation        ── per feasible slot, K candidates
  Stage 3  Dedup              ── string + semantic, order-greedy

Quality scoring (Correctness/Fluency/Completeness rubric) is handled by the
SEPARATE downstream annotator pipeline (Claude Sonnet 4.6 + 2 humans). This
script's job is purely construction-time gating.

Rejection decisions (feasibility + dedup) are written to
`data/eval/v3/rejection_log.jsonl` for downstream RLHF training.

Reads the VLM-transcribed corpus at `data/DataFiles_v2/{csir,iacs,ju,ugc}/`.

Usage:
    # 5 single-doc types over all 116 OMs, K=2 candidates per feasible slot
    python scripts/generate_dataset.py --types single-doc --per-slot 2

    # Cross-doc temporal only
    python scripts/generate_dataset.py --types temporal

    # Everything (single-doc + temporal + aggregation)
    python scripts/generate_dataset.py --types all --per-slot 2

    # Smoke test on first 5 OMs, feasibility-only
    python scripts/generate_dataset.py --types single-doc --limit-docs 5 --stop-after feasibility
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import random
import argparse
from glob import glob
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal

import frontmatter
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

load_dotenv()


def make_llm(model_name: str, temperature: float = 0.0):
    """Pick the LLM client by model-name prefix.

    Cross-family pipelines (e.g., Claude for feasibility-judging +
    GPT for question generation) reduce self-preference bias compared
    to using the same model for both stages.

      claude-*  → Anthropic (requires ANTHROPIC_API_KEY in .env)
      anything else → OpenAI (requires OPENAI_API_KEY in .env)
    """
    if model_name.startswith("claude-"):
        return ChatAnthropic(model=model_name, temperature=temperature,
                             timeout=60, max_retries=3)
    return ChatOpenAI(model=model_name, temperature=temperature,
                      timeout=60, max_retries=3)


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

CORPUS_BASE = Path("data/DataFiles_v2")
INSTITUTE_DIRS = ["csir", "iacs", "ju", "ugc"]

OUT_DIR = Path("data/eval/v4")
OUT_DATASET = OUT_DIR / "idiom_qa_v4.json"
OUT_REJECTIONS = OUT_DIR / "rejection_log.jsonl"

SINGLE_DOC_TYPES = ["Factoid", "Conceptual", "Relational", "Multi-part", "Adversarial"]
CROSS_DOC_TYPES  = ["Temporal", "Aggregation", "CrossHop"]
ALL_TYPES = SINGLE_DOC_TYPES + CROSS_DOC_TYPES

SEMANTIC_DEDUP_THRESHOLD = 0.90


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class OM:
    """One Office Memorandum loaded from DataFiles_v2/."""
    source_id: str          # e.g. "csir_xyz_2024"
    institute: str          # "CSIR" | "IACS" | "JU" | "UGC"
    issue_date: str         # ISO date string or "UNKNOWN"
    body: str               # cleaned text content
    path: Path

    def payload(self) -> str:
        return (f"METADATA - Date: {self.issue_date} | Institute: {self.institute} | "
                f"Source-ID: {self.source_id}\n\nCONTENT:\n{self.body}")


@dataclass
class TemporalSeed:
    name: str
    label: str                          # human-readable topic, passed to the prompt
    match_institutes: List[str]         # institute codes: "csir" | "iacs" | "ju" | "ugc"
    match_keywords: List[str]           # ANY keyword in body matches
    min_group_size: int = 2


@dataclass
class AggregationSeed:
    key: str                            # short identifier (used in source field)
    kind: str                           # "scheme" | "position" | "department"
    label: str                          # human-readable topic, passed to the prompt
    match_institutes: List[str]
    match_keywords: List[str]
    min_group_size: int = 2


@dataclass
class CrossHopSeed:
    """A pair of in-corpus OMs that share an entity via DIFFERENT relationships.

    Discovered via Cypher over the v2-temporal KG. The seed only carries
    discovery metadata; downstream generation operates on OM text alone
    (the KG is discarded at generation time to break circularity).
    """
    entity_id:    str                   # shared entity (e.g. "Dr_Jhilmam_Ray")
    entity_label: str                   # node label ("Person", "Scheme_or_Project", ...)
    om_a_id:      str                   # source_om of OM A
    om_b_id:      str                   # source_om of OM B
    relation_a:   str                   # relationship type linking entity ↔ OM A
    relation_b:   str                   # relationship type linking entity ↔ OM B


# ── Temporal seeds (ported from generate_temporal_questions.py) ──────────────
TEMPORAL_SEEDS: List[TemporalSeed] = [
    TemporalSeed("jrf_fellowship", "JRF / SRF fellowship rates and rules",
                 match_institutes=["csir", "iacs", "ju"],
                 match_keywords=["jrf", "junior research fellow", "srf",
                                 "senior research fellow", "fellowship"]),
    TemporalSeed("dearness_allowance", "Dearness Allowance (DA) revisions",
                 match_institutes=["csir", "iacs", "ju", "ugc"],
                 match_keywords=["dearness allowance", "da payable",
                                 "rate of da", "revision of da"]),
    TemporalSeed("house_rent_allowance", "House Rent Allowance (HRA)",
                 match_institutes=["csir", "iacs", "ju", "ugc"],
                 match_keywords=["house rent allowance", "hra"]),
    TemporalSeed("pension", "Pension rules and revisions",
                 match_institutes=["csir", "iacs", "ju", "ugc"],
                 match_keywords=["pension", "pensioner", "family pension", "commutation"]),
    TemporalSeed("csir_leadership", "CSIR Director-General / Chairman appointments",
                 match_institutes=["csir"],
                 match_keywords=["director-general", "director general", "dg, csir",
                                 "chairman, csir", "appointed as"]),
    TemporalSeed("anrf_grants", "ANRF (Anusandhan National Research Foundation) grant directives",
                 match_institutes=["csir", "iacs", "ju", "ugc"],
                 match_keywords=["anrf", "anusandhan national research foundation"]),
    TemporalSeed("ugc_net", "UGC-NET examination and eligibility",
                 match_institutes=["ugc", "csir"],
                 match_keywords=["ugc-net", "ugc net", "national eligibility test"]),
]


# ── Aggregation seeds (ported from generate_aggregation_questions.py) ────────
AGGREGATION_SEEDS: List[AggregationSeed] = [
    # Schemes (single institute)
    AggregationSeed("iacs_serb", "scheme", "IACS positions under SERB-funded projects",
                    ["iacs"], ["serb", "science and engineering research board"]),
    AggregationSeed("iacs_anrf", "scheme", "IACS positions under ANRF-funded projects",
                    ["iacs"], ["anrf", "anusandhan national research foundation"]),
    AggregationSeed("iacs_dst", "scheme", "IACS positions under DST-funded projects",
                    ["iacs"], ["dst", "department of science and technology"]),
    AggregationSeed("iacs_dbt", "scheme", "IACS positions under DBT-funded projects",
                    ["iacs"], ["dbt", "department of biotechnology"]),
    AggregationSeed("ju_serb", "scheme", "Jadavpur University positions under SERB-funded projects",
                    ["ju"], ["serb"]),
    AggregationSeed("ju_anrf", "scheme", "Jadavpur University positions under ANRF-funded projects",
                    ["ju"], ["anrf", "anusandhan national research foundation"]),
    AggregationSeed("ju_aicte", "scheme", "Jadavpur University positions under AICTE schemes",
                    ["ju"], ["aicte"]),
    AggregationSeed("csir_serb", "scheme", "CSIR-affiliated positions under SERB-funded projects",
                    ["csir"], ["serb"]),
    # Cross-institute by scheme
    AggregationSeed("all_serb", "scheme", "All SERB-funded positions across CSIR / IACS / JU",
                    ["csir", "iacs", "ju"], ["serb"]),
    AggregationSeed("all_anrf", "scheme", "All ANRF-funded positions across CSIR / IACS / JU",
                    ["csir", "iacs", "ju"], ["anrf", "anusandhan national research foundation"]),
    # Position types (single institute)
    AggregationSeed("iacs_jrf", "position", "JRF / SRF positions at IACS",
                    ["iacs"], ["jrf", "junior research fellow", "srf", "senior research fellow"]),
    AggregationSeed("iacs_ra", "position", "Research Associate positions at IACS",
                    ["iacs"], ["research associate"]),
    AggregationSeed("iacs_pa", "position", "Project Associate / Project Fellow positions at IACS",
                    ["iacs"], ["project associate", "project fellow"]),
    AggregationSeed("iacs_faculty", "position", "Faculty positions (Asst/Assoc/Prof) at IACS",
                    ["iacs"], ["assistant professor", "associate professor", " professor"]),
    AggregationSeed("ju_jrf", "position", "JRF / SRF positions at Jadavpur University",
                    ["ju"], ["jrf", "junior research fellow", "srf", "senior research fellow"]),
    AggregationSeed("ju_faculty", "position", "Faculty positions at Jadavpur University",
                    ["ju"], ["assistant professor", "associate professor", " professor",
                            "guest faculty", "lecturer"]),
    # Cross-institute by position
    AggregationSeed("all_jrf", "position", "All JRF / SRF positions across CSIR / IACS / JU",
                    ["csir", "iacs", "ju"], ["jrf", "junior research fellow",
                                              "srf", "senior research fellow"]),
]


# ════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS (LLM structured outputs)
# ════════════════════════════════════════════════════════════════════════════

class FeasibilityVerdict(BaseModel):
    """One per-type verdict in the feasibility gate."""
    type: Literal["Factoid", "Conceptual", "Relational", "Multi-part", "Adversarial"]
    feasible: bool = Field(description="Whether the OM contains content sufficient for this type.")
    rationale: str = Field(description="One sentence on why feasible/infeasible, citing the OM.")
    evidence: Optional[str] = Field(
        default=None,
        description="Quoted span from the OM that justifies feasibility (null if infeasible).",
    )


class FeasibilityReport(BaseModel):
    """All 5 per-type verdicts for one OM."""
    verdicts: List[FeasibilityVerdict]


class GeneratedQuery(BaseModel):
    input: str = Field(description="The generated question, self-contained and globally unique.")
    question_type: Literal["Factoid", "Conceptual", "Relational", "Multi-part", "Adversarial"]
    expected_output: str = Field(
        description="Ground-truth answer from the OM. For Adversarial, state what is missing."
    )


class QueryBatch(BaseModel):
    queries: List[GeneratedQuery]


# ── Cross-doc schemas (ported) ───────────────────────────────────────────────
class TemporalQuery(BaseModel):
    input: str = Field(description="A natural temporal question a real user would ask.")
    temporal_kind: Literal["descriptive", "predictive"] = Field(
        description="'descriptive' = grounded in the timeline; 'predictive' = asks for a future projection."
    )
    expected_output: str = Field(
        description=("Ground-truth answer. For 'predictive', this MUST be a graceful refusal "
                     "stating the documents do not project future values.")
    )


class TemporalBatch(BaseModel):
    queries: List[TemporalQuery]


class AggregationQuery(BaseModel):
    input: str = Field(description="A natural cross-document aggregation question.")
    expected_output: str = Field(
        description=("A grounded answer that ENUMERATES the items found across the "
                     "provided documents. Cite specific department / position / project / "
                     "supervisor names where available.")
    )


class CrossHopQuery(BaseModel):
    """A cross-document multi-hop question seeded by KG entity co-occurrence
    but answered from OM text alone."""
    input: str = Field(
        description="A natural question whose answer requires synthesising facts "
                    "from BOTH OMs. The question should NOT explicitly say it needs "
                    "two documents; the multi-hop structure should be implicit."
    )
    expected_output: str = Field(
        description="The gold answer, a single coherent statement that combines one "
                    "fact from OM_A with one fact from OM_B. Must be text-derivable "
                    "from the two OMs together and only together."
    )
    evidence_om_a: str = Field(
        description="A short quoted excerpt (~30-200 chars) from OM_A supporting the "
                    "part of the answer that comes from OM_A."
    )
    evidence_om_b: str = Field(
        description="A short quoted excerpt (~30-200 chars) from OM_B supporting the "
                    "part of the answer that comes from OM_B."
    )


class CrossHopVerifyVerdict(BaseModel):
    """Verifier verdict for a CrossHop candidate. The verifier sees only the
    two OMs' text + the question (no KG, no gold answer) in its first pass,
    then compares its derived answer to the gold in a second pass."""
    verifier_answer: str = Field(
        description="The verifier's own answer to the question, derived from OM_A "
                    "and OM_B text only. Free-form."
    )
    answer_matches_gold: bool = Field(
        description="True iff the verifier's answer is SUBSTANTIVELY equivalent to "
                    "the gold answer (cosmetic differences are OK; factual differences are not)."
    )
    rationale: str = Field(description="One-sentence justification.")


# ════════════════════════════════════════════════════════════════════════════
# REJECTION LOGGER
# ════════════════════════════════════════════════════════════════════════════

class RejectionLogger:
    """Append-only JSONL log of every accept/reject decision across all stages.

    One row per decision; downstream RLHF training reads this directly.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on init so reruns produce a clean log
        self.fh = open(path, "w", encoding="utf-8")

    def log(
        self,
        *,
        stage: Literal["feasibility", "dedup"],
        source_om: str,
        intended_type: str,
        decision: Literal["accept", "reject"],
        reason_code: Optional[str] = None,
        candidate_question: Optional[str] = None,
        candidate_answer: Optional[str] = None,
        rationale: str = "",
        om_text_excerpt: Optional[str] = None,
    ):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "source_om": source_om,
            "intended_type": intended_type,
            "decision": decision,
            "reason_code": reason_code,
            "candidate_question": candidate_question,
            "candidate_answer": candidate_answer,
            "rationale": rationale,
            "om_text_excerpt": (om_text_excerpt[:500] if om_text_excerpt else None),
        }
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# ════════════════════════════════════════════════════════════════════════════
# DEDUPER (string + semantic) — ported from generate_eval_dataset.py
# ════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


class Deduper:
    """Greedy, order-dependent dedup.

    First-seen wins: a candidate that arrives later with identical normalised
    form or cosine >= threshold to any already-kept candidate is rejected.

    `check()` returns None if the candidate is unique, or a dict with the
    rejection reason and the earlier question it duplicates, so the rejection
    log can record the pair for post-hoc analysis.
    """

    def __init__(self, semantic: bool, threshold: float = SEMANTIC_DEDUP_THRESHOLD):
        self.seen_norm: dict[str, str] = {}   # normalised form → original kept text
        self.kept_texts: list[str] = []       # parallel to self._emb rows
        self.threshold = threshold
        self.semantic = semantic
        self.model = None
        self._emb = None
        if semantic:
            try:
                from sentence_transformers import SentenceTransformer
                print("[Dedup] Loading sentence-transformer for semantic dedup...")
                # bge-small-en-v1.5 matches the encoder used by the IDiOM-RAG
                # vector store (src/tools/local_tools.py), so retrieval-time and
                # generation-time semantic spaces stay consistent.
                self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            except Exception as e:
                print(f"[Dedup] Semantic dedup unavailable ({e}); using string dedup only.")
                self.semantic = False

    def check(self, text: str) -> Optional[dict]:
        """Return None if unique, else {'reason', 'duplicate_of', 'similarity'}."""
        norm = _normalize(text)
        if not norm:
            return {"reason": "dedup_string", "duplicate_of": "<empty>", "similarity": 1.0}
        if norm in self.seen_norm:
            return {"reason": "dedup_string",
                    "duplicate_of": self.seen_norm[norm],
                    "similarity": 1.0}
        if self.semantic and self.model is not None and self._emb is not None and len(self._emb):
            import numpy as np
            v = self.model.encode([text], normalize_embeddings=True)
            sims = (self._emb @ v[0])
            top = int(sims.argmax())
            if float(sims[top]) >= self.threshold:
                return {"reason": "dedup_semantic",
                        "duplicate_of": self.kept_texts[top],
                        "similarity": float(sims[top])}
        return None

    def add(self, text: str):
        import numpy as np
        self.seen_norm[_normalize(text)] = text
        self.kept_texts.append(text)
        if self.semantic and self.model is not None:
            v = self.model.encode([text], normalize_embeddings=True)
            self._emb = v if self._emb is None else np.vstack([self._emb, v])


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FEASIBILITY GATE (single-doc types only)
# ════════════════════════════════════════════════════════════════════════════

FEASIBILITY_SYSTEM_PROMPT = f"""You are auditing one Office Memorandum (OM) from an Indian institutional corpus (CSIR, IACS, JU, UGC) to decide which question types can be principally generated from it.

For EACH of the {len(SINGLE_DOC_TYPES)} types below, return a verdict on whether the OM contains the content prerequisites for that type. Be strict: if the prerequisites are not directly evidenced in the text, mark infeasible.

CRITERIA:

1. **Factoid** — feasible iff the OM contains at least one CONCRETE identifier that a real user would ask about: a specific date (issue date or event date), an advertisement/order/file number, a specific monetary amount (fee, stipend, allowance), a named designation, or a specific committee name.
   - Evidence required: a quoted span containing the concrete datapoint.

2. **Conceptual** — feasible iff the OM states at least one POLICY-like proposition: a rule, eligibility criterion, procedure, tenure clause, or substantive definition that goes beyond mere announcement. Single-line postponement/cancellation notices are INFEASIBLE.
   - Evidence required: a quoted clause containing modal language (shall, must, will, is entitled to, the criterion is, etc.) OR a definitional statement.

3. **Relational** — feasible iff the OM names at least one PERMANENT institutional relationship — person↔role, person↔organization, committee↔member, role↔reporting-line — that matches the KG ontology (relations: {", ".join(config.KG_ALLOWED_RELATIONS)}). Temporary postings and one-off appointments are INFEASIBLE.
   - Evidence required: a quoted span containing a named person + role/designation, or a permanent membership statement.

4. **Multi-part** — feasible iff the OM contains ≥2 DISTINCT facts that are located in **DIFFERENT, WELL-SEPARATED sections** of the document — e.g. different paragraphs, different headed sections, header vs body, or body vs table/signature block. Two adjacent sentences in the same paragraph are NOT sufficient: their co-location means any reasonable retrieval system would return them together, so the question would test in-context synthesis rather than multi-hop retrieval. Single-fact notices (one announcement, one event time, one cancellation) are INFEASIBLE.
   - Evidence required: a brief description of the two facts AND the distinct sections they live in (e.g., "Para 1 names the new chair; Para 4 specifies the chair's tenure dates" — NOT "the first paragraph names the chair and their tenure"). If the two facts cannot be cleanly attributed to two different sections, mark infeasible.

5. **Adversarial** — feasible iff a plausible out-of-scope question can be anchored on this OM's topic *and* the question falls into one of the five CATEGORICALLY-OOS classes below. These classes are defined so the answer cannot exist anywhere else in the 116-OM corpus either — not just this document.

   Allowed Adversarial categories (the candidate question MUST fall into one):
     a. **Opinion / sentiment** — "What do X think / feel about Y?"
        (OMs are directives, never survey or sentiment data.)
     b. **Future prediction / speculation** — "Will the rate be raised next year?"
        (OMs document past actions; no OM speculates about future policy.)
     c. **Personal anecdote / narrative** — "Describe how the new chair felt when appointed."
        (OMs contain no narrative prose.)
     d. **Out-of-domain institution** — "What is IIT-KGP's HRA policy?" (corpus
        only covers CSIR / IACS / JU / UGC, so any other Indian institution is OOS.)
     e. **Out-of-domain topic** — questions about weather, sports, medical advice,
        product recommendations, recipes, etc. The corpus is administrative;
        these topics are trivially OOS.

   Avoid candidate questions that ask for a SPECIFIC fact (a number, date,
   designation, scheme name) that *this* OM doesn't state, because such facts
   are likely answered in some OTHER OM in the corpus. That's not OOS — it's
   intra-corpus leakage. The five categories above are immune to that
   leakage because the corpus by construction does not contain opinions,
   predictions, narratives, other-institution facts, or off-topic content.

   - Evidence required: name the OM's domain topic + which of (a)-(e) the
     candidate falls into (e.g., "OM is about JRF stipend revisions; OOS
     angle (a): 'What do JRF candidates think about the stipend?'").
   - Mark feasible=false ONLY if the OM is so generic that NO category (a)-(e)
     can anchor on it (very rare).

RULES:
- Return exactly 5 verdicts, one per type, in the order above.
- If feasible=false, evidence must be null.
- If feasible=true, evidence must be a literal quoted span from the OM (or for Adversarial, a topic descriptor + example OOS angle).
- Be conservative: when in doubt, mark infeasible. Downstream cost of contrived questions is higher than the cost of dropping a borderline type.
"""


def get_feasibility_chain(llm):
    structured = llm.with_structured_output(FeasibilityReport)
    prompt = ChatPromptTemplate.from_messages([
        ("system", FEASIBILITY_SYSTEM_PROMPT),
        ("human", "OM TO AUDIT:\n\n{om_payload}"),
    ])
    return prompt | structured


def run_feasibility(om: OM, chain, logger: RejectionLogger) -> Dict[str, FeasibilityVerdict]:
    """Returns {type_name: FeasibilityVerdict} for all 5 single-doc types.

    Logs every infeasible verdict as a 'feasibility' rejection.
    """
    report: FeasibilityReport = chain.invoke({"om_payload": om.payload()})
    by_type = {v.type: v for v in report.verdicts}
    # Validate completeness — LLM should return all 5
    for t in SINGLE_DOC_TYPES:
        if t not in by_type:
            # Safety: missing verdict = treat as infeasible
            by_type[t] = FeasibilityVerdict(
                type=t, feasible=False,
                rationale="LLM did not return a verdict for this type.",
                evidence=None,
            )
        if not by_type[t].feasible:
            logger.log(
                stage="feasibility",
                source_om=om.source_id,
                intended_type=t,
                decision="reject",
                reason_code="infeasible_type",
                rationale=by_type[t].rationale,
                om_text_excerpt=om.body,
            )
    return by_type


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1.5 — ADVERSARIAL CORPUS-WIDE OOS VERIFICATION
# ════════════════════════════════════════════════════════════════════════════
# An Adversarial candidate generated from OM-X with the gold label "refuse"
# is only valid if NO OTHER OM in the corpus answers the question either.
# This stage retrieves the top-3 most BM25-similar OMs across the WHOLE
# corpus and asks the cross-family verifier LLM (Claude Haiku, by default)
# whether any of them concretely answers the candidate. If yes, the
# candidate is rejected as `oos_leak` — its gold-refuse label would
# wrongly penalise a system that correctly retrieved the leaking OM.
#
# Non-Adversarial candidates pass through this stage untouched.

class OOSVerifyVerdict(BaseModel):
    answerable: bool = Field(description="True iff a SPECIFIC factual answer to the question is present in the retrieved OMs.")
    leak_source: Optional[str] = Field(default=None, description="If answerable, the source_id of the OM that answers the question; else null.")
    rationale: str = Field(description="One sentence on why answerable or not.")


OOS_VERIFY_SYSTEM_PROMPT = """You are verifying whether a candidate Adversarial (out-of-scope) question genuinely has NO answer anywhere in our 116-OM corpus of Indian institutional Office Memoranda (CSIR, IACS, JU, UGC).

You will see the candidate question, its expected refusal-style gold output, and the top-3 most relevant OMs from the corpus retrieved via BM25.

Decide STRICTLY:
- answerable=true iff at least one of the retrieved OMs contains a CONCRETE FACTUAL ANSWER to the question (a specific date, amount, name, rule, procedure, etc.).
- answerable=false if no OM contains a specific answer, even if some OMs touch on the topic vaguely.

A vague topical mention does NOT count as an answer. Be conservative on "answerable=true" — false positives wrongly reject genuine Adversarial questions.

When answerable=true, set leak_source to the source_id of the OM that contains the answer."""


OOS_VERIFY_USER_PROMPT = """CANDIDATE QUESTION (intended to be out-of-scope):
{question}

CANDIDATE GOLD OUTPUT (typically a refusal explaining what is missing):
{expected_output}

TOP-3 OMs from the corpus (most BM25-similar to the question):
{chunks_block}

Respond with the OOSVerifyVerdict schema."""


def get_oos_verifier_chain(llm):
    structured = llm.with_structured_output(OOSVerifyVerdict)
    prompt = ChatPromptTemplate.from_messages([
        ("system", OOS_VERIFY_SYSTEM_PROMPT),
        ("user",   OOS_VERIFY_USER_PROMPT),
    ])
    return prompt | structured


def build_oos_bm25(corpus: List[OM], k: int = 3) -> BM25Retriever:
    """In-memory BM25 retriever over the full corpus, used only by the
    Adversarial OOS-verification gate. Uses whole OM bodies as documents
    (no chunking) — the corpus is small (~116 docs, ~310 KB) and OOS
    verification needs document-level context, not chunk-level fragments."""
    docs = [
        Document(
            page_content=om.body,
            metadata={"source": om.source_id, "institute": om.institute,
                      "issue_date": om.issue_date},
        )
        for om in corpus
    ]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


def verify_adversarial_oos(
    candidate,
    om: OM,
    bm25: BM25Retriever,
    verifier_chain,
    logger: RejectionLogger,
) -> bool:
    """Return True if the candidate is genuinely OOS w.r.t. the full corpus.

    Non-Adversarial candidates pass through unchanged (return True).
    Adversarials are rejected (and logged) iff any retrieved OM answers them.
    Fails open: if the verifier itself errors, the candidate passes through
    and the error is logged for post-hoc audit.
    """
    if candidate.question_type != "Adversarial":
        return True

    retrieved = bm25.invoke(candidate.input)
    if not retrieved:
        return True                                  # nothing similar — definitely OOS

    chunks_block = "\n\n---\n\n".join(
        f"[OM #{i}: source_id={d.metadata.get('source', '?')} "
        f"institute={d.metadata.get('institute', '?')} "
        f"date={d.metadata.get('issue_date', '?')}]\n"
        f"{d.page_content[:2000]}"                   # cap each at 2K chars to bound prompt size
        for i, d in enumerate(retrieved)
    )

    try:
        verdict: OOSVerifyVerdict = verifier_chain.invoke({
            "question":        candidate.input,
            "expected_output": candidate.expected_output,
            "chunks_block":    chunks_block,
        })
    except Exception as e:
        logger.log(
            stage="oos_verify",
            source_om=om.source_id,
            intended_type="Adversarial",
            decision="accept",
            reason_code="oos_verify_error",
            rationale=f"Verifier exception: {e}; passing candidate through",
            candidate_question=candidate.input,
            candidate_answer=candidate.expected_output,
        )
        return True

    if verdict.answerable:
        logger.log(
            stage="oos_verify",
            source_om=om.source_id,
            intended_type="Adversarial",
            decision="reject",
            reason_code="oos_leak",
            rationale=f"Answer found in {verdict.leak_source or '?'}: {verdict.rationale}",
            candidate_question=candidate.input,
            candidate_answer=candidate.expected_output,
        )
        return False

    # Genuine OOS — let it through to the dedup stage.
    return True


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — GENERATION (per feasible slot)
# ════════════════════════════════════════════════════════════════════════════

_TYPE_DESCRIPTIONS = {
    "Factoid": "Factoid — a specific date, ID, number, fee, or name.",
    "Conceptual": "Conceptual — a rule, policy, tenure, eligibility, or procedure.",
    "Relational": (
        "Relational — a PERMANENT hierarchy / reporting line / institutional structure "
        "that tests one of these relationships: [{rels}]. "
        "Not temporary projects or postings."
    ),
    "Adversarial": (
        "Adversarial — a plausible out-of-scope question that, by category, has "
        "NO answer anywhere in our 116-OM corpus (not just this OM). The "
        "candidate MUST fall into ONE of these categorically-OOS classes:\n"
        "  (a) Opinion / sentiment ('What do X think about Y')\n"
        "  (b) Future prediction / speculation ('Will the rate be raised next year')\n"
        "  (c) Personal anecdote / narrative ('Describe how X felt')\n"
        "  (d) Out-of-domain institution (any Indian institution OTHER than "
        "CSIR / IACS / JU / UGC, e.g. 'What is IIT-KGP's HRA policy?')\n"
        "  (e) Out-of-domain topic (weather, sports, medical advice, recipes, etc.)\n"
        "Do NOT ask for a specific fact that THIS OM omits — that fact may be "
        "in another OM in the corpus, which would make the gold-refuse label "
        "wrong. Anchor the question on THIS OM's topic but stay in (a)-(e). "
        "expected_output must state what type of answer is missing and why "
        "the corpus by construction cannot provide it."
    ),
    "Multi-part": (
        "Multi-part — requires synthesizing facts from at least TWO DISTINCT, "
        "WELL-SEPARATED sections of this document (different paragraphs / "
        "different headed sections / header vs body / body vs table or "
        "signature block). Do NOT generate questions whose two facts come "
        "from adjacent sentences in the same paragraph — those test in-context "
        "synthesis, not multi-hop retrieval, and any reasonable retrieval "
        "system would return the two facts together. The two facts must be "
        "far enough apart in the document that retrieving one would not "
        "automatically surface the other."
    ),
}


def get_generation_chain(llm, k_per_type: int, feasible_types: List[str]):
    """Chain that produces k_per_type candidates for each feasible type."""
    allowed_rels = ", ".join(config.KG_ALLOWED_RELATIONS)
    type_block = "\n".join(
        f"{i}. {_TYPE_DESCRIPTIONS[t].format(rels=allowed_rels)}"
        for i, t in enumerate(feasible_types, start=1)
    )
    system_prompt = (
        f"You are an expert AI engineer building a realistic evaluation set for an "
        f"institutional RAG system used at CSIR, IACS, JU, and UGC.\n\n"
        f"Read the provided document (metadata + body) and generate exactly "
        f"{k_per_type} questions for EACH of the {len(feasible_types)} requested types "
        f"below (total {len(feasible_types) * k_per_type}).\n\n"
        f"GLOBAL-UNIQUENESS RULES (most important):\n"
        f"- Every question MUST be self-contained and globally unique across the WHOLE corpus.\n"
        f"- Anchor each question to the SUBJECT and ENTITIES a real user would actually remember:\n"
        f"  the specific position/fellowship title + department/school + institute, the committee /\n"
        f"  scheme / policy name, the named person, or the specific issue the document addresses.\n"
        f"- Use the issue DATE or ORDER/OM NUMBER ONLY as a last resort — when the subject and\n"
        f"  entities genuinely cannot distinguish this document from a similar one.\n"
        f"- NEVER write a generic question that could apply to multiple documents.\n\n"
        f"REALISM RULES:\n"
        f"- Frame each question the way someone who remembers the ISSUE but not the paperwork's\n"
        f"  metadata would ask it. People recall WHAT a notice was about, not its issue date or\n"
        f"  order number.\n"
        f"- Do NOT append a date as a disambiguator (\"dated ...\", \"according to the notice\n"
        f"  dated ...\"). If a date is genuinely unavoidable, refer to it the natural way a\n"
        f"  person would (\"the 2022 DA revision\"), not as formal metadata.\n"
        f"- NEVER use meta-phrases such as \"as stated in the office memorandum\", \"as mentioned\n"
        f"  in the document\", \"according to the order/text/excerpt\", or \"in the document\".\n"
        f"- Use the actual names of committees, policies, institutes, positions, and people.\n\n"
        f"TYPES:\n{type_block}\n"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the questions from this document:\n\n{om_payload}"),
    ])
    return prompt | llm.with_structured_output(QueryBatch)


def run_generation(om: OM, feasible_types: List[str], k: int, llm) -> tuple[List[GeneratedQuery], dict]:
    """Returns (kept_candidates, debug_info).
    debug_info captures cases where the LLM returned an empty batch or
    candidates with types we didn't request, so the rejection log can
    audit silent generation failures."""
    if not feasible_types:
        return [], {"raw_count": 0, "kept_count": 0}
    chain = get_generation_chain(llm, k, feasible_types)
    batch: QueryBatch = chain.invoke({"om_payload": om.payload()})
    raw = batch.queries
    kept = [q for q in raw if q.question_type in feasible_types]
    return kept, {
        "raw_count": len(raw),
        "kept_count": len(kept),
        "off_types": [q.question_type for q in raw if q.question_type not in feasible_types],
    }


# ── Cross-doc prompts (ported) ───────────────────────────────────────────────
TEMPORAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are building TEMPORAL evaluation questions for a RAG system over Indian\n"
     "government / academic Office Memoranda (CSIR, IACS, JU, UGC).\n\n"
     "A temporal question is one whose answer depends on synthesising information\n"
     "across MULTIPLE documents over time — a single OM is never enough.\n\n"
     "You will be given a chronologically-sorted timeline of OMs all related to one\n"
     "topic: **{seed_label}**.\n\n"
     "Generate exactly {n_questions} questions:\n\n"
     "  - {n_descriptive} DESCRIPTIVE question(s):\n"
     "      Ask about the trend, history, current value, or pattern of change.\n"
     "      expected_output: a concise, accurate answer grounded ONLY in the\n"
     "      timeline above. Cite specific numbers/dates from the timeline when\n"
     "      relevant. Do NOT invent.\n\n"
     "  - {n_predictive} PREDICTIVE question(s):\n"
     "      Ask the system to project a FUTURE value, date, or trend the documents\n"
     "      cannot answer.\n"
     "      expected_output: a graceful refusal that names what is missing.\n"
     "      Template (adapt naturally — do not copy verbatim):\n"
     "        \"The documents describe past values and revisions but do not project\n"
     "         future rates. Based on the timeline I can describe the historical\n"
     "         pattern (…), but I cannot reliably predict the next revision.\"\n\n"
     "PHRASING RULES:\n"
     "  - Frame each question the way a real user who remembers the TOPIC (not the\n"
     "    paperwork) would ask. People ask \"what is the current JRF fellowship?\",\n"
     "    not \"according to OM dated 8 Jan 2025, what is…\".\n"
     "  - Do NOT append \"as per the document\", \"according to the OM dated …\".\n"
     "  - The descriptive question must REQUIRE cross-document reasoning."),
    ("human", "TIMELINE for topic **{seed_label}** ({n_docs} documents):\n\n{timeline}"),
])


AGGREGATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are building CROSS-DOCUMENT AGGREGATION evaluation questions for a RAG\n"
     "system over Indian institutional Office Memoranda (CSIR, IACS, JU, UGC).\n\n"
     "An aggregation question requires synthesising facts ACROSS multiple documents\n"
     "IN PARALLEL — listing all items that share a property, enumerating positions\n"
     "across departments, comparing structures across institutes, etc. A single OM\n"
     "is never enough.\n\n"
     "You will be given a STACK of OMs all related to one topic:\n"
     "  **{seed_label}**  ({n_docs} documents)\n\n"
     "Generate EXACTLY ONE aggregation question with these properties:\n\n"
     "1. The question must REQUIRE the system to enumerate, list, or aggregate\n"
     "   across the documents — NOT pick out a single fact from one OM.\n"
     "2. Phrase it the way a user familiar with the institutional landscape would\n"
     "   actually ask. People say \"which departments at IACS are running SERB\n"
     "   projects?\" — they do NOT say \"list all OMs that mention SERB\".\n"
     "3. Do NOT name a specific document, date, or OM number in the question.\n"
     "4. The expected_output must ENUMERATE the items, citing department / position\n"
     "   / project / supervisor names from the documents. Use a bullet-style list.\n"
     "5. Do NOT invent items not present in the documents."),
    ("human", "STACK ({n_docs} documents) for topic **{seed_label}**:\n\n{group_payload}"),
])


# Quality scoring (Correctness/Fluency/Completeness, type-match, self-disambig)
# is intentionally NOT done here — that's the downstream annotator's job
# (Claude Sonnet 4.6 + 2 humans). This script's output is the candidate set
# that survived construction-time gating; the annotator decides what's good.


# ════════════════════════════════════════════════════════════════════════════
# CROSS-DOC PIPELINES
# ════════════════════════════════════════════════════════════════════════════

CROSS_DOC_BODY_CHARS = 1500   # truncate each OM in the timeline/stack payload


def _match_keywords_any(om: OM, keywords: List[str]) -> bool:
    text = (om.body or "").lower()
    return any(kw.lower() in text for kw in keywords)


def _format_doc_block(om: OM) -> str:
    body = (om.body or "")[:CROSS_DOC_BODY_CHARS]
    return (
        "━" * 60 + "\n"
        f"DATE: {om.issue_date} | INSTITUTE: {om.institute} | SOURCE: {om.source_id}\n"
        + "━" * 60 + "\n"
        + body
    )


def match_temporal_seed(seed: TemporalSeed, corpus: List[OM]) -> List[OM]:
    """OMs matching the temporal seed's institutes + keywords. Sorted by issue_date.

    Undated OMs are dropped (they corrupt the timeline ordering).
    """
    institutes = {i.lower() for i in seed.match_institutes}
    group = [
        om for om in corpus
        if om.institute.lower() in institutes
        and om.issue_date and not om.issue_date.lower().startswith("unknown")
        and _match_keywords_any(om, seed.match_keywords)
    ]
    group.sort(key=lambda om: om.issue_date)
    return group


def match_aggregation_seed(seed: AggregationSeed, corpus: List[OM]) -> List[OM]:
    """OMs matching the aggregation seed's institutes + keywords."""
    institutes = {i.lower() for i in seed.match_institutes}
    return [
        om for om in corpus
        if om.institute.lower() in institutes
        and _match_keywords_any(om, seed.match_keywords)
    ]


def _accept_or_reject_cross_doc(
    *, candidate_text: str, candidate_answer: str, question_type: str,
    source_id: str, payload: str, extra_fields: dict,
    dedup: Deduper, logger: RejectionLogger,
) -> Optional[dict]:
    """Apply dedup to a cross-doc candidate; return accepted row or None."""
    dedup_match = dedup.check(candidate_text)
    if dedup_match is not None:
        logger.log(
            stage="dedup",
            source_om=source_id,
            intended_type=question_type,
            decision="reject",
            reason_code=dedup_match["reason"],
            candidate_question=candidate_text,
            candidate_answer=candidate_answer,
            rationale=(f"Duplicate of earlier question (sim={dedup_match['similarity']:.3f}): "
                       f"'{dedup_match['duplicate_of'][:120]}'"),
        )
        return None
    dedup.add(candidate_text)
    logger.log(
        stage="dedup",
        source_om=source_id,
        intended_type=question_type,
        decision="accept",
        candidate_question=candidate_text,
        candidate_answer=candidate_answer,
    )
    row = {
        "source":          source_id,
        "input":           candidate_text,
        "question_type":   question_type,
        "expected_output": candidate_answer,
        "context_used":    payload,
    }
    row.update(extra_fields)
    return row


def process_temporal_seed(
    seed: TemporalSeed, corpus: List[OM], llm,
    per_seed: int, dedup: Deduper, logger: RejectionLogger,
) -> List[dict]:
    """Run temporal generation on one seed; return accepted rows."""
    group = match_temporal_seed(seed, corpus)
    if len(group) < seed.min_group_size:
        print(f"  [temporal/{seed.name:<20}] {len(group):>2} docs — skipping (need ≥{seed.min_group_size})")
        return []

    timeline = "\n\n".join(_format_doc_block(om) for om in group)
    n_desc = per_seed // 2 + per_seed % 2  # ceil(per_seed/2)
    n_pred = per_seed - n_desc

    chain = TEMPORAL_PROMPT | llm.with_structured_output(TemporalBatch)
    try:
        batch: TemporalBatch = chain.invoke({
            "seed_label":    seed.label,
            "n_questions":   per_seed,
            "n_descriptive": n_desc,
            "n_predictive":  n_pred,
            "n_docs":        len(group),
            "timeline":      timeline,
        })
    except Exception as e:
        print(f"  [temporal/{seed.name:<20}] FAILED: {e}")
        return []

    accepted: List[dict] = []
    seed_docs = [om.source_id for om in group]
    for q in batch.queries:
        row = _accept_or_reject_cross_doc(
            candidate_text=q.input.strip(),
            candidate_answer=q.expected_output.strip(),
            question_type="Temporal",
            source_id=f"temporal:{seed.name}",
            payload=timeline,
            extra_fields={"temporal_kind": q.temporal_kind, "seed_docs": seed_docs},
            dedup=dedup, logger=logger,
        )
        if row is not None:
            accepted.append(row)
    print(f"  [temporal/{seed.name:<20}] {len(group):>2} docs  +{len(accepted)}")
    return accepted


def process_aggregation_seed(
    seed: AggregationSeed, corpus: List[OM], llm,
    dedup: Deduper, logger: RejectionLogger,
) -> List[dict]:
    """Run aggregation generation on one seed; return accepted rows (at most 1)."""
    group = match_aggregation_seed(seed, corpus)
    if len(group) < seed.min_group_size:
        print(f"  [aggregation/{seed.key:<18}] {len(group):>2} docs — skipping (need ≥{seed.min_group_size})")
        return []

    payload = "\n\n".join(_format_doc_block(om) for om in group)
    chain = AGGREGATION_PROMPT | llm.with_structured_output(AggregationQuery)
    try:
        result: AggregationQuery = chain.invoke({
            "seed_label":    seed.label,
            "n_docs":        len(group),
            "group_payload": payload,
        })
    except Exception as e:
        print(f"  [aggregation/{seed.key:<18}] FAILED: {e}")
        return []

    row = _accept_or_reject_cross_doc(
        candidate_text=result.input.strip(),
        candidate_answer=result.expected_output.strip(),
        question_type="Aggregation",
        source_id=f"aggregation:{seed.key}",
        payload=payload,
        extra_fields={
            "aggregation_kind": seed.kind,
            "seed_docs": [om.source_id for om in group],
            "seed_institutes": seed.match_institutes,
        },
        dedup=dedup, logger=logger,
    )
    accepted = [row] if row is not None else []
    print(f"  [aggregation/{seed.key:<18}] {len(group):>2} docs  +{len(accepted)}")
    return accepted


# ════════════════════════════════════════════════════════════════════════════
# CROSS-DOC MULTI-HOP (KG-discovered, text-generated, cross-family-verified)
# ════════════════════════════════════════════════════════════════════════════
# Pipeline:
#   1. Discovery — Cypher over the temporally-aware KG (v2) finds pairs of
#      in-corpus OMs that share an entity via DIFFERENT relationship types.
#   2. Generation — Generator LLM (gpt-4.1-mini) sees only the two OMs' text
#      (KG discarded). Returns question + gold + two evidence excerpts.
#   3. Verification — Independent cross-family LLM (claude-haiku-4-5) sees
#      only the two OMs + question, derives its own answer, then compares
#      to gold. Mismatch → reject.
#   4. Dedup — same string + semantic gate as the rest of the pipeline.
#
# This breaks the KG-circularity concern: the KG is a seed discoverer, never
# an oracle. All reasoning happens in OM text.

CROSSHOP_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are building cross-document multi-hop questions for a benchmark of Indian institutional Office Memoranda (CSIR, IACS, JU, UGC).

You will be given TWO different OMs (OM_A and OM_B) that both reference the same entity (a person, scheme, committee, etc.) but in DIFFERENT relationship roles.

Your job is to write a question whose answer CANNOT be derived from either OM alone — it requires reading BOTH.

STRICT RULES:
1. The question must mention the shared entity (or a clear paraphrase) only once. Do not telegraph that the answer needs two documents.
2. The expected answer must be a single, coherent statement that combines one fact from OM_A with one fact from OM_B.
3. You MUST provide one short quoted evidence excerpt (~30-200 chars) from EACH OM. The OM_A excerpt must contain the part of the answer that comes from OM_A; the OM_B excerpt must contain the part that comes from OM_B.
4. If you cannot construct such a question from these two OMs (e.g. the shared entity reference is too superficial, or one OM doesn't actually contribute substantive content), respond with input="SKIP" and evidence fields empty. DO NOT fabricate links.
5. Stay grounded in the OM text — never invent dates, names, amounts, or relationships not present in either OM."""),
    ("user", """SHARED ENTITY: {entity_id} (a {entity_label})

OM_A (source_om={om_a_id}):
{om_a_body}

OM_B (source_om={om_b_id}):
{om_b_body}

Generate one CrossHopQuery whose answer requires both documents."""),
])


CROSSHOP_VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are verifying whether a candidate cross-document multi-hop question is genuinely answerable from the source OMs alone.

You will see the question, the text of OM_A and OM_B, and the candidate's gold answer.

Your task:
1. First, mentally try to answer the question using ONLY the text of the two OMs (do not look at the gold yet).
2. Write your derived answer in `verifier_answer`.
3. Then compare your derived answer to the gold answer. Set answer_matches_gold=true iff the two answers are SUBSTANTIVELY equivalent (cosmetic differences in phrasing, formatting, or paraphrase are OK; factual differences — different dates, different names, different amounts, missing facts — are NOT OK).

Be strict on step 3: false positives here weaken the benchmark by letting through ill-grounded questions."""),
    ("user", """OM_A:
{om_a_body}

OM_B:
{om_b_body}

QUESTION:
{question}

GOLD ANSWER (for the matching step only):
{gold_answer}

Respond with verifier_answer + answer_matches_gold + a one-sentence rationale."""),
])


def get_crosshop_generation_chain(llm):
    return CROSSHOP_GENERATION_PROMPT | llm.with_structured_output(CrossHopQuery)


def get_crosshop_verification_chain(llm):
    return CROSSHOP_VERIFY_PROMPT | llm.with_structured_output(CrossHopVerifyVerdict)


# Cypher queries — both object-side and subject-side patterns. We OR them
# into one seed list, dedup on (entity_id, om_a_id, om_b_id) afterward.
#
# We require type(r_a) <> type(r_b) so the chain is genuinely multi-hop
# (different relationship types) rather than aggregation (same relationship
# repeated across docs). We also require both r.superseded = "0" so the
# chain reflects current state.
# Two filters keep seeds away from trivial provenance/institutional chains:
#   - exclude ISSUED_BY: it's "OM was issued by X", pure provenance metadata
#     that connects every OM to its issuer. Pairing it with anything else
#     produces institutional co-occurrence, not multi-hop reasoning.
#   - exclude Organization_or_Department entities: institute names (IACS,
#     CSIR, JU, UGC) appear in every OM and produce aggregation-shaped
#     questions, not entity-chain multi-hop.
_CROSSHOP_CYPHER_OBJECT = """
MATCH (entity)<-[r_a]-()
MATCH (entity)<-[r_b]-()
WHERE r_a.source_om <> r_b.source_om
  AND type(r_a) <> type(r_b)
  AND type(r_a) <> "ISSUED_BY"
  AND type(r_b) <> "ISSUED_BY"
  AND NOT "Organization_or_Department" IN labels(entity)
  AND r_a.superseded = "0"
  AND r_b.superseded = "0"
  AND r_a.issue_date IS NOT NULL
  AND r_b.issue_date IS NOT NULL
RETURN DISTINCT
  entity.id          AS entity_id,
  labels(entity)[0]  AS entity_label,
  r_a.source_om      AS om_a_id,
  r_b.source_om      AS om_b_id,
  type(r_a)          AS relation_a,
  type(r_b)          AS relation_b
LIMIT $limit
"""

_CROSSHOP_CYPHER_SUBJECT = """
MATCH (entity)-[r_a]->()
MATCH (entity)-[r_b]->()
WHERE r_a.source_om <> r_b.source_om
  AND type(r_a) <> type(r_b)
  AND type(r_a) <> "ISSUED_BY"
  AND type(r_b) <> "ISSUED_BY"
  AND NOT "Organization_or_Department" IN labels(entity)
  AND r_a.superseded = "0"
  AND r_b.superseded = "0"
  AND r_a.issue_date IS NOT NULL
  AND r_b.issue_date IS NOT NULL
RETURN DISTINCT
  entity.id          AS entity_id,
  labels(entity)[0]  AS entity_label,
  r_a.source_om      AS om_a_id,
  r_b.source_om      AS om_b_id,
  type(r_a)          AS relation_a,
  type(r_b)          AS relation_b
LIMIT $limit
"""


def _cap_per_entity(seeds: List[CrossHopSeed],
                    max_per_entity: int) -> List[CrossHopSeed]:
    """Keep at most max_per_entity seeds per entity_id.

    Preserves discovery order within each entity group. Used by
    discover_crosshop_seeds to prevent over-connected entities (e.g. a
    popular scheme like DST/WISE-PhD that is referenced in many OMs)
    from dominating the seed pool.
    """
    counts: dict[str, int] = {}
    kept: List[CrossHopSeed] = []
    for s in seeds:
        if counts.get(s.entity_id, 0) < max_per_entity:
            kept.append(s)
            counts[s.entity_id] = counts.get(s.entity_id, 0) + 1
    return kept


def discover_crosshop_seeds(graph: Neo4jGraph,
                             corpus: List[OM],
                             max_seeds: int = 100,
                             max_per_entity: int = 3) -> List[CrossHopSeed]:
    """Query the temporally-aware KG for cross-doc entity chains.

    Returns CrossHopSeeds where:
      - Both OMs are in our 116-OM corpus (filtered against source_id set)
      - The two relationships are different types
      - The entity is shared (object-side OR subject-side)
      - At most `max_per_entity` seeds are kept per shared entity, so a
        single over-connected scheme/person can't dominate the mix

    We over-collect from Cypher (max_seeds * max_per_entity rows) so the
    per-entity cap has enough material to spread across many entities,
    then trim to max_seeds at the end.
    """
    corpus_ids = {om.source_id for om in corpus}
    seen: set[tuple[str, str, str]] = set()
    raw_seeds: List[CrossHopSeed] = []

    # Cypher LIMIT: over-collect so the cap has diverse material to choose from
    cypher_limit = max(max_seeds * max_per_entity, max_seeds)
    for cypher in (_CROSSHOP_CYPHER_OBJECT, _CROSSHOP_CYPHER_SUBJECT):
        try:
            rows = graph.query(cypher, params={"limit": cypher_limit})
        except Exception as e:
            print(f"  [crosshop discovery] Cypher query failed: {e}")
            continue
        for r in rows:
            om_a_id = r.get("om_a_id")
            om_b_id = r.get("om_b_id")
            if om_a_id not in corpus_ids or om_b_id not in corpus_ids:
                continue                          # external / referenced docs, skip
            # Normalise OM pair order so (A,B) and (B,A) collapse to one seed
            ordered = tuple(sorted([om_a_id, om_b_id]))
            key = (r.get("entity_id"), ordered[0], ordered[1])
            if key in seen:
                continue
            seen.add(key)
            raw_seeds.append(CrossHopSeed(
                entity_id    = r.get("entity_id", ""),
                entity_label = r.get("entity_label", "Entity"),
                om_a_id      = ordered[0],
                om_b_id      = ordered[1],
                relation_a   = r.get("relation_a", ""),
                relation_b   = r.get("relation_b", ""),
            ))

    # Spread coverage across entities, then trim to budget
    capped = _cap_per_entity(raw_seeds, max_per_entity)
    return capped[:max_seeds]


def process_crosshop_seed(
    seed: CrossHopSeed,
    corpus: List[OM],
    gen_chain,
    verify_chain,
    dedup: Deduper,
    logger: RejectionLogger,
    body_cap: int = 8000,
) -> List[dict]:
    """Generate → verify → dedup for one CrossHop seed.

    Fails open on verifier exceptions (lets the candidate through with a
    logged error) — mirrors the pattern in verify_adversarial_oos.
    """
    by_id = {om.source_id: om for om in corpus}
    om_a = by_id.get(seed.om_a_id)
    om_b = by_id.get(seed.om_b_id)
    if om_a is None or om_b is None:
        return []                                  # safety net — shouldn't happen after discovery

    source_id = f"crosshop:{seed.entity_id}|{seed.om_a_id}~{seed.om_b_id}"

    # ── Stage 2: Generation (text only) ─────────────────────────
    try:
        cand: CrossHopQuery = gen_chain.invoke({
            "entity_id":    seed.entity_id,
            "entity_label": seed.entity_label,
            "om_a_id":      seed.om_a_id,
            "om_a_body":    om_a.body[:body_cap],
            "om_b_id":      seed.om_b_id,
            "om_b_body":    om_b.body[:body_cap],
        })
    except Exception as e:
        logger.log(
            stage="crosshop_gen", source_om=source_id, intended_type="CrossHop",
            decision="reject", reason_code="crosshop_gen_error", rationale=str(e),
        )
        return []

    # Generator may legitimately decline if the seed is too thin
    if cand.input.strip().upper() == "SKIP" or not cand.input.strip():
        logger.log(
            stage="crosshop_gen", source_om=source_id, intended_type="CrossHop",
            decision="reject", reason_code="crosshop_gen_declined",
            rationale="Generator returned SKIP — seed was too superficial",
        )
        return []

    # ── Stage 3: Independent verification (cross-family, text only) ────
    try:
        verdict: CrossHopVerifyVerdict = verify_chain.invoke({
            "om_a_body":   om_a.body[:body_cap],
            "om_b_body":   om_b.body[:body_cap],
            "question":    cand.input,
            "gold_answer": cand.expected_output,
        })
    except Exception as e:
        # Fail open — log and let through, matching OOS verifier behaviour
        logger.log(
            stage="crosshop_verify", source_om=source_id, intended_type="CrossHop",
            decision="accept", reason_code="crosshop_verify_error",
            rationale=f"Verifier exception: {e}; passing candidate through",
            candidate_question=cand.input, candidate_answer=cand.expected_output,
        )
    else:
        if not verdict.answer_matches_gold:
            logger.log(
                stage="crosshop_verify", source_om=source_id, intended_type="CrossHop",
                decision="reject", reason_code="crosshop_verify_mismatch",
                rationale=(f"Verifier derived: '{verdict.verifier_answer[:200]}' — "
                           f"{verdict.rationale}"),
                candidate_question=cand.input, candidate_answer=cand.expected_output,
            )
            return []

    # ── Stage 4: Dedup + accept ─────────────────────────────────
    payload = (f"OM_A ({seed.om_a_id}):\n{om_a.body[:4000]}\n\n---\n\n"
               f"OM_B ({seed.om_b_id}):\n{om_b.body[:4000]}")
    row = _accept_or_reject_cross_doc(
        candidate_text   = cand.input.strip(),
        candidate_answer = cand.expected_output.strip(),
        question_type    = "CrossHop",
        source_id        = source_id,
        payload          = payload,
        extra_fields={
            "shared_entity": seed.entity_id,
            "entity_label":  seed.entity_label,
            "om_a_id":       seed.om_a_id,
            "om_b_id":       seed.om_b_id,
            "relation_a":    seed.relation_a,
            "relation_b":    seed.relation_b,
            "evidence_om_a": cand.evidence_om_a,
            "evidence_om_b": cand.evidence_om_b,
        },
        dedup=dedup, logger=logger,
    )
    return [row] if row is not None else []


# ════════════════════════════════════════════════════════════════════════════
# SINGLE-DOC ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════

def process_om(
    om: OM,
    k: int,
    feasibility_chain,
    llm,
    dedup: Deduper,
    logger: RejectionLogger,
    stop_after: Optional[str] = None,
    oos_verifier_chain=None,
    bm25_index: Optional[BM25Retriever] = None,
) -> List[dict]:
    """Run feasibility → generation → dedup on one OM. Returns accepted QA pairs.

    Rejections at feasibility and dedup are written to logger.
    Quality scoring is handled separately by the downstream annotator.
    """
    # ── Stage 1: feasibility ─────────────────────────────────
    verdicts = run_feasibility(om, feasibility_chain, logger)
    if stop_after == "feasibility":
        return []
    feasible_types = [t for t, v in verdicts.items() if v.feasible]
    if not feasible_types:
        return []

    # ── Stage 2: generation ──────────────────────────────────
    candidates, gen_info = run_generation(om, feasible_types, k, llm)
    if not candidates:
        # Audit: generation silently produced nothing despite feasible types.
        # Log one row per feasible type so the rejection log is honest about
        # the failure mode (LLM returned empty, LLM returned off-types, etc.).
        rationale = (f"Generation returned {gen_info['raw_count']} candidate(s); "
                     f"{gen_info['kept_count']} kept after type filter. "
                     f"Off-types: {gen_info.get('off_types', [])}.")
        for t in feasible_types:
            logger.log(
                stage="dedup",  # treat as same stage in the funnel
                source_om=om.source_id,
                intended_type=t,
                decision="reject",
                reason_code="generation_empty",
                rationale=rationale,
            )
    if stop_after == "generation":
        # Log all generated as 'accept' at the dedup stage placeholder
        for c in candidates:
            logger.log(
                stage="dedup",
                source_om=om.source_id,
                intended_type=c.question_type,
                decision="accept",
                candidate_question=c.input,
                candidate_answer=c.expected_output,
                rationale="--stop-after=generation set",
            )
        return []

    # ── Stage 2.5: Adversarial corpus-wide OOS verification ─────────────────
    # Non-Adversarial candidates pass through unchanged. Adversarial candidates
    # whose answer is findable in another OM are rejected as `oos_leak`.
    if oos_verifier_chain is not None and bm25_index is not None:
        candidates = [
            c for c in candidates
            if verify_adversarial_oos(c, om, bm25_index, oos_verifier_chain, logger)
        ]

    # ── Stage 3: dedup (last gate; survivors become the accepted set) ────────
    accepted: List[dict] = []
    for c in candidates:
        dedup_match = dedup.check(c.input)
        if dedup_match is not None:
            logger.log(
                stage="dedup",
                source_om=om.source_id,
                intended_type=c.question_type,
                decision="reject",
                reason_code=dedup_match["reason"],
                candidate_question=c.input,
                candidate_answer=c.expected_output,
                rationale=(f"Duplicate of earlier question (sim={dedup_match['similarity']:.3f}): "
                           f"'{dedup_match['duplicate_of'][:120]}'"),
            )
            continue
        dedup.add(c.input)
        logger.log(
            stage="dedup",
            source_om=om.source_id,
            intended_type=c.question_type,
            decision="accept",
            candidate_question=c.input,
            candidate_answer=c.expected_output,
        )
        accepted.append({
            "source":          om.source_id,
            "input":           c.input,
            "question_type":   c.question_type,
            "expected_output": c.expected_output,
            "context_used":    om.payload(),
        })
    return accepted


# ════════════════════════════════════════════════════════════════════════════
# CORPUS LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_corpus(limit_docs: Optional[int] = None) -> List[OM]:
    """Load all OMs from data/DataFiles_v2/{inst}/{inst}_om_text_files/.

    NB: must NOT recursively glob the institute folder — sibling
    `{inst}_oms_summary/` (and JU's `ju_om_summ/`) holds short summary
    .txt files that have no frontmatter and would otherwise pollute the
    corpus with non-OM content.
    """
    oms: List[OM] = []
    for inst in INSTITUTE_DIRS:
        text_dir = CORPUS_BASE / inst / f"{inst}_om_text_files"
        for path in sorted(text_dir.glob("*.txt")):
            try:
                post = frontmatter.load(open(path, encoding="utf-8"))
                oms.append(OM(
                    source_id=path.stem,
                    institute=post.metadata.get("institute", inst).upper(),
                    issue_date=str(post.metadata.get("issue_date", "UNKNOWN")),
                    body=post.content,
                    path=path,
                ))
            except Exception as e:
                print(f"  failed to load {path}: {e}")
    if limit_docs:
        oms = oms[:limit_docs]
    return oms


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--types",
                   choices=["single-doc", "temporal", "aggregation", "crosshop", "all"],
                   default="single-doc")
    p.add_argument("--crosshop-max-seeds", type=int, default=100,
                   help="Cap on CrossHop seeds returned by KG discovery.")
    p.add_argument("--crosshop-max-per-entity", type=int, default=3,
                   help="Cap on CrossHop seeds per shared entity. Spreads "
                        "coverage across many entities instead of letting one "
                        "over-connected scheme dominate.")
    p.add_argument("--per-slot", type=int, default=2,
                   help="K candidates per (OM, feasible-type) slot for single-doc types")
    p.add_argument("--per-seed", type=int, default=2,
                   help="Questions per temporal seed (split roughly half descriptive / half predictive)")
    # Cross-family split: a Claude feasibility judge auditing GPT-generated
    # candidates reduces self-preference bias compared to using one model for
    # both stages. Defaults pair Claude Haiku 4.5 (cheap cross-family judge)
    # with gpt-4.1-mini (sharp structured-output generator).
    p.add_argument("--feasibility-model", default="claude-haiku-4-5",
                   help="Model for the feasibility-gate stage. claude-* uses "
                        "Anthropic; otherwise OpenAI.")
    p.add_argument("--generation-model", default="gpt-4.1-mini",
                   help="Model for the question-generation stage.")
    p.add_argument("--model", default=None,
                   help="Legacy: set the SAME model for both stages "
                        "(overrides --feasibility-model and --generation-model).")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--limit-docs", type=int, default=None,
                   help="Smoke test: only process first N OMs")
    p.add_argument("--stop-after", choices=["feasibility", "generation"], default=None,
                   help="Halt after this stage (for debugging)")
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve which model to use at each stage. --model is a legacy escape
    # hatch that forces both stages onto the same model (back-compat).
    if args.model:
        feasibility_model = args.model
        generation_model  = args.model
    else:
        feasibility_model = args.feasibility_model
        generation_model  = args.generation_model

    print("=" * 70)
    print(f"IDiOM-QA v2 generation  |  types={args.types}  K={args.per_slot}")
    print(f"  feasibility model: {feasibility_model}")
    print(f"  generation  model: {generation_model}")
    print("=" * 70)

    feasibility_llm = make_llm(feasibility_model, args.temperature)
    generation_llm  = make_llm(generation_model,  args.temperature)

    # The OOS verifier reuses the cross-family feasibility model so the
    # Adversarial gate is also independent of the generator (mitigates
    # self-preference bias on what counts as "answerable in the corpus").
    oos_verifier_chain = get_oos_verifier_chain(feasibility_llm)

    logger = RejectionLogger(OUT_REJECTIONS)
    dedup = Deduper(semantic=not args.no_semantic)

    corpus = load_corpus(args.limit_docs)
    print(f"Loaded {len(corpus)} OMs from {CORPUS_BASE}")

    # BM25 index used only by the Adversarial OOS-verification gate.
    bm25_index = build_oos_bm25(corpus)
    print(f"Built BM25 index over {len(corpus)} OM bodies for OOS verification.")

    accepted: List[dict] = []

    if args.types in ("single-doc", "all"):
        feasibility_chain = get_feasibility_chain(feasibility_llm)
        for i, om in enumerate(corpus, 1):
            try:
                kept = process_om(om, args.per_slot, feasibility_chain,
                                  generation_llm, dedup, logger,
                                  stop_after=args.stop_after,
                                  oos_verifier_chain=oos_verifier_chain,
                                  bm25_index=bm25_index)
                accepted.extend(kept)
                print(f"  [{i}/{len(corpus)}] {om.source_id:<30}  +{len(kept)}")
            except Exception as e:
                print(f"  [{i}/{len(corpus)}] FAILED {om.source_id}: {e}")

    # Cross-doc types have no feasibility gate (seeds are hand-curated);
    # they only use the generation model.
    if args.types in ("temporal", "all"):
        for seed in TEMPORAL_SEEDS:
            try:
                kept = process_temporal_seed(seed, corpus, generation_llm,
                                             args.per_seed, dedup, logger)
                accepted.extend(kept)
            except Exception as e:
                print(f"  [temporal/{seed.name}] FAILED: {e}")

    if args.types in ("aggregation", "all"):
        for seed in AGGREGATION_SEEDS:
            try:
                kept = process_aggregation_seed(seed, corpus, generation_llm,
                                                dedup, logger)
                accepted.extend(kept)
            except Exception as e:
                print(f"  [aggregation/{seed.key}] FAILED: {e}")

    if args.types in ("crosshop", "all"):
        print("\n--- Cross-doc Multi-hop: KG seed discovery → text generation → cross-family verification ---")
        try:
            graph = Neo4jGraph(
                url=config.NEO4J_URI,
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
            )
        except Exception as e:
            print(f"  [crosshop] Failed to connect to Neo4j ({e}); skipping CrossHop type.")
        else:
            seeds = discover_crosshop_seeds(graph, corpus,
                                            max_seeds=args.crosshop_max_seeds,
                                            max_per_entity=args.crosshop_max_per_entity)
            print(f"  Discovered {len(seeds)} CrossHop seeds from KG")
            crosshop_gen_chain    = get_crosshop_generation_chain(generation_llm)
            # Verifier uses the cross-family feasibility model — same justification
            # as the OOS verifier: independence from the generator.
            crosshop_verify_chain = get_crosshop_verification_chain(feasibility_llm)
            for i, seed in enumerate(seeds, 1):
                try:
                    kept = process_crosshop_seed(
                        seed, corpus, crosshop_gen_chain, crosshop_verify_chain,
                        dedup, logger,
                    )
                    accepted.extend(kept)
                    print(f"  [{i}/{len(seeds)}] crosshop/{seed.entity_id[:25]:<26} "
                          f"({seed.relation_a} ↔ {seed.relation_b})  +{len(kept)}")
                except Exception as e:
                    print(f"  [crosshop/{seed.entity_id[:25]}] FAILED: {e}")

    # ── Write dataset ────────────────────────────────────────
    with open(OUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=2, ensure_ascii=False)

    logger.close()

    # ── Report ───────────────────────────────────────────────
    from collections import Counter
    by_type = Counter(r["question_type"] for r in accepted)
    print("\n" + "=" * 70)
    print(f"Accepted: {len(accepted)} questions")
    for t in SINGLE_DOC_TYPES + CROSS_DOC_TYPES:
        print(f"  {t:<13} {by_type.get(t, 0)}")
    print(f"Dataset    → {OUT_DATASET}")
    print(f"Rejections → {OUT_REJECTIONS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
