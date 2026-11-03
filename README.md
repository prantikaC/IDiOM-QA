# IDiOM-QA

Question-answering benchmark over 116 Office Memoranda (OMs) and related notices
from four Indian academic and research bodies (UGC, CSIR, IACS, Jadavpur University),
with the IDiOM-RAG reference system and the outputs of twelve RAG systems.

## Contents
- `data/qa/idiom_qa_v4_paper.json`: the 475 QA pairs as scored in the paper (71 Adversarial).
- `data/qa/idiom_qa_v4.json`: the corrected version (67 Adversarial; 4 Adversarial items
  re-typed and 25 gold answers corrected after a final check).
  Each item has `input`, `expected_output`, `question_type`, `source` (OM id) and
  `context_used` (gold evidence); CrossHop items also carry the two OMs and their evidence.
- `data/corpus/<body>/<body>_om_text_files/`: hand-checked VLM transcriptions of all 116 OMs
  (YAML frontmatter with `issue_date`); `<body>_oms_summary/` holds one-paragraph summaries.
- `data/kg/kg.json`: the typed knowledge graph (397 entities, 105 Document nodes, 591 triples;
  every triple carries `source_om` and `issue_date`).
- `data/qrels/`: TREC-format OM-level relevance judgements for the 404 answerable questions
  of the paper version, with `queries.tsv` and `docids.tsv`.
- `results/`: per-question outputs and scores of the twelve RAG systems and the two reference
  runs (closed book, gold evidence) for the paper version (`idiom_qa_v4_paper.json`):
  - `ragas/`: six RAGAS metrics (judge deepseek-r1:32b); each detailed CSV also holds the
    question, response, retrieved contexts and gold answer;
  - `surface/`: ROUGE, BLEU, METEOR, BERTScore;
  - `hhem/` and `factcheck/` (AlignScore, MiniCheck): context-answer support;
  - `retrieval/`: retriever runs and qrels-based metrics.
  Every number and figure in the paper is computed from these files.
- `code/`: IDiOM-RAG (`src/`), the eleven baselines (`baselines/`), index and KG builders,
  the question generator and the evaluation scripts (`scripts/`).

## Running IDiOM-RAG
    cd code && pip install -r requirements.txt
    python scripts/build_indices.py --text-root ../data/corpus
    # optional: load data/kg/kg.json into Neo4j (set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    # local models via Ollama: qwen2.5:7b (generation), qwen2.5:32b (reasoning)
    python scripts/run_idiom_rag.py --backend ollama --dataset ../data/qa/idiom_qa_v4_paper.json

Scoring: `scripts/evaluate_ragas_metrics.py` (RAGAS, judge deepseek-r1:32b via Ollama).
All hyperparameters are in `code/config.py`.

## Licence
- **Code** (`code/`): MIT, see `LICENSE`.
- **Data and results** (`data/`, `results/`): Creative Commons Attribution 4.0 International
  (CC-BY-4.0), see `LICENSE-DATA`, `data/LICENSE` and `results/LICENSE`.

The OM texts in `data/corpus/` reproduce public documents issued by UGC, CSIR, IACS and
Jadavpur University and remain subject to the terms of those bodies. CC-BY-4.0 applies to our
own contributions: the questions and gold answers, the corrections to the transcriptions, the
document summaries, the knowledge graph, the relevance judgements and the system results.
