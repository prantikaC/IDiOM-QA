import os
import sys
import glob
import re
import json
import argparse
import dspy
import frontmatter
from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional

# ---------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from langchain_neo4j import Neo4jGraph
import config

DATA_DIR = os.path.join(ROOT_DIR, "data")
ONTOLOGY_PATH = os.path.join(DATA_DIR, "om_ontology.json")
# Default text root is V2 (gpt-4o vision-corrected). Override via --text-root for V1.
DEFAULT_TEXT_ROOT = os.path.join(DATA_DIR, "DataFiles_v2")
DEFAULT_LOG_PATH  = os.path.join(DATA_DIR, "eval", "extracted_triples_log_v2.jsonl")

# ---------------------------------------------------------
# 1. Dynamic Pydantic Schema Generator
# ---------------------------------------------------------
def get_dynamic_kg_schema(ontology_filepath: str):
    """Reads the JSON ontology and dynamically generates strict Pydantic models."""
    if not os.path.exists(ontology_filepath):
        raise FileNotFoundError(f"Ontology file not found at {ontology_filepath}")

    with open(ontology_filepath, 'r') as f:
        ontology = json.load(f)
        
    allowed_nodes = ontology.get("nodes", [])
    allowed_rels = ontology.get("relations", [])
    
    nodes_str = ", ".join(allowed_nodes)
    rels_str = ", ".join(allowed_rels)
    
    DynamicNode = create_model(
        'DynamicNode',
        # We leave 'id' strictly required because a node without an ID breaks Neo4j topology
        id=(str, Field(description="The unique name or identifier...")),
        
        # Add default="Entity" to prevent missing-field crashes
        label=(str, Field(default="Entity", description=f"CRITICAL: The entity type...")),
        
        properties=(Optional[Dict[str, Optional[str]]], Field(default_factory=dict, description="..."))
    )
    
    DynamicTriple = create_model(
        'DynamicTriple',
        subject=(DynamicNode, ...),
        
        # Add default="RELATED_TO" to prevent missing-field crashes
        predicate=(str, Field(default="RELATED_TO", description=f"CRITICAL: The relationship...")),
        
        object=(DynamicNode, ...),
        properties=(Optional[Dict[str, Optional[str]]], Field(default_factory=dict, description="..."))
    )
    
    DynamicGraphExtraction = create_model(
        'DynamicGraphExtraction',
        triples=(List[DynamicTriple], Field(description="List of extracted relationship triples from the document."))
    )
    
    return DynamicGraphExtraction, ontology, allowed_nodes, allowed_rels


# ---------------------------------------------------------
# 2. Main Builder Execution
# ---------------------------------------------------------
def wipe_kg(graph):
    """Delete every node and relationship before rebuilding from clean text."""
    print("Wiping existing KG (MATCH (n) DETACH DELETE n)...")
    graph.query("MATCH (n) DETACH DELETE n")
    print("  -> all nodes and relationships removed.\n")


def build_knowledge_graph(text_root: str = DEFAULT_TEXT_ROOT,
                          log_path: str  = DEFAULT_LOG_PATH,
                          wipe: bool     = False):
    print(f"{'='*50}")
    print("🕸️  BUILDING DSPY-POWERED NEO4J KNOWLEDGE GRAPH")
    print(f"{'='*50}")
    print(f"text root : {text_root}")
    print(f"log path  : {log_path}")
    print(f"wipe-first: {wipe}")
    print()

    # 1. Load Dynamic Ontology and Compile the Rules Prompt
    try:
        UserGraphModel, ontology_data, allowed_nodes, allowed_rels = get_dynamic_kg_schema(ONTOLOGY_PATH)
        
        rules_str = f"Context: {ontology_data.get('description', '')}\n\nRules:\n" + "\n".join(ontology_data.get('rules', []))
        
        dr_constraints = ontology_data.get("domain_range", {})
        if dr_constraints:
            rules_str += "\n\nSTRICT RELATIONSHIP CONSTRAINTS (Domain -> Range):\n"
            for rel, constraint in dr_constraints.items():
                rules_str += f"- {rel}: {constraint}\n"
                
        rules_str += "\n\nALLOWED NODE PROPERTIES:\n" + "\n".join(ontology_data.get('node_properties', []))
        rules_str += "\n\nALLOWED RELATION PROPERTIES:\n" + "\n".join(ontology_data.get('relation_properties', []))
        
        print(" Successfully loaded dynamic ontology schema with constraints.")
    except Exception as e:
        print(f" Failed to load ontology: {e}")
        return

    # 2. Define the DSPy Signature dynamically using the new Pydantic Model
    class DynamicExtractInstitutionalGraph(dspy.Signature):
        """
        You are an expert ontology extractor for Academic and Institutional Communications.
        Extract the knowledge graph according to the STRICT user-defined ontology rules provided below.
        Focus heavily on structural data (directives, funding, personnel, positions).
        """
        ontology_rules: str = dspy.InputField(desc="The specific semantic rules, domain/range constraints, and context for this extraction.")
        document_text: str = dspy.InputField()
        
        # Natively inject the generated model here!
        knowledge_graph: UserGraphModel = dspy.OutputField()

    # 3. Connect to Database
    try:
        graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD
        )
        print(" Successfully connected to Neo4j Database.\n")
    except Exception as e:
        print(f" Failed to connect to Neo4j: {e}")
        return

    # 3b. Wipe existing KG + triples log if requested (recommended for V2 rebuild
    # so OCR-noisy V1 entities don't coexist with V2-clean entities).
    if wipe:
        wipe_kg(graph)
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f" Removed prior triples log: {log_path}\n")

    # 4. Configure DSPy
    try:
        llm = dspy.LM(provider="openai", model="gpt-4.1-mini", temperature=0.0)
        dspy.configure(lm=llm)
        
        # Instantiate the Predictor with our dynamic class
        extractor = dspy.Predict(DynamicExtractInstitutionalGraph)
        
    except Exception as e:
        print(f" Failed to configure DSPy: {e}")
        return

    # 5. Execute Extraction
    institutions = {
        "csir": os.path.join(text_root, "csir", "csir_om_text_files"),
        "iacs": os.path.join(text_root, "iacs", "iacs_om_text_files"),
        "ju":   os.path.join(text_root, "ju",   "ju_om_text_files"),
        "ugc":  os.path.join(text_root, "ugc",  "ugc_om_text_files"),
    }

    total_triples_inserted = 0

    # Ensure log directory exists.
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    print(f" Evaluation Log Path set to: {log_path}\n")

    for inst, folder_path in institutions.items():
        if not os.path.exists(folder_path):
            continue

        file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
        print(f"--- Processing {len(file_paths)} Markdown/Text files for {inst.upper()} ---")

        for file_path in file_paths:
            filename = os.path.basename(file_path)

            try:
                with open(file_path, mode='r', encoding='utf-8') as f:
                    post = frontmatter.load(f)

                # Pass the body (post-frontmatter) to DSPy so the LLM never
                # sees structured metadata it might confuse with content.
                text = post.content
                if not text.strip():
                    continue

                # Frontmatter-derived temporal stamps. These are injected
                # deterministically onto every relationship + every Document
                # node, regardless of what the LLM extracts. Authoritative.
                doc_basename = os.path.splitext(filename)[0]
                temporal_props = {
                    "issue_date":  str(post.metadata.get("issue_date", "UNKNOWN")),
                    "source_om":   doc_basename,
                    "superseded":  str(post.metadata.get("superseded", "0")),
                }
                doc_node_props = {
                    "issue_date":    temporal_props["issue_date"],
                    "superseded":    temporal_props["superseded"],
                    "superseded_by": str(post.metadata.get("superseded_by", "")),
                }

                print(f" Extracting triples via DSPy: {filename}")

                result = extractor(ontology_rules=rules_str, document_text=text)
                triples = result.knowledge_graph.triples
                inserted_for_file = 0

                # Initialize the evaluation record for this specific document
                eval_record = {
                    "document": filename,
                    "institute": inst.upper(),
                    "triples": []
                }

                for triple in triples:
                    # Sanitize Relation
                    raw_relation = str(triple.predicate).upper()
                    clean_relation = re.sub(r'[^A-Z0-9_]+', '', raw_relation)
                    
                    if clean_relation not in allowed_rels:
                        clean_relation = "RELATED_TO"

                    # Sanitize Node Labels
                    subj_label = re.sub(r'[^A-Za-z0-9_]', '', triple.subject.label)
                    obj_label = re.sub(r'[^A-Za-z0-9_]', '', triple.object.label)
                    
                    if subj_label not in allowed_nodes: subj_label = "Document"
                    if obj_label not in allowed_nodes: obj_label = "Directive_or_Decision"

                    # Extract Properties Dicts safely and remove 'None' values
                    subj_props = {k: v for k, v in (triple.subject.properties or {}).items() if v is not None}
                    obj_props = {k: v for k, v in (triple.object.properties or {}).items() if v is not None}
                    rel_props = {k: v for k, v in (triple.properties or {}).items() if v is not None}

                    # ─── KG v2: deterministic temporal stamps ────────────
                    # Inject issue_date + source_om + superseded onto EVERY
                    # relationship. When the subject is a Document, also stamp
                    # the Document node with issue_date / superseded /
                    # superseded_by sourced from its frontmatter. These are
                    # authoritative — they override any LLM-extracted props
                    # with the same keys.
                    rel_props.update(temporal_props)
                    if subj_label == "Document":
                        subj_props.update(doc_node_props)

                    # --- APPEND TO EVAL RECORD ---
                    # We log the sanitized variables to reflect exactly what goes into Neo4j
                    eval_record["triples"].append({
                        "subject_id": triple.subject.id.strip(),
                        "subject_label": subj_label,
                        "subject_properties": subj_props,
                        
                        "relation": clean_relation,
                        "relation_properties": rel_props,
                        
                        "object_id": triple.object.id.strip(),
                        "object_label": obj_label,
                        "object_properties": obj_props
                    })

                    # Construct and Execute Cypher
                    cypher_query = f"""
                    MERGE (s:{subj_label} {{id: $subj_id}})
                    ON CREATE SET s.source = $institute
                    SET s += $subj_props
                    
                    MERGE (o:{obj_label} {{id: $obj_id}})
                    ON CREATE SET o.source = $institute
                    SET o += $obj_props
                    
                    MERGE (s)-[r:{clean_relation}]->(o)
                    SET r += $rel_props
                    """
                    
                    graph.query(
                        cypher_query, 
                        params={
                            "subj_id": triple.subject.id.strip(),
                            "obj_id": triple.object.id.strip(),
                            "institute": inst.upper(),
                            "subj_props": subj_props,
                            "obj_props": obj_props,
                            "rel_props": rel_props
                        }
                    )
                    inserted_for_file += 1
                    total_triples_inserted += 1
                
                # --- WRITE RECORD TO JSONL ---
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(json.dumps(eval_record) + '\n')
                
                print(f" Inserted {inserted_for_file} triples and logged to evaluation file.")
                        
            except Exception as e:
                print(f" Failed to process {filename}: {e}")

    graph.refresh_schema()
    print(f"\n KG Build Complete! Inserted {total_triples_inserted} automated relationships.")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Neo4j KG from clean OM text. "
                    "Default reads from V2 (gpt-4o vision) text under "
                    "data/DataFiles_v2/.  Pass --wipe when rebuilding so V1's "
                    "OCR-noisy entities don't pollute the V2 graph."
    )
    parser.add_argument("--text-root", default=DEFAULT_TEXT_ROOT,
                        help="Root of the cleaned text tree. "
                             "Use data/DataFiles_Cleaned to rebuild from V1 OCR text.")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH,
                        help="JSONL log of extracted triples (append-only).")
    parser.add_argument("--wipe", action="store_true",
                        help="Wipe the Neo4j KG and the triples log before building. "
                             "Recommended when rebuilding from V2 to avoid V1/V2 entity coexistence.")
    args = parser.parse_args()
    build_knowledge_graph(text_root=args.text_root, log_path=args.log, wipe=args.wipe)