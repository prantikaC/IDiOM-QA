"""
Load the released knowledge graph (data/kg/kg.json) into Neo4j, so IDiOM-RAG's KG
retriever sees the same graph as in the paper (502 nodes, 591 relations).

    python scripts/load_kg.py --kg ../data/kg/kg.json

Connection settings come from config.py (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
NEO4J_DATABASE, all overridable by environment variables). The target database must
be empty; pass --wipe to delete everything in it first.
"""
import argparse
import json
import os
import sys

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# One id names two nodes (a DA revision stored as both a Directive_or_Decision and a
# Scheme_or_Project). Triples carry no labels, so the end of a project relation takes
# the Scheme_or_Project node and every other relation the first node with that id.
PROJECT_RELATIONS = {"FUNDS_PROJECT", "LEADS_PROJECT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="../data/kg/kg.json")
    ap.add_argument("--wipe", action="store_true", help="delete all nodes in the target database first")
    a = ap.parse_args()
    kg = json.load(open(a.kg, encoding="utf-8"))

    labels = {}
    for n in kg["nodes"]:
        labels.setdefault(n["id"], []).append(n["label"])

    def label_of(node_id, relation, is_target):
        options = labels[node_id]
        if len(options) > 1 and is_target and relation in PROJECT_RELATIONS and "Scheme_or_Project" in options:
            return "Scheme_or_Project"
        return options[0]

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    with driver.session(database=config.NEO4J_DATABASE) as s:
        existing = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if existing and not a.wipe:
            raise SystemExit(f"database '{config.NEO4J_DATABASE}' already has {existing} nodes; "
                             "use --wipe to replace them")
        if a.wipe:
            s.run("MATCH (n) DETACH DELETE n")
        for n in kg["nodes"]:
            props = {k: v for k, v in n.items() if k != "label"}
            s.run(f"CREATE (x:`{n['label']}`) SET x = $props", props=props)
        for t in kg["triples"]:
            props = {k: v for k, v in t.items() if k not in ("source", "relation", "target")}
            s.run(f"MATCH (a:`{label_of(t['source'], t['relation'], False)}` {{id: $s}}) "
                  f"MATCH (b:`{label_of(t['target'], t['relation'], True)}` {{id: $o}}) "
                  f"CREATE (a)-[r:`{t['relation']}`]->(b) SET r = $props",
                  s=t["source"], o=t["target"], props=props)
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    driver.close()
    print(f"loaded {nodes} nodes and {rels} relations into '{config.NEO4J_DATABASE}' at {config.NEO4J_URI}")


if __name__ == "__main__":
    main()
