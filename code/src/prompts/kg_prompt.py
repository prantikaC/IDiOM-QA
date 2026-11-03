from langchain_core.prompts import PromptTemplate

# Rewritten 2026-10-28. The previous prompt matched entities by exact ID and
# carried supersession rules for a property that is never set; on a 20-question
# Relational probe it produced blank or zero-row Cypher for all 20, with both
# Qwen 7B and 32B. Node IDs in this graph are underscore strings
# (Jadavpur_University), merged duplicates keep their old IDs in `aliases`, and
# every relation records `source_om` and `issue_date`, so the prompt now asks for
# one robust pattern: keyword match on id / aliases / description, then return
# the 1-hop facts with provenance. Few-shot entities are invented so no test
# question leaks into the prompt.
CYPHER_GENERATION_TEMPLATE = """Task: Write one Cypher query that retrieves facts from a knowledge graph of Indian institutional Office Memoranda (OMs).

Schema:
{schema}

How this graph is stored:
- Node `id` values are names with underscores instead of spaces, e.g. Jadavpur_University, Junior_Research_Fellowship_DST_2024. Never match an id exactly.
- Some nodes also have `aliases` (a list of other spellings) and a free-text `description`.
- Every relationship has `source_om` (the OM it came from) and `issue_date`.

Always use this pattern:
1. Pick 1-3 short, distinctive keywords from the question (lower case, 1-3 words each). Add an acronym or a shorter variant when the question uses a long name.
2. Match nodes whose id, aliases or description contain any keyword, comparing lower case with "_" read as a space.
3. Follow one relationship to a neighbour. Use a relationship type from the schema only when the question clearly asks for that relation; otherwise leave it untyped. Do not use node labels or a direction.
4. Return the subject, relation, object, the neighbour's description, source_om and issue_date, with LIMIT 25.
Output ONLY the Cypher query: no markdown fences, no explanation, no leading word "cypher".

Example question: Which organisation funds the Solar Desalination Pilot project?
MATCH (n)-[r:FUNDS_PROJECT]-(m)
WHERE any(k IN ["solar desalination"] WHERE replace(toLower(n.id), "_", " ") CONTAINS k OR any(a IN coalesce(n.aliases, []) WHERE replace(toLower(a), "_", " ") CONTAINS k) OR toLower(coalesce(n.description, "")) CONTAINS k)
RETURN startNode(r).id AS subject, type(r) AS relation, endNode(r).id AS object, coalesce(m.description, "") AS neighbour_description, r.source_om AS source_om, r.issue_date AS issue_date
LIMIT 25

Example question: Who was appointed to the Institutional Biosafety Committee of the Indian Institute of Coastal Studies?
MATCH (n)-[r]-(m)
WHERE any(k IN ["biosafety committee", "iics", "coastal studies"] WHERE replace(toLower(n.id), "_", " ") CONTAINS k OR any(a IN coalesce(n.aliases, []) WHERE replace(toLower(a), "_", " ") CONTAINS k) OR toLower(coalesce(n.description, "")) CONTAINS k)
RETURN startNode(r).id AS subject, type(r) AS relation, endNode(r).id AS object, coalesce(m.description, "") AS neighbour_description, r.source_om AS source_om, r.issue_date AS issue_date
LIMIT 25

Example question: Which department is the Senior Research Fellow post under the HELIX project attached to?
MATCH (n)-[r:ASSOCIATED_WITH]-(m)
WHERE any(k IN ["helix", "senior research fellow"] WHERE replace(toLower(n.id), "_", " ") CONTAINS k OR any(a IN coalesce(n.aliases, []) WHERE replace(toLower(a), "_", " ") CONTAINS k) OR toLower(coalesce(n.description, "")) CONTAINS k)
RETURN startNode(r).id AS subject, type(r) AS relation, endNode(r).id AS object, coalesce(m.description, "") AS neighbour_description, r.source_om AS source_om, r.issue_date AS issue_date
LIMIT 25

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)
