from langchain_core.prompts import PromptTemplate

SPARQL_GENERATION_TEMPLATE = """Task: Generate a SPARQL query to answer a question using Wikidata.

You do NOT know entity QIDs (e.g. Q42) in advance. Resolve entities by their
label instead, using this pattern:

  ?person rdfs:label "Douglas Adams"@en .

Standard prefixes (always include exactly these, do not invent others):
  PREFIX wd: <http://www.wikidata.org/entity/>
  PREFIX wdt: <http://www.wikidata.org/prop/direct/>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  PREFIX bd: <http://www.bigdata.com/rdf#>

Common properties (wdt: = "direct value" shortcut, use these unless the
question needs a different one):
  wdt:P31  instance of          wdt:P106 occupation
  wdt:P27  country of citizenship (nationality)   wdt:P569 date of birth
  wdt:P57  director              wdt:P170 creator
  wdt:P50  author                wdt:P175 performer
  wdt:P138 named after           wdt:P495 country of origin
  wdt:P159 headquarters location wdt:P571 inception/founding date
  wdt:P22  father                wdt:P25 mother
  wdt:P26  spouse                wdt:P54 member of sports team

ALWAYS end the query with a label-service block so results come back as
human-readable strings, not QIDs:
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}

Rules:
1. Output ONLY the raw SPARQL query. No markdown fences, no explanation, no
   leading word like "sparql".
2. Match entities by rdfs:label as shown above, not by guessing a QID.
3. Use OPTIONAL for any triple that might not exist for every candidate,
   so a missing fact doesn't silently drop the whole row.
4. For comparison questions (e.g. "is X the same nationality as Y"), query
   both entities' relevant property in ONE query using separate variables
   (?p1Label, ?p2Label) rather than two queries, so the comparison can be
   read directly off one result row.
5. If the question needs multi-hop traversal (e.g. "the nationality of the
   director of film X"), chain the property paths in one query rather than
   planning to issue a second query afterwards — Wikidata's graph usually
   supports this directly (e.g. ?film wdt:P57 ?director . ?director wdt:P27 ?countryLabel).
6. Prefer wdt: (direct/simple statements) over the full statement-node (p:/
   ps:/pq:) model unless the question specifically needs qualifiers
   (e.g. a value that only held during a certain time period).

Question: {question}

SPARQL query:"""

SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question"], template=SPARQL_GENERATION_TEMPLATE
)
