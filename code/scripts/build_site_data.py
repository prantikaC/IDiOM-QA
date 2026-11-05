"""Build docs/data.json for the IDiOM-QA explorer page from the release files.
The original PDFs are expected in docs/pdfs/<body>/<om id>.pdf.

    python build_site_data.py <repo root>
"""
import glob
import json
import os
import sys

import frontmatter

root = sys.argv[1]
qs = json.load(open(os.path.join(root, "data/qa/idiom_qa_v4.json"), encoding="utf-8"))
oms = {}
for f in sorted(glob.glob(os.path.join(root, "data/corpus/*/*_om_text_files/*.txt"))):
    post = frontmatter.load(f)
    stem = os.path.basename(f)[:-4]
    rel = os.path.relpath(f, root)
    body_dir = rel.split(os.sep)[2]
    pdf = os.path.join("pdfs", body_dir, stem + ".pdf")
    oms[stem] = {"body": str(post.metadata.get("institute", "")).upper(),
                 "date": str(post.metadata.get("issue_date", "")),
                 "path": rel, "text": post.content.strip(),
                 "pdf": pdf if os.path.exists(os.path.join(root, "docs", pdf)) else None}

out = []
for i, r in enumerate(qs):
    ids = []
    if r.get("source") and ":" not in r["source"]:
        ids.append(r["source"])
    ids += [r[k] for k in ("om_a_id", "om_b_id") if r.get(k)]
    ids += list(r.get("seed_docs") or [])
    ids = list(dict.fromkeys(ids))
    if r["question_type"] == "Adversarial":
        ids = []  # no OM answers these; the OM the generator started from is not evidence
    ids.sort(key=lambda d: oms[d]["date"])
    ev = {}
    for k, om in (("evidence_om_a", r.get("om_a_id")), ("evidence_om_b", r.get("om_b_id"))):
        if r.get(k) and om:
            ev.setdefault(om, []).append(r[k])
    out.append({"id": i + 1, "type": r["question_type"], "q": r["input"], "a": r["expected_output"],
                "oms": ids, "ev": ev})

json.dump({"questions": out, "oms": oms}, open(os.path.join(root, "docs/data.json"), "w", encoding="utf-8"),
          ensure_ascii=False, separators=(",", ":"))
print(len(out), "questions,", len(oms), "OMs")
