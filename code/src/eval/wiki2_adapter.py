"""
Adapter: 2WikiMultihopQA question -> the frontmatter-tagged .txt file shape
that build_indices.py / build_kg.py already expect for the OM corpus.

Each question's `context` (a list of [title, [sentences]] pairs -- 2 gold +
8 distractor paragraphs in the dataset's "distractor setting") becomes its
own small corpus: one .txt file per paragraph, written under
    <root>/wiki2/wiki2_om_text_files/*.txt
so the existing institutions-dict convention in build_indices.py/build_kg.py
("wiki2" standing in for "csir"/"iacs"/etc.) works with zero changes to
either script.

This corpus is meant to be rebuilt fresh per question (or per small batch)
and torn down afterward -- 2Wiki's "distractor setting" means retrieval
should be scoped to just that question's 10 paragraphs, not pooled into one
shared index the way the OM corpus is. See scripts/run_idiom_rag_2wiki.py.
"""
import os
import re
import shutil
import uuid
import pickle
import json
import glob

import faiss
import frontmatter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src.core.device import get_device


def question_text_root(workdir: str) -> str:
    """Root passed as --text-root to build_indices.py/build_kg.py."""
    return os.path.join(workdir, "wiki2")


def write_paragraphs(row: dict, workdir: str) -> str:
    """
    Wipes and rewrites <workdir>/wiki2/wiki2_om_text_files/*.txt from
    row["context"]. Returns the text_root (== question_text_root(workdir)).
    """
    text_root = question_text_root(workdir)
    out_dir = os.path.join(text_root, "wiki2_om_text_files")
    if os.path.isdir(text_root):
        shutil.rmtree(text_root)
    os.makedirs(out_dir, exist_ok=True)

    for i, (title, sentences) in enumerate(row["context"]):
        body = " ".join(sentences)
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:60].strip("_") or f"para{i}"
        frontmatter = (
            "---\n"
            "issue_date: UNKNOWN\n"
            "institute: WIKI2\n"
            "superseded: 0\n"
            f"source_title: {title}\n"
            "---\n\n"
        )
        fname = f"{i:02d}_{safe_title}.txt"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(frontmatter + body)

    return text_root


def cleanup(workdir: str):
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)


def build_wiki2_indices(text_root: str, vector_store_path: str, bm25_dir: str,
                         embeddings: "HuggingFaceEmbeddings | None" = None):
    """
    FAISS (HNSW) + BM25 index build scoped to ONE flat folder of paragraph
    .txt files (no institutions loop -- unlike build_indices.py, there's
    only ever one "institution" here). Deliberately a separate, minimal
    implementation rather than calling build_indices.py's
    build_local_databases() with a monkey-patched institutions dict, so
    there is zero risk of this touching the OM index-building code path.

    Pass a pre-loaded `embeddings` to avoid reloading the model on every
    question when called in a loop (see run_idiom_rag_2wiki.py).
    """
    file_paths = glob.glob(os.path.join(text_root, "wiki2_om_text_files", "*.txt"))
    documents = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        metadata = {str(k): str(v) for k, v in post.metadata.items()}
        metadata["institution"] = "WIKI2"
        metadata["source_file"] = os.path.basename(file_path)
        documents.append(Document(page_content=post.content, metadata=metadata))

    if os.path.isdir(vector_store_path):
        shutil.rmtree(vector_store_path)
    for name in ("bm25_model.pkl", "bm25_data.json"):
        p = os.path.join(bm25_dir, name)
        if os.path.isfile(p):
            os.remove(p)

    if not documents:
        # Still write empty artifacts so get_local_vector_store/
        # get_local_bm25_retriever's loaders don't hard-fail.
        documents = [Document(page_content="(no context)", metadata={})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": get_device()},
            encode_kwargs={"normalize_embeddings": True},
        )

    vector_dimension = 384
    hnsw_index = faiss.IndexHNSWFlat(vector_dimension, 32)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=hnsw_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    uuids = [str(uuid.uuid4()) for _ in chunks]
    vectorstore.add_documents(documents=chunks, ids=uuids)
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    vectorstore.save_local(vector_store_path)

    tokenized_corpus = [c.page_content.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    os.makedirs(bm25_dir, exist_ok=True)
    with open(os.path.join(bm25_dir, "bm25_model.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(bm25_dir, "bm25_data.json"), "w", encoding="utf-8") as f:
        json.dump({"corpus": [c.page_content for c in chunks],
                   "metadata": [c.metadata for c in chunks]}, f, ensure_ascii=False)

    return len(chunks)
