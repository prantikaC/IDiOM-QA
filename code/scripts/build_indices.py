import os
import sys
import glob
import json
import uuid
import shutil
import argparse
import faiss
import pickle
import frontmatter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

import config
from src.core.device import get_device

# Default text root is the vision-language-corrected corpus.
DEFAULT_TEXT_ROOT = os.path.join("data", "DataFiles_v2")


def build_local_databases(text_root: str = DEFAULT_TEXT_ROOT,
                          vector_store_path: str | None = None,
                          bm25_dir: str | None = None,
                          wipe_first: bool = False):
    print(f"{'='*50}")
    print("🚀 BUILDING TIME-AWARE FAISS (HNSW) AND BM25 INDICES")
    print(f"{'='*50}")
    print(f"text root         : {text_root}")
    print(f"vector store path : {vector_store_path or config.VECTOR_STORE_PATH}")
    print(f"bm25 dir          : {bm25_dir or '(from config)'}")
    print(f"wipe-first        : {wipe_first}\n")

    # Map institutions to their specific text folders
    institutions = {
        "csir": os.path.join(text_root, "csir", "csir_om_text_files"),
        "iacs": os.path.join(text_root, "iacs", "iacs_om_text_files"),
        "ju":   os.path.join(text_root, "ju",   "ju_om_text_files"),
        "ugc":  os.path.join(text_root, "ugc",  "ugc_om_text_files"),
    }

    # Resolve output paths (allow override).
    vector_store_path = vector_store_path or config.VECTOR_STORE_PATH
    bm25_dir_resolved = bm25_dir or (
        os.path.dirname(config.BM25_CORPUS_PATH)
        if hasattr(config, "BM25_CORPUS_PATH")
        else os.path.join("data", "local_db")
    )

    # Optional wipe — guarantees a clean rebuild, no stragglers from the
    # previous corpus (e.g., V1 chunks in a partially-overwritten folder).
    if wipe_first:
        if os.path.isdir(vector_store_path):
            shutil.rmtree(vector_store_path)
            print(f"  Wiped FAISS folder: {vector_store_path}")
        for name in ("bm25_model.pkl", "bm25_data.json"):
            p = os.path.join(bm25_dir_resolved, name)
            if os.path.isfile(p):
                os.remove(p)
                print(f"  Removed: {p}")
        print()

    documents = []

    for inst, folder_path in institutions.items():
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found -> {folder_path}")
            continue

        file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
        print(f"--- Processing {len(file_paths)} Markdown files for {inst.upper()} ---")

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            try:
                # 1. Parse the YAML Frontmatter to get the Time-Aware Metadata
                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)
                
                text_content = post.content
                
                # Force all metadata values to strings to prevent JSON serialization crashes
                metadata = {str(k): str(v) for k, v in post.metadata.items()}
                
                # Ensure metadata has fallbacks
                if 'issue_date' not in metadata:
                    metadata['issue_date'] = 'UNKNOWN'
                
                metadata['institution'] = inst.upper()
                metadata['source_file'] = filename

                # 2. Create LangChain Document
                doc = Document(page_content=text_content, metadata=metadata)
                documents.append(doc)
                
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    if not documents:
        print("\nNo documents loaded. Check your file paths.")
        return

    # 3. Chunk the Markdown safely (respecting tables and paragraphs)
    print("\n--- Chunking Markdown Documents ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        separators=["\n\n", "\n", "|", " ", ""] # '|' protects Docling's markdown tables
    )
    all_chunks = text_splitter.split_documents(documents)

    # Inject metadata header into every chunk so date/institute are always
    # searchable, even in chunks whose OCR body doesn't contain the date.
    for chunk in all_chunks:
        meta_prefix = (
            f"Date: {chunk.metadata.get('issue_date', 'UNKNOWN')} | "
            f"Institute: {chunk.metadata.get('institution', 'UNKNOWN')}\n"
        )
        chunk.page_content = meta_prefix + chunk.page_content

    print(f"Total chunks generated: {len(all_chunks)}")
    
    # =========================================================================
    # OPTIMIZATION BLOCK: GPU Acceleration & HNSW Indexing
    # =========================================================================
    print("\n--- Initializing CUDA-Accelerated Embeddings ---")
    
    # Force the embedding model into the GPU VRAM
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': get_device()},
        encode_kwargs={'normalize_embeddings': True} 
    )
    
    print("--- Building FAISS HNSW Vector Store ---")
    vector_dimension = 384  # Exact dimension for bge-small
    hnsw_index = faiss.IndexHNSWFlat(vector_dimension, 32)
    
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=hnsw_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    print("--- Inserting chunks into HNSW Graph ---")
    uuids = [str(uuid.uuid4()) for _ in range(len(all_chunks))]
    vectorstore.add_documents(documents=all_chunks, ids=uuids)
    
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    vectorstore.save_local(vector_store_path)
    print(f"FAISS (HNSW) index saved to: {vector_store_path}")
    # =========================================================================

    # =========================================================================
    # BM25 LEXICAL INDEX BUILDER
    # =========================================================================
    print("\n--- Building BM25 Keyword Index ---")
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "corpus": [chunk.page_content for chunk in all_chunks],
        "metadata": [chunk.metadata for chunk in all_chunks]
    }
    
    # Ensure local DB folder exists for BM25 artifacts
    os.makedirs(bm25_dir_resolved, exist_ok=True)

    # Save BM25 logic and data
    bm25_model_path  = os.path.join(bm25_dir_resolved, "bm25_model.pkl")
    bm25_corpus_path = os.path.join(bm25_dir_resolved, "bm25_data.json")

    with open(bm25_model_path, 'wb') as f:
        pickle.dump(bm25, f)
    with open(bm25_corpus_path, 'w', encoding='utf-8') as f:
        json.dump(bm25_data, f, ensure_ascii=False, indent=2)

    print(f"BM25 corpus & model saved to: {bm25_dir_resolved}")
    # =========================================================================

    print(f"\n{'='*50}")
    print("DATABASE BUILD COMPLETE. Ready for Time-Aware RAG.")
    print(f"{'='*50}")


def add_documents_to_index(new_text_files: list):
    """
    Incrementally add new text files to the existing FAISS and BM25 indices.
    Avoids a full rebuild — only embeds and indexes the provided files.
    """
    print(f"\n--- [Incremental Index] Adding {len(new_text_files)} new file(s) ---")

    new_docs = []
    for file_path in new_text_files:
        if not os.path.exists(file_path):
            print(f"  Skipping missing file: {file_path}")
            continue
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            metadata = {str(k): str(v) for k, v in post.metadata.items()}
            metadata.setdefault('issue_date', 'UNKNOWN')
            # Derive institution from path
            parts = file_path.replace("\\", "/").split("/")
            for inst in ("csir", "iacs", "ju", "ugc"):
                if inst in parts:
                    metadata['institution'] = inst.upper()
                    break
            metadata.setdefault('institution', 'UNKNOWN')
            metadata['source_file'] = filename
            new_docs.append(Document(page_content=post.content, metadata=metadata))
        except Exception as e:
            print(f"  Failed to read {filename}: {e}")

    if not new_docs:
        print("  No valid documents to add.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", "|", " ", ""]
    )
    new_chunks = text_splitter.split_documents(new_docs)
    for chunk in new_chunks:
        meta_prefix = (
            f"Date: {chunk.metadata.get('issue_date', 'UNKNOWN')} | "
            f"Institute: {chunk.metadata.get('institution', 'UNKNOWN')}\n"
        )
        chunk.page_content = meta_prefix + chunk.page_content
    print(f"  Generated {len(new_chunks)} new chunk(s).")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': get_device()},
        encode_kwargs={'normalize_embeddings': True}
    )

    # ── FAISS incremental update ──────────────────────────────────────────────
    try:
        vectorstore = FAISS.load_local(
            folder_path=config.VECTOR_STORE_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception:
        # No existing index — create fresh
        vector_dimension = 384
        hnsw_index = faiss.IndexHNSWFlat(vector_dimension, 32)
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=hnsw_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    uuids = [str(uuid.uuid4()) for _ in new_chunks]
    vectorstore.add_documents(documents=new_chunks, ids=uuids)
    vectorstore.save_local(config.VECTOR_STORE_PATH)
    print(f"  FAISS index updated: {config.VECTOR_STORE_PATH}")

    # ── BM25 incremental update ───────────────────────────────────────────────
    local_db_path = os.path.dirname(config.BM25_CORPUS_PATH) if hasattr(config, 'BM25_CORPUS_PATH') else os.path.join("data", "local_db")
    bm25_corpus_path = os.path.join(local_db_path, "bm25_data.json")
    bm25_model_path = os.path.join(local_db_path, "bm25_model.pkl")

    existing_corpus = []
    existing_metadata = []
    if os.path.exists(bm25_corpus_path):
        with open(bm25_corpus_path, 'r', encoding='utf-8') as f:
            bm25_data = json.load(f)
        existing_corpus = bm25_data.get("corpus", [])
        existing_metadata = bm25_data.get("metadata", [])

    existing_corpus.extend([c.page_content for c in new_chunks])
    existing_metadata.extend([c.metadata for c in new_chunks])

    tokenized = [text.lower().split() for text in existing_corpus]
    bm25 = BM25Okapi(tokenized)

    with open(bm25_model_path, 'wb') as f:
        pickle.dump(bm25, f)
    with open(bm25_corpus_path, 'w', encoding='utf-8') as f:
        json.dump({"corpus": existing_corpus, "metadata": existing_metadata},
                  f, ensure_ascii=False, indent=2)
    print(f"  BM25 index updated: {bm25_corpus_path}")
    print("--- [Incremental Index] Done ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS (HNSW) and BM25 indices over the clean OM text corpus. "
                    "Default reads from V2 (gpt-4o vision-corrected) text. "
                    "Pass --wipe-first when rebuilding from a different corpus to "
                    "guarantee no stale chunks from a previous run."
    )
    parser.add_argument("--text-root", default=DEFAULT_TEXT_ROOT,
                        help="Root of the cleaned text tree "
                             "(default: data/DataFiles_v2).")
    parser.add_argument("--vector-store-path", default=None,
                        help="Override FAISS output folder "
                             "(default: config.VECTOR_STORE_PATH).")
    parser.add_argument("--bm25-dir", default=None,
                        help="Override BM25 output folder "
                             "(default: dirname of config.BM25_CORPUS_PATH).")
    parser.add_argument("--wipe-first", action="store_true",
                        help="Delete existing FAISS folder and BM25 files before building.")
    args = parser.parse_args()
    build_local_databases(
        text_root=args.text_root,
        vector_store_path=args.vector_store_path,
        bm25_dir=args.bm25_dir,
        wipe_first=args.wipe_first,
    )