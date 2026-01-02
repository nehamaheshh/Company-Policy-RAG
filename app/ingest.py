# app/ingest.py
"""
Ingest a PDF policy document into a local Chroma vector DB.

Works on Windows + PowerShell.

Usage (CLI):
  python -m app.ingest --company-id acme --doc-name handbook_2025 --pdf-path data/raw/acme_handbook_2025.pdf

This module is also imported by FastAPI (app/api.py) as:
  ingest_pdf(company_id, doc_name, pdf_path)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Union

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from a PDF using pypdf.
    Note: Some PDFs are scanned images; those will return little/no text unless OCR is used.
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages_text: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        # Add a page marker to help debugging / citations later
        pages_text.append(f"\n\n--- Page {i+1} ---\n{t}")

    return "".join(pages_text).strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Good enough to start. You can later switch to token-based chunking.

    chunk_size: number of characters per chunk
    overlap: number of characters of overlap between consecutive chunks
    """
    text = (text or "").strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def ingest_pdf(
    company_id: str,
    doc_name: str,
    pdf_path: Union[str, Path],
    collection_name: str = "company_policies",
    chunk_size: int = 1200,
    overlap: int = 200,
    batch_size: int = 32,
) -> dict:
    """
    Ingest a PDF into Chroma, stored under a company_id namespace (metadata filter).

    Returns a dict with summary stats.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1) Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError(
            "No text extracted from PDF. If it's a scanned PDF, you'll need OCR (e.g., 'unstructured' or Tesseract)."
        )

    # 2) Chunk
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Chunking produced no chunks. Check PDF text extraction output.")

    # 3) Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 4) Connect to Chroma
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(name=collection_name)

    # 5) Embed in batches + upsert
    ids: List[str] = []
    metadatas: List[dict] = []
    documents: List[str] = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{company_id}::{doc_name}::{i}")
        metadatas.append(
            {
                "company_id": company_id,
                "doc_name": doc_name,
                "chunk_idx": i,
                "source_file": pdf_path.name,
            }
        )
        documents.append(chunk)

    # Compute embeddings (SentenceTransformers handles batching internally too,
    # but we keep explicit batching for stability on smaller GPUs)
    all_embeddings: List[list] = []
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        emb = model.encode(batch_docs, show_progress_bar=False)
        all_embeddings.extend(emb.tolist())

    # Add to Chroma
    # Note: Chroma "add" will error if IDs already exist.
    # If you want overwrite behavior, delete by doc_name first or use unique doc_name each time.
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    return {
        "status": "ok",
        "company_id": company_id,
        "doc_name": doc_name,
        "pdf_path": str(pdf_path),
        "chunks_added": len(documents),
        "collection": collection_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest a company policy PDF into Chroma.")
    parser.add_argument("--company-id", required=True, help="e.g. acme")
    parser.add_argument("--doc-name", required=True, help="e.g. handbook_2025")
    parser.add_argument("--pdf-path", required=True, help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result = ingest_pdf(
        company_id=args.company_id,
        doc_name=args.doc_name,
        pdf_path=args.pdf_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(result)


if __name__ == "__main__":
    main()
