# app/rag.py
"""
RAG (Retrieval-Augmented Generation) for company policy Q&A.

- Retrieves top-k relevant chunks from Chroma (filtered by company_id)
- Builds a grounded prompt
- Calls a local LLM (Ollama) via app/llm_client.py
- Returns the answer (and optionally sources)

Used by FastAPI (app/api.py):
  from app.rag import answer_question
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME
from app.llm_client import call_llm


# ----------------------------
# Singletons (load once)
# ----------------------------
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_collection(collection_name: str = "company_policies"):
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(name=collection_name)
    return _collection


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None  # Chroma distance/score may not always be returned


# ----------------------------
# Retrieval
# ----------------------------
def retrieve_chunks(
    company_id: str,
    question: str,
    k: int = 5,
    collection_name: str = "company_policies",
) -> List[RetrievedChunk]:
    """
    Retrieve top-k chunks relevant to the question, filtered to a company_id.
    """
    model = get_embedding_model()
    collection = get_collection(collection_name)

    q_emb = model.encode([question])[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where={"company_id": company_id},
        include=["documents", "metadatas", "distances"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]  # smaller = closer (usually)

    retrieved: List[RetrievedChunk] = []
    for i in range(len(docs)):
        retrieved.append(
            RetrievedChunk(
                text=docs[i],
                metadata=metas[i] if i < len(metas) else {},
                score=dists[i] if i < len(dists) else None,
            )
        )
    return retrieved


def build_context_block(chunks: List[RetrievedChunk], max_chars: int = 12000) -> str:
    """
    Build a context string from retrieved chunks, with light formatting.
    Trims to max_chars to avoid overloading the LLM context window.
    """
    parts: List[str] = []
    used = 0

    for ch in chunks:
        doc_name = ch.metadata.get("doc_name", "document")
        chunk_idx = ch.metadata.get("chunk_idx", "?")
        source_file = ch.metadata.get("source_file", "")

        header = f"[Source: {doc_name} | chunk {chunk_idx}"
        if source_file:
            header += f" | file {source_file}"
        header += "]\n"

        block = header + ch.text.strip() + "\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:  # add partial chunk if some meaningful space remains
                parts.append(block[:remaining])
            break

        parts.append(block)
        used += len(block)

    return "\n---\n".join(parts).strip()


# ----------------------------
# Prompting + Answering
# ----------------------------
def answer_question(
    company_id: str,
    question: str,
    k: int = 5,
    collection_name: str = "company_policies",
    return_sources: bool = False,
) -> Dict[str, Any]:
    """
    Answer a user question using RAG + local LLM.

    Returns:
      {
        "answer": "...",
        "sources": [...]   # optional
      }
    """
    retrieved = retrieve_chunks(
        company_id=company_id,
        question=question,
        k=k,
        collection_name=collection_name,
    )

    context = build_context_block(retrieved)

    system_prompt = (
        "You are a company policy assistant.\n"
        "You MUST answer strictly using the provided policy context.\n"
        "Do NOT invent policy details.\n"
        "If the policy context does not clearly contain the answer, say:\n"
        "  'I can't find this explicitly in the provided policy documents.'\n"
        "and suggest contacting HR / the policy owner.\n"
        "When you provide an answer, reference the relevant Source labels."
    )

    user_prompt = f"""
Company Policy Context:
{context}

Employee Question:
{question}

Instructions:
- Answer using ONLY the Company Policy Context.
- If the answer is not clearly present, say you can't find it in the provided docs.
- Include 1–3 short citations like: (Source: <doc_name> | chunk <idx>)
""".strip()

    # Call local LLM (Ollama)
    llm_text = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)

    result: Dict[str, Any] = {"answer": llm_text}

    if return_sources:
        # Return compact source list for UI
        sources = []
        for ch in retrieved:
            sources.append(
                {
                    "doc_name": ch.metadata.get("doc_name"),
                    "chunk_idx": ch.metadata.get("chunk_idx"),
                    "source_file": ch.metadata.get("source_file"),
                    "score": ch.score,
                }
            )
        result["sources"] = sources

    return result
