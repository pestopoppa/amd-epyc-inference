"""LLM tools - embeddings, similarity, classification."""

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def embed_text(text: str) -> list[float]:
    """Generate embedding for text using TaskEmbedder."""
    try:
        from orchestration.repl_memory.embedder import TaskEmbedder
        embedder = TaskEmbedder()
        embedding = embedder.embed_text(text)
        return embedding.tolist()
    except Exception as e:
        return {"error": str(e)}


def similarity_search(query: str, items: list[str], top_k: int = 5) -> list[dict]:
    """Find most similar items by embedding similarity."""
    try:
        from orchestration.repl_memory.embedder import TaskEmbedder
        embedder = TaskEmbedder()

        # Embed query
        query_emb = embedder.embed_text(query)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Embed and score all items
        results = []
        for i, item in enumerate(items):
            item_emb = embedder.embed_text(item)
            item_norm = item_emb / (np.linalg.norm(item_emb) + 1e-8)
            score = float(np.dot(query_norm, item_norm))
            results.append({
                "item": item,
                "score": score,
                "index": i,
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        return [{"error": str(e)}]


def classify_text(text: str, categories: list[str]) -> dict:
    """Classify text into one of the given categories using embedding similarity."""
    try:
        results = similarity_search(text, categories, top_k=len(categories))
        if results and not isinstance(results[0], dict) or "error" in results[0]:
            return results[0]

        best = results[0]
        scores = {r["item"]: r["score"] for r in results}

        return {
            "category": best["item"],
            "confidence": best["score"],
            "scores": scores,
        }
    except Exception as e:
        return {"error": str(e)}


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks for processing."""
    # Simple word-based chunking (approximate tokens)
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
        })
        start = end - overlap
        if start >= len(words):
            break

    return chunks


def summarize_chunks(chunks: list[str], query: str | None = None) -> list[dict]:
    """Rank text chunks by relevance to query (or by position if no query)."""
    if query:
        return similarity_search(query, chunks, top_k=len(chunks))
    else:
        # Return in order with position scores
        return [{"item": chunk, "score": 1.0 - i/len(chunks), "index": i}
                for i, chunk in enumerate(chunks)]
