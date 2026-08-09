"""TF-IDF and optional semantic retrieval for FAQ candidates."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from smart_faq.data_loader import clean_text, ensure_chunks


class EmbeddingModel(Protocol):
    """Protocol for sentence-transformer-like models used in semantic search."""

    def encode(self, sentences: Sequence[str]) -> Any:
        """Encode sentences into vectors."""


def validate_query(question: str) -> str:
    """Validate and return a normalized user query."""

    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if clean_text(question) == "":
        raise ValueError("question must not be empty.")
    return question


def _build_result(chunk: dict[str, object], score: float) -> dict[str, object]:
    return {
        "id": chunk["id"],
        "source": chunk["source"],
        "question": chunk["question"],
        "answer": chunk["answer"],
        "score": float(score),
    }


def search_tfidf(
    question: str,
    chunks: Sequence[dict[str, object]],
    top_k: int = 3,
) -> list[dict[str, object]]:
    """Search FAQ chunks with TF-IDF cosine similarity."""

    validate_query(question)
    prepared_chunks = ensure_chunks(chunks)
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    texts = [clean_text(chunk["text"]) for chunk in prepared_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([clean_text(question)])

    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    ranked_indexes = scores.argsort()[::-1]

    results: list[dict[str, object]] = []
    for index in ranked_indexes[:top_k]:
        results.append(_build_result(prepared_chunks[int(index)], float(scores[int(index)])))

    return results


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Load the optional sentence-transformer model with a clear dependency error."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def search_semantic(
    question: str,
    chunks: Sequence[dict[str, object]],
    top_k: int = 3,
    model: EmbeddingModel | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict[str, object]]:
    """Search FAQ chunks with sentence embeddings and cosine similarity."""

    validate_query(question)
    prepared_chunks = ensure_chunks(chunks)
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    embedding_model = model or get_sentence_transformer(model_name)
    texts = [str(chunk["text"]) for chunk in prepared_chunks]

    chunk_vectors = embedding_model.encode(texts)
    question_vector = embedding_model.encode([question])

    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    ranked_indexes = scores.argsort()[::-1]

    results: list[dict[str, object]] = []
    for index in ranked_indexes[:top_k]:
        results.append(_build_result(prepared_chunks[int(index)], float(scores[int(index)])))

    return results
