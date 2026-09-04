"""TF-IDF and semantic FAQ retrieval."""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from smart_faq.data_loader import clean_text, ensure_chunks


def validate_query(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string.")

    if not clean_text(question):
        raise ValueError("question must not be empty.")

    return question


def build_result(
    chunk: dict[str, object],
    score: float,
) -> dict[str, object]:

    return {
        "id": chunk["id"],
        "source": chunk["source"],
        "question": chunk["question"],
        "answer": chunk["answer"],
        "score": float(score),
    }


def search_tfidf(
    question: str,
    chunks: list[dict[str, object]],
    top_k: int = 3,
) -> list[dict[str, object]]:

    validate_query(question)
    prepared_chunks = ensure_chunks(chunks)

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    texts = [
        clean_text(chunk["text"])
        for chunk in prepared_chunks
    ]

    vectorizer = TfidfVectorizer(stop_words="english")

    chunk_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([clean_text(question)])

    scores = cosine_similarity(
        question_vector,
        chunk_vectors,
    )[0]

    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        result = build_result(
            prepared_chunks[int(index)],
            scores[int(index)],
        )

        results.append(result)

    return results


def get_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
):

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers first."
        ) from exc

    return SentenceTransformer(model_name)


def search_semantic(
    question: str,
    chunks: list[dict[str, object]],
    top_k: int = 3,
    model=None,
) -> list[dict[str, object]]:

    validate_query(question)
    prepared_chunks = ensure_chunks(chunks)

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if model is None:
        model = get_sentence_transformer()

    texts = [
        str(chunk["text"])
        for chunk in prepared_chunks
    ]

    chunk_vectors = model.encode(texts)
    question_vector = model.encode([question])

    scores = cosine_similarity(
        question_vector,
        chunk_vectors,
    )[0]

    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        result = build_result(
            prepared_chunks[int(index)],
            scores[int(index)],
        )

        results.append(result)

    return results
