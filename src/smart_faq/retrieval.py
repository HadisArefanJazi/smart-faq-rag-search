import numpy as np
from rank_bm25 import BM25Okapi


def build_bm25(chunks):
    tokenized_docs = [chunk["text"].split() for chunk in chunks]
    return BM25Okapi(tokenized_docs)


def build_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    return model.encode(
        texts,
        normalize_embeddings=True,
    )


def normalize(scores):
    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        return scores

    if scores.max() == scores.min():
        return np.zeros_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


def retrieve(
    query,
    chunks,
    bm25=None,
    embedding_model=None,
    chunk_embeddings=None,
    method="hybrid",
    top_k=10,
    alpha=0.5,
):
    query = query.strip().lower()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    bm25_scores = None
    dense_scores = None

    if method in {"bm25", "hybrid"}:
        if bm25 is None:
            raise ValueError("BM25 index is required.")

        bm25_scores = bm25.get_scores(query.split())

    if method in {"semantic", "hybrid"}:
        if embedding_model is None or chunk_embeddings is None:
            raise ValueError("Embedding model and embeddings are required.")

        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
        )

        dense_scores = chunk_embeddings @ query_embedding

    if method == "bm25":
        final_scores = normalize(bm25_scores)

    elif method == "semantic":
        final_scores = dense_scores

    elif method == "hybrid":
        final_scores = (
            alpha * normalize(bm25_scores)
            + (1 - alpha) * normalize(dense_scores)
        )

    else:
        raise ValueError("method must be bm25, semantic, or hybrid")

    indexes = np.argsort(final_scores)[::-1][:top_k]

    results = []

    for i in indexes:
        result = chunks[int(i)].copy()
        result["score"] = float(final_scores[i])

        if bm25_scores is not None:
            result["bm25_score"] = float(bm25_scores[i])

        if dense_scores is not None:
            result["dense_score"] = float(dense_scores[i])

        results.append(result)

    return results


def rerank(query, results, reranker, top_k=3):
    pairs = [[query, result["text"]] for result in results]

    scores = reranker.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(
            1 / (1 + np.exp(-float(score)))
        )

    return sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )[:top_k]


def passes_threshold(result, threshold=0.30):
    return result["rerank_score"] >= threshold
