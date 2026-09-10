import numpy as np
from rank_bm25 import BM25Okapi


def build_bm25(chunks):
    tokenized_docs = [
        chunk["text"].split()
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_docs)


def build_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    return model.encode(
        texts,
        normalize_embeddings=True
    )


def normalize(scores):
    scores = np.array(scores, dtype=float)

    if scores.max() == scores.min():
        return np.zeros_like(scores)

    return (
        (scores - scores.min())
        / (scores.max() - scores.min())
    )


def retrieve(
    query,
    chunks,
    bm25,
    embedding_model,
    chunk_embeddings,
    method="hybrid",
    top_k=10,
    alpha=0.5,
):
    query = query.strip().lower()

    if not query:
        raise ValueError("Query cannot be empty.")

    # Sparse retrieval: BM25
    bm25_scores = bm25.get_scores(query.split())

    # Dense retrieval: embeddings
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
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
        raise ValueError(
            "method must be bm25, semantic, or hybrid"
        )

    indexes = np.argsort(final_scores)[::-1][:top_k]

    results = []

    for i in indexes:
        result = chunks[int(i)].copy()

        result["score"] = float(final_scores[i])
        result["bm25_score"] = float(bm25_scores[i])
        result["dense_score"] = float(dense_scores[i])

        results.append(result)

    return results


def rerank(query, results, reranker, top_k=3):
    pairs = [
        [query, result["text"]]
        for result in results
    ]

    raw_scores = reranker.predict(pairs)

    for result, score in zip(results, raw_scores):

        # sigmoid → 0 to 1
        rerank_score = 1 / (1 + np.exp(-float(score)))

        result["rerank_score"] = rerank_score

    results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]


def passes_threshold(result, threshold=0.30):
    return result["rerank_score"] >= threshold
