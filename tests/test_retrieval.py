import numpy as np

from smart_faq.retrieval import (
    build_bm25,
    build_embeddings,
    retrieve,
    rerank,
    passes_threshold,
)


CHUNKS = [
    {
        "id": 1,
        "source": "account",
        "question": "How do I reset my password?",
        "answer": "Use Forgot Password.",
        "text": "how do i reset my password use forgot password",
    },
    {
        "id": 2,
        "source": "support",
        "question": "How do I contact support?",
        "answer": "Email support.",
        "text": "how do i contact support email support",
    },
    {
        "id": 3,
        "source": "billing",
        "question": "What is the refund policy?",
        "answer": "Refunds are available within 14 days.",
        "text": "what is the refund policy refunds available within 14 days",
    },
]


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        single = isinstance(texts, str)
        texts = [texts] if single else texts

        vectors = []

        for text in texts:
            text = text.lower()

            vector = np.array([
                float("password" in text),
                float("support" in text),
                float("refund" in text),
            ])

            norm = np.linalg.norm(vector)

            if normalize_embeddings and norm:
                vector = vector / norm

            vectors.append(vector)

        result = np.array(vectors)

        return result[0] if single else result


class FakeReranker:
    def predict(self, pairs):
        return np.array([
            4.0 if "support" in text.lower() else 0.0
            for _, text in pairs
        ])


def test_bm25_retrieval():
    model = FakeEmbeddingModel()
    bm25 = build_bm25(CHUNKS)
    embeddings = build_embeddings(CHUNKS, model)

    results = retrieve(
        "reset password",
        CHUNKS,
        bm25,
        model,
        embeddings,
        method="bm25",
    )

    assert results[0]["id"] == 1


def test_semantic_retrieval():
    model = FakeEmbeddingModel()
    bm25 = build_bm25(CHUNKS)
    embeddings = build_embeddings(CHUNKS, model)

    results = retrieve(
        "contact support",
        CHUNKS,
        bm25,
        model,
        embeddings,
        method="semantic",
    )

    assert results[0]["id"] == 2


def test_hybrid_retrieval():
    model = FakeEmbeddingModel()
    bm25 = build_bm25(CHUNKS)
    embeddings = build_embeddings(CHUNKS, model)

    results = retrieve(
        "refund policy",
        CHUNKS,
        bm25,
        model,
        embeddings,
        method="hybrid",
    )

    assert results[0]["id"] == 3


def test_cross_encoder_reranking():
    results = [
        dict(CHUNKS[0], score=0.8),
        dict(CHUNKS[1], score=0.7),
    ]

    reranked = rerank(
        "contact support",
        results,
        FakeReranker(),
    )

    assert reranked[0]["id"] == 2
    assert passes_threshold(reranked[0], 0.5)
