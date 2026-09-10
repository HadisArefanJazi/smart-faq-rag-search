from smart_faq.main import answer_question


def test_answer_question_returns_grounded_answer(monkeypatch):
    candidates = [
        {
            "id": 1,
            "answer": "Reset your password.",
            "source": "account",
            "text": "reset password",
            "score": 0.9,
        }
    ]

    monkeypatch.setattr(
        "smart_faq.main.retrieve",
        lambda **kwargs: candidates,
    )

    monkeypatch.setattr(
        "smart_faq.main.rerank",
        lambda *args, **kwargs: [
            dict(candidates[0], rerank_score=0.9)
        ],
    )

    response = answer_question(
        query="forgot password",
        chunks=[],
        bm25=None,
        embedding_model=None,
        chunk_embeddings=None,
        reranker=None,
    )

    assert response["answer"] == "Reset your password."
    assert response["source"] == "account"


def test_answer_question_uses_fallback(monkeypatch):
    candidates = [
        {
            "id": 1,
            "answer": "Some answer",
            "source": "account",
            "text": "text",
            "score": 0.1,
        }
    ]

    monkeypatch.setattr(
        "smart_faq.main.retrieve",
        lambda **kwargs: candidates,
    )

    monkeypatch.setattr(
        "smart_faq.main.rerank",
        lambda *args, **kwargs: [
            dict(candidates[0], rerank_score=0.1)
        ],
    )

    response = answer_question(
        query="unrelated question",
        chunks=[],
        bm25=None,
        embedding_model=None,
        chunk_embeddings=None,
        reranker=None,
    )

    assert response["answer"] == "I do not have enough information."
    assert response["source"] is None
