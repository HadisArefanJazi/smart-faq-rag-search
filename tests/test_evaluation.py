import pytest

from smart_faq.evaluation import (
    recall_at_k,
    reciprocal_rank,
    evaluate,
)


def test_recall_at_k():
    results = [{"id": 2}, {"id": 1}, {"id": 3}]

    assert recall_at_k(results, expected_id=1, k=2) == 1
    assert recall_at_k(results, expected_id=3, k=2) == 0


def test_reciprocal_rank():
    results = [{"id": 2}, {"id": 1}, {"id": 3}]

    assert reciprocal_rank(results, 1) == 0.5


def test_evaluate():
    test_cases = [
        {"question": "q1", "expected_id": 1},
        {"question": "q2", "expected_id": 2},
    ]

    def fake_retrieve(question):
        if question == "q1":
            return [{"id": 2}, {"id": 1}]

        return [{"id": 2}, {"id": 1}]

    metrics = evaluate(
        test_cases,
        fake_retrieve,
        k=2,
    )

    assert metrics["Recall@2"] == 1.0
    assert metrics["MRR"] == pytest.approx(0.75)
