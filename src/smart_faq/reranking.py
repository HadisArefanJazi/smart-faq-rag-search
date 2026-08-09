"""Reranking, score combination, thresholding, and final result ordering."""

from __future__ import annotations

from typing import Sequence

from smart_faq.data_loader import clean_text
from smart_faq.retrieval import validate_query


def rerank(question: str, results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Apply the keyword-overlap bonus and sort by reranking score."""

    validate_query(question)
    words = clean_text(question).split()
    reranked: list[dict[str, object]] = []

    for result in results:
        item = dict(result)
        text = clean_text(f"{item['question']} {item['answer']}")
        bonus = 0.0

        for word in words:
            if word in text:
                bonus += 0.05

        item["rerank_score"] = float(item["score"]) + bonus
        reranked.append(item)

    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)


def passes_threshold(result: dict[str, object], threshold: float = 0.20) -> bool:
    """Return whether the best retrieval score is confident enough to answer."""

    if threshold < 0:
        raise ValueError("threshold cannot be negative.")
    return float(result["score"]) >= threshold


def best_result(results: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return the first reranked result with clear handling for empty results."""

    if not results:
        raise ValueError("No retrieval results available.")
    return dict(results[0])
