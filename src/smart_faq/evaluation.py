"""Retrieval evaluation queries, top-k checks, and result summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smart_faq.prompting import answer_faq


@dataclass(frozen=True)
class TestQuestion:
    """Evaluation question with expected source category."""

    question: str
    expected_source: str


DEFAULT_TEST_QUESTIONS = [
    TestQuestion("how do I reset my password?", "account"),
    TestQuestion("how do I contact support?", "support"),
    TestQuestion("how do I cancel billing?", "billing"),
]


def evaluate(
    chunks: Sequence[dict[str, object]],
    method: str = "tfidf",
    test_questions: Sequence[TestQuestion] = DEFAULT_TEST_QUESTIONS,
) -> dict[str, object]:
    """Evaluate whether the top answered source matches expected categories."""

    if not test_questions:
        raise ValueError("test_questions must not be empty.")

    correct = 0
    rows: list[dict[str, object]] = []

    for test in test_questions:
        response = answer_faq(test.question, chunks, method=method)
        predicted_source = response["sources"][0] if response["sources"] else None

        if predicted_source == test.expected_source:
            correct += 1

        rows.append(
            {
                "question": test.question,
                "expected_source": test.expected_source,
                "predicted_source": predicted_source,
            }
        )

    accuracy = correct / len(test_questions)
    return {"accuracy": round(accuracy, 2), "rows": rows}


def format_evaluation(summary: dict[str, object]) -> str:
    """Format evaluation rows as a concise text report."""

    lines: list[str] = []
    for row in summary["rows"]:
        lines.append("-" * 70)
        lines.append(f"Question: {row['question']}")
        lines.append(f"Expected: {row['expected_source']}")
        lines.append(f"Predicted: {row['predicted_source']}")

    lines.append("")
    lines.append(f"Accuracy: {summary['accuracy']}")
    return "\n".join(lines)
