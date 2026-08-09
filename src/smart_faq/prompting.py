"""Grounded FAQ prompt construction and answer formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smart_faq.reranking import best_result, passes_threshold, rerank
from smart_faq.retrieval import search_semantic, search_tfidf, validate_query

FALLBACK_ANSWER = "I do not have enough information in the FAQ data."


@dataclass(frozen=True)
class FAQAnswer:
    """Structured FAQ answer response."""

    question: str
    answer: str
    sources: list[object]
    best_score: float
    retrieved_faqs: list[dict[str, object]]
    prompt: str

    def as_dict(self) -> dict[str, object]:
        """Return the response as a dictionary."""

        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "best_score": self.best_score,
            "retrieved_faqs": self.retrieved_faqs,
            "prompt": self.prompt,
        }


def format_sources(results: Sequence[dict[str, object]]) -> str:
    """Format retrieved FAQs as grounded context."""

    context = ""
    for result in results:
        context += f"Source: {result['source']}\n"
        context += f"FAQ Question: {result['question']}\n"
        context += f"FAQ Answer: {result['answer']}\n\n"
    return context


def make_prompt(question: str, results: Sequence[dict[str, object]]) -> str:
    """Construct a grounded prompt from retrieved FAQ context."""

    validate_query(question)
    context = format_sources(results)
    prompt = f"""
Use only the FAQ context below to answer the question.

If the answer is not in the FAQ context, say:
I do not have enough information in the FAQ data.

FAQ Context:
{context}

User Question:
{question}

Answer:
"""
    return prompt.strip()


def answer_faq(
    question: str,
    chunks: Sequence[dict[str, object]],
    method: str = "tfidf",
    top_k: int = 3,
    threshold: float = 0.20,
    semantic_model: object | None = None,
) -> dict[str, object]:
    """Retrieve, rerank, threshold, and answer an FAQ question."""

    validate_query(question)

    if method == "tfidf":
        results = search_tfidf(question, chunks, top_k)
    elif method == "semantic":
        results = search_semantic(question, chunks, top_k, model=semantic_model)
    else:
        raise ValueError("method must be 'tfidf' or 'semantic'")

    results = rerank(question, results)
    best = best_result(results)
    prompt = make_prompt(question, results)

    if not passes_threshold(best, threshold):
        answer = FALLBACK_ANSWER
        sources: list[object] = []
    else:
        answer = str(best["answer"])
        sources = [best["source"]]

    return FAQAnswer(
        question=question,
        answer=answer,
        sources=sources,
        best_score=round(float(best["score"]), 3),
        retrieved_faqs=results,
        prompt=prompt,
    ).as_dict()
