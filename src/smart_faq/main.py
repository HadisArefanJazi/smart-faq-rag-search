"""Command-line entry point for the Smart FAQ RAG search demo."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from smart_faq.data_loader import DEFAULT_DATA_PATH, load_faqs, make_faq_chunks
from smart_faq.evaluation import evaluate, format_evaluation
from smart_faq.prompting import answer_faq

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES_PATH = REPO_ROOT / "examples" / "sample_queries.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Smart FAQ RAG search")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--method", choices=["tfidf", "semantic"], default="tfidf")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--examples-path", type=str, default=str(DEFAULT_EXAMPLES_PATH))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def load_example_questions(path: str | Path = DEFAULT_EXAMPLES_PATH) -> list[str]:
    """Load demonstration questions from JSON."""

    example_path = Path(path)
    if not example_path.exists():
        raise FileNotFoundError(f"Examples file not found: {example_path}")

    try:
        data = json.loads(example_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Examples file is malformed JSON: {example_path}") from exc

    if not isinstance(data, list) or not all(isinstance(item, str) and item.strip() for item in data):
        raise ValueError("Examples file must be a JSON list of non-empty strings.")

    return data


def format_response(response: dict[str, object]) -> str:
    """Format a response for the command-line demonstration."""

    lines = [
        "=" * 70,
        "Question:",
        str(response["question"]),
        "",
        "Answer:",
        str(response["answer"]),
        "",
        "Sources:",
        str(response["sources"]),
        "",
        "Best score:",
        str(response["best_score"]),
        "",
        "Retrieved FAQs:",
    ]

    for faq in response["retrieved_faqs"]:
        lines.extend(
            [
                "-" * 50,
                f"Source: {faq['source']}",
                f"Question: {faq['question']}",
                f"Answer: {faq['answer']}",
                f"Score: {round(float(faq['score']), 3)}",
            ]
        )

    return "\n".join(lines)


def run_demo(args: argparse.Namespace) -> None:
    """Run the default command-line demonstration."""

    faqs = load_faqs(args.data_path)
    chunks = make_faq_chunks(faqs)

    if args.question:
        questions = [args.question]
    else:
        questions = load_example_questions(args.examples_path)

    for question in questions:
        response = answer_faq(
            question,
            chunks,
            method=args.method,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        LOGGER.info("%s", format_response(response))

    if not args.question:
        LOGGER.info("")
        LOGGER.info("Grounded Prompt Example:")
        response = answer_faq("how do I reset my password?", chunks, method=args.method)
        LOGGER.info("%s", response["prompt"])

    if args.evaluate or not args.question:
        LOGGER.info("")
        LOGGER.info("Evaluation:")
        LOGGER.info("%s", format_evaluation(evaluate(chunks, method=args.method)))

    if args.method == "tfidf":
        LOGGER.info("")
        LOGGER.info("Semantic search is optional.")
        LOGGER.info("Install it with:")
        LOGGER.info("pip install sentence-transformers")
        LOGGER.info("")
        LOGGER.info("Then test:")
        LOGGER.info("python -m smart_faq.main --method semantic --question 'how do I cancel my plan?'")


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    run_demo(args)


if __name__ == "__main__":
    main()
