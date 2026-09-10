import argparse

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from smart_faq.data import (
    load_faqs,
    make_faq_chunks,
)

from smart_faq.retrieval import (
    build_bm25,
    build_embeddings,
    retrieve,
    rerank,
    passes_threshold,
)

from smart_faq.evaluation import (
    TEST_CASES,
    evaluate,
)


FALLBACK = "I do not have enough information."


def answer_question(
    query,
    chunks,
    bm25,
    embedding_model,
    chunk_embeddings,
    reranker,
    method="hybrid",
    top_k=3,
    threshold=0.30,
    alpha=0.5,
):
    candidates = retrieve(
        query=query,
        chunks=chunks,
        bm25=bm25,
        embedding_model=embedding_model,
        chunk_embeddings=chunk_embeddings,
        method=method,
        top_k=10,
        alpha=alpha,
    )

    results = rerank(
        query,
        candidates,
        reranker,
        top_k=top_k,
    )

    best = results[0]

    if not passes_threshold(
        best,
        threshold
    ):
        return {
            "answer": FALLBACK,
            "source": None,
            "results": results,
        }

    return {
        "answer": best["answer"],
        "source": best["source"],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["bm25", "semantic", "hybrid"],
        default="hybrid",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
    )

    args = parser.parse_args()

    # Load data
    df = load_faqs()
    chunks = make_faq_chunks(df)

    # Models
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Build indexes once
    bm25 = build_bm25(chunks)

    chunk_embeddings = build_embeddings(
        chunks,
        embedding_model,
    )

    # Evaluation
    if args.evaluate:

        def retrieve_for_eval(query):
            return retrieve(
                query,
                chunks,
                bm25,
                embedding_model,
                chunk_embeddings,
                method=args.method,
                top_k=10,
            )

        metrics = evaluate(
            TEST_CASES,
            retrieve_for_eval,
            k=3,
        )

        print(metrics)
        return

    # Interactive user input
    print("Ask a question. Type 'quit' to stop.")

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in {
            "quit",
            "exit",
        }:
            break

        if not query:
            continue

        response = answer_question(
            query,
            chunks,
            bm25,
            embedding_model,
            chunk_embeddings,
            reranker,
            method=args.method,
        )

        print(
            "\nAnswer:",
            response["answer"]
        )

        print(
            "Source:",
            response["source"]
        )


if __name__ == "__main__":
    main()
