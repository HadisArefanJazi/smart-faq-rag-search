import argparse

from smart_faq.data import load_faqs, make_faq_chunks

from smart_faq.retrieval import (
    build_bm25,
    build_embeddings,
    retrieve,
    rerank,
    passes_threshold,
)

from smart_faq.evaluation import TEST_CASES, evaluate


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

    print("\n--- Best Result Score ---")
    print("ID:", best["id"])
    print("Retrieval score:", best.get("score"))
    print("BM25 score:", best.get("bm25_score"))
    print("Dense score:", best.get("dense_score"))
    print("Rerank score:", best.get("rerank_score"))
    print("Threshold:", threshold)
    print("-------------------------")

    if not passes_threshold(best, threshold):
        return {
            "answer": FALLBACK,
            "source": None,
            "id": None,
            "results": results,
        }

    return {
        "answer": best["answer"],
        "source": best["source"],
        "id": best["id"],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Smart FAQ Hybrid Retrieval System"
    )

    parser.add_argument(
        "--method",
        choices=["bm25", "semantic", "hybrid"],
        default="hybrid",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
    )

    args = parser.parse_args()

    df = load_faqs()
    chunks = make_faq_chunks(df)

    bm25 = None
    embedding_model = None
    chunk_embeddings = None

    if args.method in {"bm25", "hybrid"}:
        bm25 = build_bm25(chunks)

    if args.method in {"semantic", "hybrid"}:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        chunk_embeddings = build_embeddings(
            chunks,
            embedding_model,
        )

    if args.evaluate:

        def retrieve_for_eval(query):
            return retrieve(
                query=query,
                chunks=chunks,
                bm25=bm25,
                embedding_model=embedding_model,
                chunk_embeddings=chunk_embeddings,
                method=args.method,
                top_k=len(chunks),
                alpha=args.alpha,
            )

        metrics = evaluate(
            TEST_CASES,
            retrieve_for_eval,
            k=args.top_k,
        )

        print(metrics)
        return

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("Ask a question. Type 'quit' to stop.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in {"quit", "exit"}:
            break

        if not query:
            continue

        response = answer_question(
            query=query,
            chunks=chunks,
            bm25=bm25,
            embedding_model=embedding_model,
            chunk_embeddings=chunk_embeddings,
            reranker=reranker,
            method=args.method,
            top_k=args.top_k,
            threshold=args.threshold,
            alpha=args.alpha,
        )

        print("\nAnswer:", response["answer"])
        print("FAQ ID:", response["id"])
        print("Source:", response["source"])


if __name__ == "__main__":
    main()
