TEST_CASES = [
    {
        "question": "I forgot my password",
        "expected_id": 1,
    },
    {
        "question": "I cannot sign into my account",
        "expected_id": 2,
    },
    {
        "question": "I want to cancel my plan",
        "expected_id": 6,
    },
    {
        "question": "Can I get my money back?",
        "expected_id": 9,
    },
    {
        "question": "Why was my card rejected?",
        "expected_id": 10,
    },
]


def recall_at_k(results, expected_id, k=3):
    ids = [
        result["id"]
        for result in results[:k]
    ]

    return int(expected_id in ids)


def reciprocal_rank(results, expected_id):
    for rank, result in enumerate(results, start=1):

        if result["id"] == expected_id:
            return 1 / rank

    return 0


def evaluate(test_cases, retrieve_function, k=3):
    recalls = []
    reciprocal_ranks = []

    for test in test_cases:

        results = retrieve_function(
            test["question"]
        )

        recalls.append(
            recall_at_k(
                results,
                test["expected_id"],
                k
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                results,
                test["expected_id"]
            )
        )

    return {
        f"Recall@{k}":
            sum(recalls) / len(recalls),

        "MRR":
            sum(reciprocal_ranks)
            / len(reciprocal_ranks),
    }
