TEST_CASES = [
    {
        "question": "I need a new student ID card",
        "expected_id": 1,
    },
    {
        "question": "I lost my student ID",
        "expected_id": 2,
    },
    {
        "question": "Can I use my ID to enter campus buildings?",
        "expected_id": 3,
    },
    {
        "question": "How do I get permission to park on campus?",
        "expected_id": 4,
    },
    {
        "question": "Where am I allowed to park?",
        "expected_id": 5,
    },
    {
        "question": "What happens if I park without a permit?",
        "expected_id": 6,
    },
    {
        "question": "I want to dispute my parking ticket",
        "expected_id": 7,
    },
    {
        "question": "Can my parents park on campus when they visit?",
        "expected_id": 8,
    },
    {
        "question": "How do I enroll in classes?",
        "expected_id": 9,
    },
    {
        "question": "Why is the system not letting me register for a class?",
        "expected_id": 10,
    },
    {
        "question": "I want to drop one of my courses",
        "expected_id": 11,
    },
    {
        "question": "What does prerequisite mean?",
        "expected_id": 12,
    },
    {
        "question": "I want to switch my major",
        "expected_id": 13,
    },
    {
        "question": "How can I find my academic advisor?",
        "expected_id": 14,
    },
    {
        "question": "I am having difficulty with one of my classes",
        "expected_id": 15,
    },
    {
        "question": "How is my GPA calculated?",
        "expected_id": 16,
    },
    {
        "question": "I need an official copy of my transcript",
        "expected_id": 17,
    },
    {
        "question": "Where can I check my grades?",
        "expected_id": 18,
    },
    {
        "question": "How can I pay my tuition bill?",
        "expected_id": 20,
    },
    {
        "question": "When do I have to pay tuition?",
        "expected_id": 21,
    },
    {
        "question": "How do I apply for financial aid?",
        "expected_id": 23,
    },
    {
        "question": "Where can I see the status of my financial aid?",
        "expected_id": 24,
    },
    {
        "question": "How can I apply for a scholarship?",
        "expected_id": 25,
    },
    {
        "question": "Can I get a job on campus?",
        "expected_id": 26,
    },
    {
        "question": "How do I apply for a dorm room?",
        "expected_id": 27,
    },
    {
        "question": "Can I change my roommate?",
        "expected_id": 28,
    },
    {
        "question": "Something in my dorm room is broken",
        "expected_id": 29,
    },
    {
        "question": "How can I buy a meal plan?",
        "expected_id": 30,
    },
    {
        "question": "Can I use university databases from home?",
        "expected_id": 35,
    },
    {
        "question": "I forgot my university password",
        "expected_id": 36,
    },
    {
        "question": "I cannot connect to campus Wi-Fi",
        "expected_id": 37,
    },
    {
        "question": "How do I access my university email?",
        "expected_id": 38,
    },
    {
        "question": "I need technical support",
        "expected_id": 39,
    },
    {
        "question": "Where can I see a doctor on campus?",
        "expected_id": 40,
    },
    {
        "question": "Does the university have mental health counseling?",
        "expected_id": 41,
    },
    {
        "question": "What should I do during an emergency on campus?",
        "expected_id": 42,
    },
    {
        "question": "How do I request disability accommodations?",
        "expected_id": 44,
    },
    {
        "question": "How do I use the campus shuttle?",
        "expected_id": 45,
    },
    {
        "question": "I need help with my student visa",
        "expected_id": 47,
    },
    {
        "question": "Can someone help me find an internship?",
        "expected_id": 48,
    },
    {
        "question": "How do I apply to graduate?",
        "expected_id": 49,
    },
    {
        "question": "What do I need to complete before graduation?",
        "expected_id": 50,
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
