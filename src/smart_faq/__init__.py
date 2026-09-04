"""Smart FAQ retrieval and grounded prompt construction."""

from smart_faq.data_loader import (
    faq_columns,
    faq_chunk,
    clean_text,
    load_faqs,
    make_faq_chunks,
)

from smart_faq.prompting import faq_answer, answer_faq, make_prompt
from smart_faq.retrieval import search_semantic, search_tfidf
from smart_faq.reranking import rerank

__all__ = [
    "faq_answer",
    "faq_chunk",
    "faq_columns",
    "answer_faq",
    "clean_text",
    "load_faqs",
    "make_faq_chunks",
    "make_prompt",
    "rerank",
    "search_semantic",
    "search_tfidf",
]
