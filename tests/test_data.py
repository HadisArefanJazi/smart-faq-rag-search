import pandas as pd
import pytest

from smart_faq.data import clean_text, load_faqs, make_faq_chunks


def test_clean_text():
    assert clean_text("  Reset   PASSWORD  ") == "reset password"


def test_load_and_make_chunks(tmp_path):
    path = tmp_path / "faqs.csv"

    pd.DataFrame([
        {
            "id": 1,
            "question": "Reset password?",
            "answer": "Use Forgot Password.",
            "category": "account",
        }
    ]).to_csv(path, index=False)

    df = load_faqs(path)
    chunks = make_faq_chunks(df)

    assert len(chunks) == 1
    assert chunks[0]["id"] == 1
    assert chunks[0]["source"] == "account"
    assert chunks[0]["text"] == "reset password? use forgot password."


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "bad.csv"

    pd.DataFrame([
        {"id": 1, "question": "Question"}
    ]).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_faqs(path)


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"

    pd.DataFrame([
        {"id": 1, "question": "Q1", "answer": "A1", "category": "a"},
        {"id": 1, "question": "Q2", "answer": "A2", "category": "b"},
    ]).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_faqs(path)
