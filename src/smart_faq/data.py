"""FAQ data loading, validation, cleaning, and chunk preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


faq_columns = ("id", "question", "answer", "category")

repo_root = Path(__file__).resolve().parents[2]
default_data_path = repo_root / "data" / "raw" / "faqs.csv"
sample_data_path = repo_root / "data" / "sample_faqs.csv"


@dataclass(frozen=True)
class faq_chunk:
    id: int
    source: str
    question: str
    answer: str
    text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "question": self.question,
            "answer": self.answer,
            "text": self.text,
        }


def clean_text(text: object) -> str:
    return " ".join(str(text).lower().strip().split())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"FAQ dataset not found: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"FAQ dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"FAQ dataset is malformed: {path}") from exc


def validate_faqs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("FAQ data is empty.")

    missing_columns = set(faq_columns).difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    prepared = df.loc[:, faq_columns].copy()

    for column in faq_columns:
        if prepared[column].isna().any():
            raise ValueError(f"Missing values in column: {column}")

    for column in ("question", "answer", "category"):
        if prepared[column].map(lambda value: clean_text(value) == "").any():
            raise ValueError(f"Empty values in column: {column}")

    try:
        prepared["id"] = prepared["id"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("FAQ id values must be integers.") from exc

    return prepared


def load_faqs(
    path: str | Path | None = None,
    allow_sample_fallback: bool = True,
) -> pd.DataFrame:

    data_path = Path(path) if path else default_data_path

    if data_path.exists():
        df = read_csv(data_path)
    elif allow_sample_fallback:
        df = read_csv(sample_data_path)
    else:
        raise FileNotFoundError(f"FAQ dataset not found: {data_path}")

    return validate_faqs(df)


def make_faq_chunks(df: pd.DataFrame) -> list[dict[str, object]]:
    prepared = validate_faqs(df)
    chunks = []

    for _, row in prepared.iterrows():
        chunk = faq_chunk(
            id=int(row["id"]),
            source=str(row["category"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            text=f"{row['question']} {row['answer']}",
        )

        chunks.append(chunk.as_dict())

    return chunks


def ensure_chunks(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    if not chunks:
        raise ValueError("FAQ chunks are empty.")

    required = {"id", "source", "question", "answer", "text"}

    for index, chunk in enumerate(chunks):
        missing = required.difference(chunk)

        if missing:
            raise ValueError(f"FAQ chunk {index} is missing keys: {sorted(missing)}")

    return chunks
