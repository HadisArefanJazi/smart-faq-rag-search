"""FAQ data loading, validation, cleaning, and chunk preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

FAQ_COLUMNS = ("id", "question", "answer", "category")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "raw" / "faqs.csv"
SAMPLE_DATA_PATH = REPO_ROOT / "data" / "sample_faqs.csv"


@dataclass(frozen=True)
class FAQChunk:
    """Prepared FAQ record used by retrieval and prompting."""

    id: int
    source: str
    question: str
    answer: str
    text: str

    def as_dict(self) -> dict[str, object]:
        """Return the dictionary shape expected by the retrieval pipeline."""

        return {
            "id": self.id,
            "source": self.source,
            "question": self.question,
            "answer": self.answer,
            "text": self.text,
        }


def clean_text(text: object) -> str:
    """Lowercase text, trim it, and collapse repeated whitespace."""

    return " ".join(str(text).lower().strip().split())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"FAQ dataset not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"FAQ dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"FAQ dataset is malformed: {path}") from exc


def validate_faqs(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required FAQ columns and non-empty question/answer/category data."""

    if df.empty:
        raise ValueError("FAQ data is empty.")

    required_columns = set(FAQ_COLUMNS)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    prepared = df.loc[:, FAQ_COLUMNS].copy()
    for column in FAQ_COLUMNS:
        if prepared[column].isna().any():
            raise ValueError(f"FAQ data contains missing values in column: {column}")

    for column in ("question", "answer", "category"):
        if prepared[column].map(lambda value: clean_text(value) == "").any():
            raise ValueError(f"FAQ data contains empty values in column: {column}")

    try:
        prepared["id"] = prepared["id"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("FAQ id values must be integers.") from exc

    return prepared


def load_faqs(path: str | Path | None = None, *, allow_sample_fallback: bool = True) -> pd.DataFrame:
    """Load FAQ rows from a CSV file, falling back to the sample data by default."""

    data_path = Path(path) if path else DEFAULT_DATA_PATH
    if data_path.exists():
        df = _read_csv(data_path)
    elif allow_sample_fallback:
        df = _read_csv(SAMPLE_DATA_PATH)
    else:
        raise FileNotFoundError(f"FAQ dataset not found: {data_path}")

    return validate_faqs(df)


def make_faq_chunks(df: pd.DataFrame) -> list[dict[str, object]]:
    """Prepare FAQ rows as retrieval chunks."""

    prepared = validate_faqs(df)
    chunks: list[dict[str, object]] = []

    for _, row in prepared.iterrows():
        chunk = FAQChunk(
            id=int(row["id"]),
            source=str(row["category"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            text=f"{row['question']} {row['answer']}",
        )
        chunks.append(chunk.as_dict())

    return chunks


def add_faq_to_frame(
    faqs: pd.DataFrame,
    question: str,
    answer: str,
    category: str,
) -> pd.DataFrame:
    """Return a new dataframe with one FAQ appended and the next sequential ID."""

    if clean_text(question) == "":
        raise ValueError("question must not be empty.")
    if clean_text(answer) == "":
        raise ValueError("answer must not be empty.")
    if clean_text(category) == "":
        raise ValueError("category must not be empty.")

    prepared = validate_faqs(faqs)
    new_id = int(prepared["id"].max()) + 1
    new_row = pd.DataFrame(
        [
            {
                "id": new_id,
                "question": question,
                "answer": answer,
                "category": category,
            }
        ]
    )
    return pd.concat([prepared, new_row], ignore_index=True)


def ensure_chunks(chunks: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Validate that retrieval has non-empty prepared chunks."""

    if not chunks:
        raise ValueError("FAQ chunks are empty.")
    required = {"id", "source", "question", "answer", "text"}
    for index, chunk in enumerate(chunks):
        missing = required.difference(chunk)
        if missing:
            raise ValueError(f"FAQ chunk {index} is missing keys: {sorted(missing)}")
    return list(chunks)
