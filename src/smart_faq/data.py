from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "sample_faqs.csv"

FAQ_COLUMNS = ["id", "question", "answer", "category"]


def clean_text(text):
    return " ".join(str(text).lower().strip().split())


def load_faqs(path=DEFAULT_DATA_PATH):
    df = pd.read_csv(path)

    missing = set(FAQ_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("FAQ dataset is empty.")

    if df[FAQ_COLUMNS].isnull().any().any():
        raise ValueError("FAQ dataset contains missing values.")

    df["id"] = df["id"].astype(int)

    if df["id"].duplicated().any():
        raise ValueError("FAQ IDs must be unique.")

    return df


def make_faq_chunks(df):
    chunks = []

    for _, row in df.iterrows():
        chunks.append({
            "id": int(row["id"]),
            "source": str(row["category"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "text": clean_text(
                f"{row['question']} {row['answer']}"
            ),
        })

    return chunks
