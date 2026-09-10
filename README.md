# Smart FAQ Hybrid Retrieval System

A lightweight FAQ retrieval project demonstrating modern RAG retrieval techniques.

The system retrieves the most relevant FAQ using sparse and dense retrieval, reranks candidates with a CrossEncoder, and returns either a grounded FAQ answer or a fallback when confidence is low.

## Retrieval Pipeline

```text
User Question
      ↓
BM25 sparse retrieval
      +
Dense embeddings
      ↓
Hybrid retrieval
      ↓
Top-K candidates
      ↓
CrossEncoder reranking
      ↓
Confidence threshold
      ↓
Answer + source
```

## Features

- FAQ loading and validation
- Text cleaning and chunk preparation
- BM25 sparse retrieval
- SentenceTransformer dense embeddings
- Hybrid sparse + dense retrieval
- Configurable hybrid weight (`alpha`)
- CrossEncoder reranking
- Top-K retrieval
- Confidence threshold
- Source/provenance tracking
- Recall@K evaluation
- Mean Reciprocal Rank (MRR)
- Interactive command-line user input

## Repository Structure

```text
smart-faq-rag-search/
├── src/
│   └── smart_faq/
│       ├── __init__.py
│       ├── data.py
│       ├── retrieval.py
│       ├── evaluation.py
│       └── main.py
├── data/
│   └── sample_faqs.csv
├── examples/
│   └── sample_queries.json
├── tests/
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
python -m pip install -e .
```

## Run

```bash
python -m smart_faq.main
```

Then enter questions interactively.

Compare retrieval methods:

```bash
python -m smart_faq.main --method bm25
python -m smart_faq.main --method semantic
python -m smart_faq.main --method hybrid
```

Run retrieval evaluation:

```bash
python -m smart_faq.main --evaluate
```

## Evaluation

The project evaluates retrieval using:

- Recall@K — whether the correct FAQ appears in the top K results
- MRR — how highly the correct FAQ is ranked

## Current Limitations

- Small demonstration FAQ dataset
- No external LLM generation
- No vector database
- Designed as an interpretable retrieval/RAG experiment rather than a production service
