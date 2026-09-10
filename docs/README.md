# Extension Notes

## Long-Document Chunking

The current project uses FAQ rows as retrieval units.

```text
1 FAQ row → 1 retrieval chunk
```

Because each FAQ is already short, the current implementation does not need a chunk size or overlap!

For the extended version using long documents such as PDFs, articles, or documentation, the documents should first be split into smaller chunks.

Example:

```text
Long document
     ↓
clean text
     ↓
split into chunks
     ↓
chunk 1
chunk 2
chunk 3
     ↓
BM25 / embeddings / hybrid retrieval
```

Typical parameters would be:

```python
chunk_size = 200
overlap = 40
```

`chunk_size` controls how much text is placed in each chunk.

`overlap` repeats some text between neighboring chunks so important context is not lost at chunk boundaries.

Example Python implementation:

```python
def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size."
        )

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks
```

Example:

```python
chunks = chunk_text(
    document_text,
    chunk_size=200,
    overlap=40,
)
```

For production RAG systems, chunking is often token-based, sentence-aware, or recursive rather than based only on words :)