"""
CHECKPOINT 2 — Chunking + ChromaDB: Store and Retrieve

Now that you understand embeddings, the next problem is scale.
In Checkpoint 1, Part 4, you compared a query against 5 facts using a loop.
Real documents have thousands of chunks. You need a smarter structure.

This checkpoint covers:
  1. Why chunking matters and how to do it with overlap
  2. Storing chunks + embeddings in ChromaDB
  3. Querying ChromaDB and getting ranked results back

Run this file. Observe the output at each part before moving on.
"""

import re
import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENT_PATH = "sample_document.txt"


# ─── Load the document ───────────────────────────────────────────────────────
with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

print(f"Document loaded: {len(full_text)} characters, ~{len(full_text.split())} words\n")


# ─── PART 1: Chunking strategies ─────────────────────────────────────────────
print("=" * 60)
print("PART 1: Why chunking matters")
print("=" * 60)
print("""
You cannot embed an entire document as one vector — you'd lose precision.
Asking "what is dropout?" would match the whole document, not the paragraph
that actually defines it.

You also can't embed each sentence individually — a single sentence often
lacks the surrounding context needed to make sense.

The solution: split the document into overlapping chunks.
Overlap ensures that information spanning a boundary isn't lost.
""")


def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into chunks of approximately `chunk_size` words,
    with `overlap` words shared between consecutive chunks.

    Why words, not characters? Words are more meaningful units.
    A 300-word chunk is roughly one to two paragraphs — enough
    context for an LLM to work with, small enough to be specific.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Advance by (chunk_size - overlap) so the next chunk
        # re-uses the last `overlap` words of the current chunk.
        start += chunk_size - overlap

    return chunks


chunks = chunk_text(full_text, chunk_size=300, overlap=50)

print(f"  Document split into {len(chunks)} chunks")
print(f"  Chunk size: ~300 words | Overlap: ~50 words\n")

# Show the boundary between chunk 0 and chunk 1 to make overlap tangible
words = full_text.split()
boundary_chunk_0_end = " ".join(words[250:300])   # last 50 words of chunk 0
boundary_chunk_1_start = " ".join(words[250:300])  # first 50 words of chunk 1

print("  Last 20 words of chunk 0:")
print(f"    ...{' '.join(chunks[0].split()[-20:])}")
print("\n  First 20 words of chunk 1:")
print(f"    {' '.join(chunks[1].split()[:20])}...")
print("\n  ↑ These overlap. If a concept spans this boundary, both chunks capture it.\n")


# ─── PART 2: Embed all chunks ────────────────────────────────────────────────
print("=" * 60)
print("PART 2: Embedding all chunks")
print("=" * 60)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Embedding {len(chunks)} chunks (this takes a few seconds)...")
embeddings = model.encode(chunks, show_progress_bar=True)

print(f"\nEmbedding matrix shape: {embeddings.shape}")
print(f"  → {embeddings.shape[0]} chunks × {embeddings.shape[1]} dimensions\n")


# ─── PART 3: Store in ChromaDB ───────────────────────────────────────────────
print("=" * 60)
print("PART 3: Storing in ChromaDB")
print("=" * 60)
print("""
ChromaDB is a vector database. It stores your chunk text + embeddings
together, and lets you query by vector similarity instead of keywords.

Under the hood it uses HNSW (Hierarchical Navigable Small World),
a graph-based index that finds approximate nearest neighbors in
O(log n) time — fast even with millions of vectors.

We're using an in-memory client here (data lives only for this run).
ChromaDB also supports persistent storage to disk — we'll use that
in the full pipeline.
""")

# In-memory client — no files written, resets each run.
# Swap to: chromadb.PersistentClient(path="./chroma_db") to persist.
client = chromadb.Client()

# A "collection" is like a table. It holds your vectors + metadata.
collection = client.create_collection(
    name="rag_document",
    metadata={"hnsw:space": "cosine"},  # use cosine similarity (same as Checkpoint 1)
)

# ChromaDB expects:
#   documents — the original text strings
#   embeddings — the vectors (list of lists, not numpy arrays)
#   ids — unique string ID per chunk
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

print(f"  Stored {collection.count()} chunks in ChromaDB collection 'rag_document'")
print()


# ─── PART 4: Query ChromaDB ──────────────────────────────────────────────────
print("=" * 60)
print("PART 4: Querying ChromaDB")
print("=" * 60)
print("""
Now the interesting part: ask a question, get the most relevant chunks back.
ChromaDB embeds your query and computes cosine similarity against every
stored vector — returning the top-k most similar chunks.
""")


def query_collection(question, n_results=3):
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    # results["distances"] uses cosine *distance* (0 = identical, 2 = opposite)
    # Convert to similarity for readability: similarity = 1 - distance
    docs = results["documents"][0]
    distances = results["distances"][0]

    print(f'  Query: "{question}"\n')
    for rank, (doc, dist) in enumerate(zip(docs, distances), 1):
        similarity = 1 - dist
        # Truncate chunk for display
        preview = doc[:200].replace("\n", " ") + "..."
        print(f"  Rank #{rank}  [similarity: {similarity:.4f}]")
        print(f"  {preview}")
        print()


# Test with questions whose answers live in different parts of the document
test_queries = [
    "What is dropout and why is it used?",
    "How does the attention mechanism work in transformers?",
    "What are the two phases of a RAG pipeline?",
    "What happens when the learning rate is too large?",
]

for query in test_queries:
    print("-" * 60)
    query_collection(query)

print()


# ─── PART 5: Observe chunking artifacts ──────────────────────────────────────
print("=" * 60)
print("PART 5: Retrieval quality — what can go wrong")
print("=" * 60)
print("""
Try a query that targets information near a chunk boundary.
Notice whether the retrieved chunk contains the full answer,
or whether overlap helped stitch the context together.
""")

tricky_query = "What indexing structure does ChromaDB use internally?"
print("-" * 60)
query_collection(tricky_query, n_results=2)

print("""
─────────────────────────────────────────────────────────────
CHECKPOINT 2 COMPLETE

What you just built:
  ✓ A chunking function with configurable size and overlap
  ✓ Batch embedding of all chunks into a matrix
  ✓ A ChromaDB collection storing text + vectors
  ✓ Semantic search: question in → ranked relevant chunks out

What you're still missing:
  The retrieved chunks are just text sitting in a variable.
  Nothing is doing anything with them yet. The LLM hasn't
  entered the picture at all.

→ That's Checkpoint 3: plug the retrieved chunks into Claude
  and build the full RAG answer pipeline.
─────────────────────────────────────────────────────────────
""")
