"""
CHECKPOINT 1 — Embeddings: What are they?

An embedding is a list of numbers (a vector) that represents the *meaning*
of a piece of text. The key property: similar meanings → similar vectors.

This is the foundation of RAG. Before we store or retrieve anything,
we need to understand what we're actually storing.

Run this file and follow the printed output.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


# ─── Load the embedding model ────────────────────────────────────────────────
# This downloads ~90MB on first run, then caches locally.
# 'all-MiniLM-L6-v2' is small, fast, and good enough for learning.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")


# ─── PART 1: What does an embedding look like? ───────────────────────────────
sentence = "The cat sat on the mat."
embedding = model.encode(sentence)

print("=" * 60)
print("PART 1: What is an embedding?")
print("=" * 60)
print(f"\nSentence : '{sentence}'")
print(f"Embedding shape : {embedding.shape}")   # (384,) — 384 numbers
print(f"First 8 values  : {embedding[:8].round(4)}")
print(f"Data type       : {embedding.dtype}")
print("""
Each sentence becomes a point in 384-dimensional space.
Meaning is encoded in which dimensions are activated and by how much.
You can't interpret individual numbers — the *distances between vectors*
are what matter.
""")


# ─── PART 2: Cosine similarity ────────────────────────────────────────────────
def cosine_similarity(a, b):
    """
    Measures the angle between two vectors.
    1.0  = identical direction (same meaning)
    0.0  = perpendicular (unrelated)
   -1.0  = opposite directions (opposite meaning)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("=" * 60)
print("PART 2: Cosine similarity")
print("=" * 60)

sentence_pairs = [
    # (sentence_a, sentence_b, expected_relationship)
    (
        "I love programming in Python.",
        "Python is my favorite programming language.",
        "HIGH similarity — same topic, different words",
    ),
    (
        "I love programming in Python.",
        "The stock market crashed yesterday.",
        "LOW similarity — completely unrelated",
    ),
    (
        "The bank by the river flooded.",
        "She deposited money at the bank.",
        "MEDIUM — same word 'bank', different meanings",
    ),
]

for a, b, label in sentence_pairs:
    vec_a = model.encode(a)
    vec_b = model.encode(b)
    score = cosine_similarity(vec_a, vec_b)
    print(f"\n  A: \"{a}\"")
    print(f"  B: \"{b}\"")
    print(f"  Similarity: {score:.4f}  ← {label}")

print()


# ─── PART 3: The "king - man + woman" intuition ──────────────────────────────
print("=" * 60)
print("PART 3: Vector arithmetic (the famous analogy test)")
print("=" * 60)
print("""
If embeddings encode meaning, then meaning should be *composable*.
  king - man + woman  ≈  queen  (in theory)

Let's test this with our model (MiniLM is small, so results vary):
""")

words = ["king", "man", "woman", "queen", "prince", "princess"]
vecs = {w: model.encode(w) for w in words}

analogy_result = vecs["king"] - vecs["man"] + vecs["woman"]

scores = {
    w: cosine_similarity(analogy_result, vecs[w])
    for w in words
    if w not in ("king", "man", "woman")
}

print("  king - man + woman  →  closest matches:")
for word, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"    {word:<12} {score:.4f}")

print()


# ─── PART 4: Batch encoding (what we'll do at scale) ─────────────────────────
print("=" * 60)
print("PART 4: Batch encoding — encoding many sentences at once")
print("=" * 60)
print("""
In a real RAG pipeline, you embed hundreds or thousands of text chunks.
Batch encoding does this efficiently in one call.
""")

facts = [
    "Python was created by Guido van Rossum.",
    "The Eiffel Tower is in Paris, France.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Neural networks are inspired by the human brain.",
    "The Great Wall of China is visible from space.",  # actually a myth!
]

embeddings = model.encode(facts)  # shape: (5, 384)
print(f"  Encoded {len(facts)} sentences → matrix shape: {embeddings.shape}")
print(f"  Each row is one sentence's embedding.\n")

# Now simulate a query against those facts
query = "Who invented Python?"
query_vec = model.encode(query)

print(f"  Query: \"{query}\"")
print(f"  Ranking facts by relevance:\n")

scores = [(fact, cosine_similarity(query_vec, emb)) for fact, emb in zip(facts, embeddings)]
scores.sort(key=lambda x: -x[1])

for rank, (fact, score) in enumerate(scores, 1):
    print(f"  #{rank}  [{score:.4f}]  {fact}")

print("""
Notice how the most relevant fact lands at #1 even though the query
and the fact use different words ("invented" vs "created").
The embedding captured the semantic relationship.

─────────────────────────────────────────────────────────────
CHECKPOINT 1 COMPLETE

What you just saw:
  ✓ An embedding is a fixed-size vector representing text meaning
  ✓ Cosine similarity measures how "close" two meanings are
  ✓ Embeddings are language-aware — synonym-aware, context-aware
  ✓ Batch encoding produces a matrix of embeddings (chunks × dimensions)

What's missing: we're computing similarity by looping over every fact.
With millions of documents, that's too slow. We need a vector database.

→ That's Checkpoint 2: chunking documents + storing embeddings in ChromaDB.
─────────────────────────────────────────────────────────────
""")
