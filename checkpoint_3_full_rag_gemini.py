"""
CHECKPOINT 3 — Full RAG Pipeline (Gemini version)

Same pipeline as checkpoint_3_full_rag.py, but the generation step
uses Google's Gemini instead of Claude.

Everything up to the generation step is identical:
  - ChromaDB for vector storage
  - sentence-transformers for embeddings
  - same chunking and retrieval logic

Only the LLM call changes. This is intentional — in a real RAG system,
the vector DB and embedding model are independent of the LLM you choose.

Before running: set your API key in the terminal:
  export GOOGLE_API_KEY="your-key-here"   (Mac/Linux)
  set GOOGLE_API_KEY=your-key-here        (Windows CMD)

Get a key at: https://aistudio.google.com/app/apikey
"""

import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

DOCUMENT_PATH = "sample_document.txt"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_document"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash"


# ─── Configure Gemini ─────────────────────────────────────────────────────────
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set.\n"
        "Get a key at https://aistudio.google.com/app/apikey, then run:\n"
        "  export GOOGLE_API_KEY='your-key-here'"
    )

genai.configure(api_key=api_key)
gemini = genai.GenerativeModel(GEMINI_MODEL)


# ─── Phase A: Ingestion ───────────────────────────────────────────────────────
print("=" * 60)
print("PHASE A: Ingestion")
print("=" * 60)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
existing = [c.name for c in chroma_client.list_collections()]

if COLLECTION_NAME in existing:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"  Collection '{COLLECTION_NAME}' already exists with "
          f"{collection.count()} chunks. Skipping ingestion.\n")
else:
    print("  Building collection for the first time...")

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    def chunk_text(text, chunk_size=300, overlap=50):
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            chunks.append(" ".join(words[start : start + chunk_size]))
            start += chunk_size - overlap
        return chunks

    chunks = chunk_text(full_text)
    print(f"  Document → {len(chunks)} chunks")

    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"  Embedding {len(chunks)} chunks...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True)

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"  Stored {collection.count()} chunks in '{CHROMA_PATH}'\n")


# ─── Phase B: Q&A Pipeline ───────────────────────────────────────────────────
print("=" * 60)
print("PHASE B: Retrieval + Generation")
print("=" * 60)

embed_model = SentenceTransformer(EMBED_MODEL)


def retrieve(question: str, n_results: int = 3) -> list[str]:
    """Embed the question and return the top-n most relevant chunks."""
    query_vec = embed_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=["documents", "distances"],
    )
    docs = results["documents"][0]
    distances = results["distances"][0]

    print(f"\n  [RETRIEVE] Query: \"{question}\"")
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        print(f"    Chunk #{i+1} (similarity: {1 - dist:.3f}): "
              f"{doc[:100].replace(chr(10), ' ')}...")
    return docs


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    print(f"""
    Use the following context to answer the question.
If the answer is not in the context, say so — do not make things up.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""")
    return f"""Use the following context to answer the question.
If the answer is not in the context, say so — do not make things up.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def generate(question: str, context_chunks: list[str]) -> str:
    """Call Gemini with the retrieved context, streaming the response."""
    prompt = build_prompt(question, context_chunks)

    print(f"\n  [GENERATE] Calling Gemini ({GEMINI_MODEL}) with context...")
    print(f"  [ANSWER]\n")

    # Gemini streaming: generate_content with stream=True returns an iterator.
    # Each chunk has a .text attribute with the partial response.
    # Usage metadata (token counts) is on the last chunk.
    response = gemini.generate_content(prompt, stream=True)

    full_text = []
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_text.append(chunk.text)

    # Usage metadata is available after the stream is exhausted
    usage = response.usage_metadata
    print(f"\n\n  [USAGE] Input tokens: {usage.prompt_token_count} | "
          f"Output tokens: {usage.candidates_token_count}")

    return "".join(full_text)


# ─── Demo: Three questions ────────────────────────────────────────────────────
demo_questions = [
    "What is dropout and when is it used?",
    "Explain how multi-head attention works.",
    "What are the two phases of a RAG pipeline?",
]

for question in demo_questions:
    print("\n" + "─" * 60)
    chunks = retrieve(question)
    generate(question, chunks)


# ─── The "aha moment": WITH context vs WITHOUT context ───────────────────────
print("\n\n" + "=" * 60)
print("THE AHA MOMENT: Same question, with vs without retrieval")
print("=" * 60)

aha_question = (
    "According to the document, what indexing structure does ChromaDB use, "
    "and why does it matter for RAG?"
)

print(f'Question: "{aha_question}"\n')

# Without RAG
print("─── WITHOUT retrieval (Gemini on its own) ───")
response = gemini.generate_content(aha_question, stream=True)
for chunk in response:
    if chunk.text:
        print(chunk.text, end="", flush=True)
print("\n")

# With RAG
print("─── WITH retrieval (Gemini + your document) ───")
chunks = retrieve(aha_question)
generate(aha_question, chunks)

print("""

─────────────────────────────────────────────────────────────
CHECKPOINT 3 (Gemini) COMPLETE

What changed vs the Claude version:
  - anthropic.Anthropic()          → genai.GenerativeModel()
  - client.messages.stream(...)    → model.generate_content(..., stream=True)
  - stream.text_stream             → iterating response chunks directly
  - final.usage.input_tokens       → response.usage_metadata.prompt_token_count

What stayed the same:
  - ChromaDB retrieval (100% identical)
  - Embedding model (100% identical)
  - Chunking logic (100% identical)
  - Prompt template (100% identical)

The retrieval layer is LLM-agnostic. You can swap Claude for Gemini
(or any other model) without touching a single line of the vector DB code.
─────────────────────────────────────────────────────────────
""")
