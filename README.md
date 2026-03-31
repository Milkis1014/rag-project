# RAG Learning Project

A hands-on walkthrough I built to understand Retrieval-Augmented Generation (RAG) from scratch. Each checkpoint is a standalone, runnable script that teaches one concept — run it, read the output, then move to the next.

## What is RAG?

Instead of asking an LLM to rely only on its training data, RAG first **retrieves** relevant text from your own documents, then **augments** the LLM's prompt with it, so the answer is grounded in your source material.

```
Your Docs → Chunk → Embed → Store in Vector DB   ← (one-time ingestion)
                                    ↓
User Query → Embed → Search DB → Relevant Chunks → LLM → Answer
```

## Stack

- **sentence-transformers** — local embedding model (`all-MiniLM-L6-v2`), no API key needed
- **ChromaDB** — vector database, runs entirely on disk
- **Anthropic SDK** — Claude as the LLM (checkpoint 3, Claude version)
- **google-generativeai** — Gemini as the LLM (checkpoint 3, Gemini version)

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
source venv/bin/activate        # Mac/Linux
```

**2. Install dependencies**
```bash
pip install sentence-transformers chromadb anthropic google-generativeai
```

**3. Set API keys (use `set` for windows)**

For the Claude version:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

For the Gemini version:
```bash
export GOOGLE_API_KEY="your-key-here"
```

Get keys at:
- Claude: https://console.anthropic.com
- Gemini: https://aistudio.google.com/app/apikey

## Checkpoints

### Checkpoint 1 — Embeddings
```bash
python checkpoint_1_embeddings.py
```
Answers: what is an embedding, what does cosine similarity measure, and why can you do vector arithmetic on meaning (king - man + woman ≈ queen). No database, no LLM — just the embedding model.

### Checkpoint 2 — Chunking + ChromaDB
```bash
python checkpoint_2_chromadb.py
```
Loads `sample_document.txt`, splits it into overlapping chunks, embeds them all, stores them in ChromaDB, and runs semantic search queries against the collection. The retrieval layer — no LLM yet.

### Checkpoint 3 — Full RAG Pipeline
Two versions of the same pipeline, differing only in which LLM handles the generation step.

**Claude version**
```bash
python checkpoint_3_full_rag.py
```

**Gemini version**
```bash
python checkpoint_3_full_rag_gemini.py
```

Both scripts:
1. Ingest `sample_document.txt` into a persistent ChromaDB collection (skipped on re-runs)
2. Retrieve the top-3 most relevant chunks for each query
3. Pass those chunks as context to the LLM and stream the answer

The last section of each script runs the same question **with** and **without** retrieval, side by side — this is the clearest way to see what RAG actually changes about the answer.

## Key insight: the retrieval layer is LLM-agnostic

The fact that checkpoint 3 has two versions (Claude and Gemini) that share identical chunking, embedding, and ChromaDB code is intentional. Swapping the LLM requires changing only the generation function — nothing in the vector database layer changes. In a real system, this means you can swap models without re-ingesting your documents.

## Project structure

```
rag-project/
├── checkpoint_1_embeddings.py       # What embeddings are
├── checkpoint_2_chromadb.py         # Chunking + vector search
├── checkpoint_3_full_rag.py         # Full pipeline with Claude
├── checkpoint_3_full_rag_gemini.py  # Full pipeline with Gemini
├── sample_document.txt              # Source document used across all checkpoints
├── chroma_db/                       # Generated — created when checkpoint 2 or 3 runs
└── venv/                            # Generated — not committed
```
