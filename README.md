# Minimal RAG System

A working retrieval-augmented generation system demonstrating end-to-end understanding of core RAG concepts.

## Architecture

```
PDF Input
   ↓
Chunking (semantic overlap)
   ↓
Embeddings (SentenceTransformers)
   ↓
Vector Index (FAISS)
   ↓
User Query → Embed → Top-k Retrieval → Prompt Construction → LLM Output
```

## Why Each Component

**Chunking** - PDFs are too large for embeddings. Overlap ensures context isn't lost at boundaries.

**SentenceTransformers** - Lightweight, semantic embeddings. Better than sparse methods (TF-IDF) for understanding meaning.

**FAISS** - Fast similarity search at scale. L2 distance finds closest chunks in embedding space.

**Prompt Construction** - Grounding answers in retrieved text prevents hallucination. System prompt controls behavior.

**OpenAI API** - Cheap inference. Can be swapped for Ollama (local).

## Setup

```bash
pip install openai sentence-transformers faiss-cpu PyPDF2 fastapi uvicorn streamlit
```

Set your OpenAI key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

**1. Ingest PDFs**
Place PDF files in `data/` folder, then:
```bash
python ingest.py
```
Creates FAISS index + embeddings.

**2. Test Retrieval**
```bash
python retrieve.py
```
Verifies index was built correctly.

**3. Run Streamlit UI**
```bash
streamlit run streamlit_app.py
```
Interactive Q&A interface.

**4. Or use FastAPI**
```bash
python fastapi_app.py
```
Then POST to `http://localhost:8000/ask`:
```json
{"question": "What is...", "top_k": 3}
```

## Key Decisions

- **Chunk size 500** - Balance between context loss and embedding efficiency
- **Overlap 100** - Prevents semantic breaks at chunk boundaries
- **Top-k=3** - Enough context without token bloat
- **Temperature 0.2** - Keep answers grounded in retrieved text (not creative)
- **all-MiniLM-L6-v2** - Fast, good quality, works offline

## What This Demonstrates

✓ Understanding of chunking strategies
✓ Embedding space and similarity search
✓ Prompt engineering for grounding
✓ Integration across components
✓ Multiple interfaces (CLI, API, UI)

Not production-ready (no caching, no error handling edge cases), but shows core RAG mechanics.
