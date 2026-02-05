# Healthcare Policy RAG Assistant

A lightweight **conversational RAG (Retrieval-Augmented Generation) assistant** for answering questions about **public healthcare policy guidance** (e.g., NHS operational planning priorities). The system ingests mixed document formats (PDF/Markdown), chunks and embeds them, indexes with **FAISS**, retrieves relevant context, and generates answers grounded in the retrieved sources.

This project is intentionally designed to be **simple, clean, and reproducible** for reviewers.

---

## Key Features

- **Conversational chat UI** using **Streamlit**
- **RAG pipeline**: Ingestion → Chunking → Embedding → Indexing → Retrieval → Generation
- **Local embeddings** using `sentence-transformers/all-MiniLM-L6-v2` (no embedding rate limits)
- **FAISS vector search** for fast, local similarity retrieval
- **Source transparency**: Retrieved sources shown in the UI
- **Clean modular code** under `rag/` for easy migration to FastAPI later

---

## Tech Stack

- Python
- Streamlit (simple chat UI)
- LangChain (document loaders, splitters, orchestration)
- SentenceTransformers (local embeddings)
- FAISS (local vector index)
- OpenAI (LLM for response generation)

---

## Project Structure

```text
healthcare-policy-rag-assistant/
├── app.py
├── core/
│   └── config.py
├── rag/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── retriever.py
├── data/
│   └── docs/
│       ├── pdfs/
│       └── text/
├── tests/
│   └── test_rag.py
├── requirements.txt
├── .env    
└── README.md


---

## Setup

### 1. Create and activate a virtual environment

Recommended Python version: **3.11+**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

