# NHS Operational Planning RAG Assistant
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![RAG](https://img.shields.io/badge/Architecture-RAG-green)

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

## 🚀 Quick Start — Running the Project

### **1. Clone the Repository**

```bash
git clone https://github.com/Sasikiranmayi/healthcare-policy-rag-assistant.git
cd healthcare-policy-rag-assistant
```

### **2. Create and Activate Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate        # Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Your Environment Variables**

Create a `.env` file:

```bash
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4.1-mini
TEMPERATURE=0
```

### Running the Application

Launch the browser-based UI:

```bash
streamlit run app.py
```
The system is deliberately **modular**, so the same RAG core can later be exposed via FastAPI without refactoring.

### 🧪 Testing & Quality Assurance

To ensure the reliability of the ingestion and retrieval pipeline, run the test suite:

```bash
python -m pytest tests/
```

## 📁 Project Structure

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
```

---

## Core Design Decisions

### 1. Retrieval-Augmented Generation (RAG)
LLMs are powerful language generators but are not reliable sources of factual truth.  
RAG ensures that **answers are grounded strictly in retrieved document context**, which is essential for healthcare-related content.

If the relevant information is not present in the documents, the assistant explicitly states that it does not know.

---

### 2. Local Embeddings (HuggingFace)
I used **local sentence-transformer embeddings** (`all-MiniLM-L6-v2`) instead of managed embedding APIs.

**Why:**
- avoids rate limits and quota issues during indexing
- ensures reviewers can run the project easily
- faster iteration during development
- still provides strong semantic retrieval quality

The embedding layer is abstracted and can be replaced with managed embeddings in production.

---

### 3. FAISS for Vector Search
FAISS is used as a **local vector store**.

**Why:**
- zero external dependencies
- fast and reliable for small-to-medium corpora
- ideal for a take-home assignment focused on correctness and clarity

In a production system, this could be swapped for Pinecone / Weaviate / OpenSearch with minimal changes.

---

### 4. Streamlit for UI
Streamlit is used only as a **lightweight conversational interface**.

**Why:**
- minimal UI effort
- quick feedback loop
- UI is not the evaluation focus per assignment instructions

The backend RAG logic is framework-agnostic.

---

## Safety & Scope Control

Because the domain is healthcare-related:
- The assistant answers **only from retrieved context**
- If context does not contain the answer, it says so
- No medical diagnosis or treatment advice is provided
- The system is informational only

---

## Productionization & Scalability (AWS / GCP / Azure)

To productionize this solution, the following would be required:

### Infrastructure

- Replace FAISS with a managed vector database:

    - AWS OpenSearch / Pinecone / Weaviate

- Deploy API layer using FastAPI behind:

    - AWS ALB / API Gateway

- Containerize using Docker

- Deploy on ECS / EKS / GKE / Azure Container Apps

### Data & Indexing

- Separate offline ingestion pipeline (batch jobs)

- Versioned document storage (S3 / GCS / Blob Storage)

- Background re-indexing and deduplication

### Scalability & Reliability

- Stateless API with external vector DB

- Horizontal scaling of inference layer

- Rate limiting and request throttling

- Caching frequent queries

### Observability

- Structured logging

- Metrics for:

    - retrieval latency

    - embedding latency

    - LLM latency

- Tracing (OpenTelemetry)

- LLM output monitoring

---

## Engineering Standards Followed (and Skipped)
### Followed

- Clear separation of concerns

- Type hints where useful

- Logging for key steps

- Basic automated tests

- Deterministic behavior

### Skipped (Intentionally)

- Full CI/CD pipeline

- Distributed tracing

- Advanced prompt templating frameworks

- Multi-tenant support

These were consciously skipped to stay within scope.

---

## 📈 Roadmap & Future Enhancements
- Add FastAPI-based API layer

- Add structured citations (page, section)

- Introduce reranking / MMR retrieval

- Add evaluation harness (golden Q&A)

- Improve document preprocessing (tables, sections)

- Add async ingestion pipeline

---

## Final Note

This project is intentionally scoped to demonstrate:

- end-to-end RAG correctness

- thoughtful engineering trade-offs

- clean, maintainable code

- realistic production thinking

- The emphasis is on how the system is designed and why, not on UI polish or feature volume.

---

## References / Data Sources

This assistant is built over publicly available NHS guidance documents and supporting notes placed under `data/docs/`.

Primary reference used in this repo:

- **NHS England – 2025/26 Priorities and Operational Planning Guidance (January 2025)**  
  Source (PDF): https://www.england.nhs.uk/wp-content/uploads/2023/04/PRN01625-25-26-priorities-and-operational-planning-guidance-january-2025.pdf

Notes:
- Documents are used for **informational** question answering only.
- Any interpretations are generated by the model and should be validated against the original source.

