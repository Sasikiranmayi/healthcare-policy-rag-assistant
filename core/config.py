import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables ONCE, at startup
load_dotenv()


@dataclass(frozen=True)
class Settings:
    DOCS_PATH: str = os.getenv("DOCS_PATH", "data/docs")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "4"))

    # Embeddings (local)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
