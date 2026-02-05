import logging
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages local text embeddings using HuggingFace models.

    Uses a lightweight, high-quality sentence-transformer model
    suitable for RAG use cases.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set")

        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model=self.model_name)
        logger.info(
            f"Initialized EmbeddingManager with model: {self.model_name}")

    def get_embeddings(self) -> Embeddings:
        return self.embeddings
