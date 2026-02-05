import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text embeddings using OpenAI.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set")

        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=self.model_name)
        logger.info(
            f"Initialized EmbeddingManager with model: {self.model_name}")

    def get_embeddings(self) -> Embeddings:
        return self.embeddings
