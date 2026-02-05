import os
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Manages the FAISS vector store and retrieval logic.
    """

    def __init__(self, embedding_function, index_path: str = "./faiss_index"):
        self.index_path = index_path
        self.embedding_function = embedding_function
        self.vector_store = None

        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_function,
                    allow_dangerous_deserialization=True  # Required for loading local pickle files
                )
                logger.info("Existing FAISS index loaded from %s",
                            self.index_path)
            except Exception as e:
                logger.error("Failed to load FAISS index: %s", str(e))

    def add_documents(self, chunks: List[Document]):
        """
        Adds document chunks to the FAISS index. 
        Creates a new index if one doesn't exist.
        """
        if not chunks:
            return

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                chunks, self.embedding_function)
            logger.info(
                "Initialized new FAISS index with %d chunks.", len(chunks))
        else:
            self.vector_store.add_documents(chunks)
            logger.info(
                "Added %d chunks to existing FAISS index.", len(chunks))

        self.vector_store.save_local(self.index_path)
        logger.info("FAISS index saved locally at %s", self.index_path)

    def get_retriever(self, k: int = 5):
        """Returns a retriever object for use in a RAG chain."""
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Add documents first.")

        return self.vector_store.as_retriever(search_kwargs={"k": k})
