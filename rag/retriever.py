import logging
from typing import List
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Manages the Chroma vector database and retrieval logic.
    """

    def __init__(self, embeddings, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            collection_name="policy_docs",
            embedding_function=embeddings,
            persist_directory=self.persist_directory
        )
        logger.info(f"Vector store initialized at {self.persist_directory}")

    def add_documents(self, chunks: List[Document]):
        """Adds processed document chunks to the vector store."""
        if chunks:
            self.vector_store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks to vector store.")

    def get_retriever(self, k: int = 5):
        """
        Returns a retriever object for use in a RAG chain.
        The retriever will return the top k most relevant chunks for a given query.

        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})
