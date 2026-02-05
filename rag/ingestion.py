# Ingestion Pipeline for RAG (Retrieval-Augmented Generation)
import logging
import os
from typing import List
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """
    A class to load documents from a directory for a RAG pipeline (PDF + Markdown/Text).
    Returns LangChain Document objects with normalized metadata for citations.
    """

    def __init__(self, directory_path: str):
        self.directory_path = directory_path

        # Factory mapping for different file extensions
        self.loaders = {
            ".pdf": PyPDFLoader,
            ".md": TextLoader,
        }

    def load(self) -> List[Document]:
        """
        Scans the directory for supported files and returns a list of Document objects.
        """

        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(
                f"The directory {self.directory_path} does not exist.")

        all_documents = []

        for ext, loader_cls in self.loaders.items():
            # Use DirectoryLoader to recursively find and load files of a specific type
            loader = DirectoryLoader(
                path=self.directory_path,
                glob=f"**/*{ext}",
                loader_cls=loader_cls,
                show_progress=True,
                use_multithreading=True  # Optimizes loading for larger collections
            )

            try:
                docs = loader.load()
                docs = self._normalize_metadata(
                    docs, source_type=ext.lstrip("."))
                all_documents.extend(docs)
                logger.info("Loaded %d documents for %s", len(docs), ext)
            except Exception:
                logger.exception("Failed to load files for extension %s", ext)

        return all_documents

    @staticmethod
    def _normalize_metadata(docs: List[Document], source_type: str) -> List[Document]:
        """
        Ensure consistent metadata across loaders for downstream citations.
        """
        for d in docs:
            # DirectoryLoader typically sets "source" to the file path
            source = d.metadata.get("source", "unknown")
            d.metadata["source_file"] = os.path.basename(source)
            d.metadata["source_type"] = source_type
        return docs
