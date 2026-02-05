import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Split documents into overlapping chunks suitable for RAG.

    Design considerations:
    - Chunk size balances semantic coherence vs retrieval granularity
    - Overlap preserves context across chunk boundaries
    - Metadata is preserved for citations
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,  # Crucial for citation tracking
            # Important for handling both PDFs and Markdown files
            separators=["\n\n", "\n", " ", ""],
        )

    def process(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents provided for processing.")
            return []

        logger.info("Processing %d documents into chunks...", len(documents))

        chunks = self.text_splitter.split_documents(documents)

        logger.info("Successfully created %d chunks (Avg chunk size: %d)",
                    len(chunks),
                    sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0)

        return chunks
