import pytest
import os
from dotenv import load_dotenv
from rag.ingestion import DocumentIngestion
from rag.chunking import DocumentProcessor
from rag.embeddings import EmbeddingManager
from rag.retriever import RetrievalService
from langchain_core.documents import Document

load_dotenv()


def test_document_ingestion():
    loader = DocumentIngestion(directory_path="./data/docs")
    docs = loader.load()
    assert isinstance(docs, list)


def test_document_processor():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    test_doc = [Document(page_content="This is a long sentence for testing chunking.", metadata={
                         "source": "test.txt"})]
    chunks = processor.process(test_doc)

    assert len(chunks) > 0
    assert chunks[0].metadata["source"] == "test.txt"


def test_faiss_persistence(tmp_path):
    # Use a temporary directory for test index
    index_dir = str(tmp_path / "test_index")
    embed_manager = EmbeddingManager()
    service = RetrievalService(
        embedding_function=embed_manager.get_embeddings(), index_path=index_dir)

    test_chunks = [Document(page_content="Policy detail A", metadata={
                            "source_file": "doc1.pdf"})]
    service.add_documents(test_chunks)

    # Verify index file was created
    assert os.path.exists(index_dir)

    # Test reloading
    new_service = RetrievalService(
        embedding_function=embed_manager.get_embeddings(), index_path=index_dir)
    assert new_service.vector_store is not None
