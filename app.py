import os
import logging
from dotenv import load_dotenv
from regex import template
from rag import retriever
from rag.ingestion import DocumentIngestion
from rag.chunking import DocumentProcessor
from rag.embeddings import EmbeddingManager
from rag.retriever import RetrievalService
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure Logging for Observability
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    load_dotenv()  # Load OpenAI API Key from .env

    # 1. Ingestion & Chunking
    loader = DocumentIngestion(directory_path="./data/docs")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

    logger.info("Starting ingestion...")
    raw_docs = loader.load()
    chunks = processor.process(raw_docs)

    # 2. Embedding & Retrieval Setup
    embedding_manager = EmbeddingManager()
    embedding_function = embedding_manager.get_embeddings()
    retriever_service = RetrievalService(embedding_function=embedding_function)

    # 3. Add to Vector DB (Only need to do this once if persistent)
    retriever_service.add_documents(chunks)
    retriever = retriever_service.get_retriever(k=5)
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)

    # Build text context
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    # 4. RAG Chain Construction
    template = """
        You are an AI assistant that helps answer questions based on the following context {context} from a collection of documents.
        Use the provided context to answer the question as accurately as possible. If the context does not contain the answer, say you don't know.
        Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm_chain = prompt | llm
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",  # "Stuffs" all retrieved context into the prompt
    #     retriever=retriever_service.get_retriever()
    # )

    # Example Query
    query = "Summarize the key strategic goals mentioned in the documents."
    response = llm_chain.invoke({"context": context, "question": query})
    print(f"\nAI Response: {response.content}")


if __name__ == "__main__":
    main()
