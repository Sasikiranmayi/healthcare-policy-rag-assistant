import streamlit as st
import os
from core.config import Settings
from rag.ingestion import DocumentIngestion
from rag.chunking import DocumentProcessor
from rag.embeddings import EmbeddingManager
from rag.retriever import RetrievalService
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Healthcare Planning Assistant", page_icon="🏥")
st.title("🏥 Healthcare Operational Planning Chatbot")

settings = Settings()

FAISS_DIR = "./faiss_index"
FAISS_FILES = [os.path.join(FAISS_DIR, "index.faiss"),
               os.path.join(FAISS_DIR, "index.pkl")]


def faiss_index_ready() -> bool:
    return all(os.path.exists(p) for p in FAISS_FILES)


@st.cache_resource
def initialize_rag():
    # 1. Load and Process documents
    loader = DocumentIngestion(directory_path=settings.DOCS_PATH)
    processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    raw_docs = loader.load()
    chunks = processor.process(raw_docs)

    # 2. Setup Vector Store (FAISS + HuggingFace)
    embed_manager = EmbeddingManager()
    retrieval_service = RetrievalService(
        embedding_function=embed_manager.get_embeddings(),
        index_path=FAISS_DIR
    )

    # Build index only if missing
    if not faiss_index_ready():
        retrieval_service.add_documents(chunks)

    retriever = retrieval_service.get_retriever(k=4)

    template = """
        You are a healthcare policy and public guidance assistant.
        Answer ONLY using the provided Context.
        If the Context does not contain the answer, say you cannot find it in the provided sources.
        Do NOT provide medical diagnosis or treatment advice.

        Conversation (recent):
        {history}

        Context:
        {context}

        Question: {question}
        Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    chain = prompt | llm

    return retriever, chain


retriever, chain = initialize_rag()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about healthcare policy...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Retrieve context
        retrieved_docs = retriever.invoke(user_input)
        context = "\n\n".join(d.page_content for d in retrieved_docs)

        # Build recent history (last 6 messages)
        recent = st.session_state.messages[-6:]
        history = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in recent])

        response = chain.invoke({
            "history": history,
            "context": context,
            "question": user_input
        })

        st.markdown(response.content)

        # Optional: show sources (helps trust + evaluation)
        with st.expander("Sources"):
            for d in retrieved_docs:
                md = d.metadata or {}
                source_file = md.get("source_file") or md.get(
                    "source") or "unknown"
                page = md.get("page") or md.get("page_number")
                chunk_index = md.get("chunk_index")
                st.write(
                    f"- {source_file} | page={page} | chunk={chunk_index}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content})
