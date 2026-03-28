from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="📄 RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

/* Use Streamlit theme variables */
:root {
    --bg-color: var(--background-color);
    --text-color: var(--text-color);
    --secondary-bg: var(--secondary-background-color);
    --accent: #4CAF50;
}

/* Chat container spacing */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* User Bubble */
.chat-bubble-user {
    background: linear-gradient(135deg, #ff4b4b, #ff7a7a);
    color: white;
    padding: 12px 16px;
    border-radius: 15px 15px 5px 15px;
    max-width: 75%;
    align-self: flex-end;
    font-size: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* AI Bubble */
.chat-bubble-ai {
    background-color: var(--secondary-bg);
    color: var(--text-color);
    padding: 12px 16px;
    border-radius: 15px 15px 15px 5px;
    max-width: 75%;
    align-self: flex-start;
    font-size: 14px;
    border: 1px solid rgba(150,150,150,0.2);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Sidebar Title */
.sidebar-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-color);
    margin-bottom: 10px;
}

/* Input box styling */
textarea {
    border-radius: 10px !important;
}

/* Scrollbar (optional nice touch) */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- PROCESS DOCUMENT --------------------    
def process_document(path):

    # Load PDFs
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    # 🔥 GOOGLE EMBEDDINGS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # LLM
    llm = ChatGroq(model="openai/gpt-oss-20b")

    # Tool
    @tool
    def retrieve_context(query: str):
        """Retrieve relevant document chunks."""
        results = vector_db.similarity_search(query=query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    # Prompt
    system_prompt = """
    You are a professional AI assistant.
    Answer ONLY from the provided document context.
    If unsure, say "I couldn't find this in the document."

    ALWAYS use retrieve_context tool before answering.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">📂 Upload Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("🔄 Processing documents..."):
            path = "./doc_files/"
            os.makedirs(path, exist_ok=True)

            for file in uploaded_files:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())

            process_document(path)

        st.success("✅ Documents processed successfully!")

# -------------------- MAIN UI --------------------
st.title("🤖 RAG Document Assistant")
st.caption("Ask questions based on your uploaded PDFs")

# Chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask something about your document...")

if query and st.session_state.agent:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🤖 Thinking..."):
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": 1}}
        )

    answer = response["messages"][-1].content

    st.session_state.messages.append({"role": "ai", "content": answer})
    st.rerun()

# Empty state
if not st.session_state.document_uploaded:
    st.info("📌 Upload PDFs from the sidebar to get started.")