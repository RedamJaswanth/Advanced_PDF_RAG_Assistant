import streamlit as st

from src.rag_pipeline import create_rag_pipeline, ask_question


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Advanced PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #777;
            margin-bottom: 30px;
        }

        .source-box {
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Advanced PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask intelligent questions about your PDF using Retrieval-Augmented Generation.</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ RAG System")

    st.write("**Embedding Model**")
    st.code("all-MiniLM-L6-v2")

    st.write("**Vector Database**")
    st.code("ChromaDB")

    st.write("**LLM**")
    st.code("Qwen2.5:3B")

    st.write("**Framework**")
    st.code("LangChain")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Load RAG Pipeline
# --------------------------------------------------

@st.cache_resource
def load_rag_pipeline():

    return create_rag_pipeline()


retriever, llm = load_rag_pipeline()


# --------------------------------------------------
# Chat History
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# User Input
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your PDF..."
)


# --------------------------------------------------
# Process Question
# --------------------------------------------------

if question:

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("🔎 Searching the document..."):

          answer, documents, sources = ask_question(
    question,
    retriever,
    llm,
    st.session_state.messages
)

        st.markdown(answer)


        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        if sources:

            st.markdown("### 📚 Sources")

            for source in sources:

                st.markdown(
                    f'<div class="source-box">📄 {source}</div>',
                    unsafe_allow_html=True
                )


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )