from src.document_loader import load_pdf, split_documents
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, get_retriever
from src.llm import create_llm


def create_rag_pipeline():

    # 1. Load PDF
    documents = load_pdf()

    # 2. Split PDF into chunks
    chunks = split_documents(documents)

    # 3. Create embeddings
    embeddings = create_embeddings()

    # 4. Create vector store
    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    # 5. Create retriever
    retriever = get_retriever(
        vector_store
    )

    # 6. Load local LLM
    llm = create_llm()

    return retriever, llm


def ask_question(
    question,
    retriever,
    llm,
    chat_history=None
):

    # --------------------------------------------------
    # Conversation History
    # --------------------------------------------------

    if chat_history is None:
        chat_history = []

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history
    )


    # --------------------------------------------------
    # Retrieve Relevant Documents
    # --------------------------------------------------

    documents = retriever.invoke(question)


    # --------------------------------------------------
    # Combine Retrieved Context
    # --------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # --------------------------------------------------
    # Create Prompt
    # --------------------------------------------------

    prompt = f"""
You are a helpful AI assistant answering questions
from a PDF document.

Use only the provided PDF context to answer the question.

You may use the conversation history to understand
follow-up questions.

If the answer cannot be found in the PDF context,
say:

"I could not find the answer in the provided document."

Conversation History:
{history_text}

PDF Context:
{context}

Current Question:
{question}

Answer:
"""


    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    response = llm.invoke(prompt)


    # --------------------------------------------------
    # Extract Source Pages
    # --------------------------------------------------

    sources = []

    for document in documents:

        page = document.metadata.get("page")

        if page is not None:

            page_number = page + 1

            sources.append(
                f"Page {page_number}"
            )


    # Remove duplicate pages
    sources = list(
        dict.fromkeys(sources)
    )


    return (
        response.content,
        documents,
        sources
    )