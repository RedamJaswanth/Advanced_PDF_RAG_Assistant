from langchain_chroma import Chroma


CHROMA_PATH = "chroma_db"


def create_vector_store(chunks, embeddings):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("Chroma vector store created successfully.")

    return vector_store


def get_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    print("Retriever created successfully.")

    return retriever