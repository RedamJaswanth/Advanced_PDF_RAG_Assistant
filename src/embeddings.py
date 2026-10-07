from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Hugging Face embeddings model loaded.")

    return embeddings