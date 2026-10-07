from langchain_ollama import ChatOllama


def create_llm():
    llm = ChatOllama(
        model="qwen2.5:3b"
    )

    print("Ollama LLM loaded successfully.")

    return llm