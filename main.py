from embedding.gemini import GeminiEmbeddingClient
from misc.proxy import setup_proxy


def main():
    setup_proxy()

    print("Hello from rag-example!")
    print("testing Gemini embedding API...")

    client = GeminiEmbeddingClient(dimension=768)
    response = client.get_embedding("Hello, world!")


if __name__ == "__main__":
    main()
