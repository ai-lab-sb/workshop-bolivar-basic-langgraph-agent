from langchain_google_vertexai import VertexAIEmbeddings


def get_embedding(text: str, credentials_dict: dict) -> list[float]:
    """Takes a text and returns its embedding as a Python list using Google's Gemini embedding model via Vertex AI."""
    embeddings_model = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=credentials_dict.get("project_id"),
        credentials=credentials,
    )
    return embeddings_model.embed_query(text)


if __name__ == "__main__":
    import json

    with open("credentials.json") as f:
        creds = json.load(f)

    sample = "Hello, this is a test sentence."
    result = get_embedding(sample, creds)
    print(f"Embedding length: {len(result)}")
    print(f"First 5 values: {result[:5]}")
