from app.config import EMBEDDING_MODEL, client

def embed_texts(text: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return [e.values for e in result.embeddings]