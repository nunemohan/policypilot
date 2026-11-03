from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.config import settings


def get_vector_store() -> QdrantVectorStore:
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model, base_url=settings.ollama_base_url
    )
    return QdrantVectorStore.from_existing_collection(
        collection_name=settings.collection_name,
        embedding=embeddings,
        url=settings.qdrant_url,
    )


def get_retriever(k: int | None = None):
    return get_vector_store().as_retriever(
        search_kwargs={"k": k or settings.top_k}
    )