import hashlib
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.config import settings


def load_policies(folder: str):
    docs = []
    for pdf in Path(folder).glob("*.pdf"):
        for page in PyMuPDFLoader(str(pdf)).load():
            page.metadata["policy_name"] = pdf.stem
            page.metadata["page"] = page.metadata.get("page", 0) + 1
            docs.append(page)
        print(f"Loaded {pdf.name}")
    return docs


def chunk(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def make_id(doc) -> str:
    key = f"{doc.metadata['policy_name']}|{doc.metadata['page']}|{doc.page_content}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def main():
    docs = load_policies("data/policies")
    if not docs:
        print("No PDFs found in data/policies")
        return

    chunks = chunk(docs)
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model, base_url=settings.ollama_base_url
    )
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        ids=[make_id(c) for c in chunks],
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
    )
    print(f"Ingested {len(chunks)} chunks from {len(docs)} pages")


if __name__ == "__main__":
    main()
