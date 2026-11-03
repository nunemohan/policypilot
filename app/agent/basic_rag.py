import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import settings
from app.retrieval.retriever import get_retriever

SYSTEM_PROMPT = """You are PolicyPilot, an assistant that answers questions about insurance policies.

Rules:
1. Answer ONLY using the context below. Do not use outside knowledge.
2. Cite every fact as [policy_name, p.X] using the labels in the context.
3. If the answer is not in the context, say exactly:
   "I could not find this in the provided policies."
4. Be concise and precise. Quote amounts and limits exactly."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata['policy_name']}, p.{d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )


def answer(question: str):
    docs = get_retriever().invoke(question)
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    chain = PROMPT | llm | StrOutputParser()
    result = chain.invoke({"context": format_docs(docs), "question": question})
    return result, docs


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the ambulance limit in the SBI policy?"
    result, docs = answer(question)

    print(f"\nQUESTION: {question}\n")
    print(f"ANSWER:\n{result}\n")
    print("RETRIEVED CHUNKS:")
    for d in docs:
        print(f"  - [{d.metadata['policy_name']}, p.{d.metadata['page']}]")