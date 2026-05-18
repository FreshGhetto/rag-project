import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[Documento: {os.path.basename(d.metadata.get('source', 'sconosciuto'))} "
        f"- pagina {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )


def make_rag_chain(vectorstore, model: str | None = None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un assistente preciso. Rispondi usando SOLO il contesto fornito. "
         "Se la risposta non è nel contesto, dillo chiaramente."),
        ("human", "Domanda: {question}\n\nContesto:\n{context}")
    ])

    model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    llm = ChatMistralAI(model=model, temperature=0)

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
