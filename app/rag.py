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
         "Interpreta la domanda in modo naturale: maiuscole/minuscole non contano "
         "e piccoli refusi o forme colloquiali vanno corretti mentalmente. "
         "Per esempio, 'rag', 'RAG' e 'retrieval augmented generation' indicano lo stesso concetto. "
         "Se il contesto contiene informazioni pertinenti, rispondi anche se la domanda non usa "
         "le stesse identiche parole del testo. Se invece la risposta non è davvero nel contesto, "
         "dillo chiaramente."),
        ("human", "Domanda: {question}\n\nContesto:\n{context}")
    ])

    model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    llm = ChatMistralAI(model=model, temperature=0)

    answer_chain = prompt | llm | StrOutputParser()

    def answer_with_sources(question: str) -> dict:
        docs = retriever.invoke(question)
        answer = answer_chain.invoke({
            "context": _format_docs(docs),
            "question": question,
        })
        return {"answer": answer, "sources": docs}

    return RunnablePassthrough() | answer_with_sources
