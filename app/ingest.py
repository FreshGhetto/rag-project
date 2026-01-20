import os
from glob import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings


def build_or_load_vectorstore(pdf_dir: str, persist_dir: str) -> Chroma:
    """
    Crea (se non esiste) o carica (se esiste) un Chroma vectorstore persistente.

    - pdf_dir: cartella con i PDF (nel container: /app/data)
    - persist_dir: cartella per persistere Chroma (nel container: /app/chroma_db)
    """
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    # Se esiste già un DB, caricalo
    if os.listdir(persist_dir):
        return Chroma(
            collection_name="pdf_rag",
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    # Altrimenti ingest dei PDF
    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise RuntimeError(
            f"Nessun PDF trovato in {pdf_dir}. Metti i PDF nella cartella ./data del progetto."
        )

    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(p).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_rag",
        persist_directory=persist_dir,
    )


    return vs
