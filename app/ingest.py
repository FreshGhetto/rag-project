import os
from glob import glob
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings


COLLECTION_NAME = "pdf_rag"


def _embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(model="mistral-embed")


def _vectorstore(persist_dir: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        embedding_function=_embeddings(),
    )


def has_vectorstore(persist_dir: str) -> bool:
    return Path(persist_dir, "chroma.sqlite3").exists()


def load_vectorstore(persist_dir: str) -> Chroma:
    if not has_vectorstore(persist_dir):
        raise RuntimeError("Indice Chroma non trovato. Premi 'Indicizza PDF' prima di fare domande.")

    return _vectorstore(persist_dir)


def list_indexed_sources(persist_dir: str) -> set[str]:
    if not has_vectorstore(persist_dir):
        return set()

    data = _vectorstore(persist_dir).get(include=["metadatas"])
    sources = set()
    for metadata in data.get("metadatas", []):
        source = metadata.get("source") if metadata else None
        if source:
            sources.add(os.path.basename(source))
    return sources


def index_pdfs(pdf_paths: Iterable[str], persist_dir: str) -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)

    pdf_paths = sorted(str(Path(p)) for p in pdf_paths)
    if not pdf_paths:
        raise RuntimeError("Nessun PDF da indicizzare.")

    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(p).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise RuntimeError("I PDF selezionati non contengono testo indicizzabile.")

    if has_vectorstore(persist_dir):
        vs = _vectorstore(persist_dir)
        vs.add_documents(chunks)
        return vs

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return vs


def build_or_load_vectorstore(pdf_dir: str, persist_dir: str) -> Chroma:
    """
    Compatibilita CLI: carica l'indice se esiste, altrimenti indicizza tutti i PDF.
    La GUI usa funzioni separate per evitare indicizzazioni automatiche all'avvio.
    """
    if has_vectorstore(persist_dir):
        return load_vectorstore(persist_dir)

    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise RuntimeError(
            f"Nessun PDF trovato in {pdf_dir}. Metti i PDF nella cartella ./data del progetto."
        )

    return index_pdfs(pdf_paths, persist_dir)
