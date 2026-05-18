import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.ingest import build_or_load_vectorstore
from app.rag import make_rag_chain


BASE_DIR = Path(__file__).resolve().parent


def get_paths() -> tuple[Path, Path]:
    pdf_dir = Path(os.getenv("PDF_DIR", BASE_DIR / "data"))
    chroma_root = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
    return pdf_dir, chroma_root


def get_active_index_dir(chroma_root: Path) -> Path:
    marker = chroma_root / ".active_index"
    if marker.exists():
        name = marker.read_text(encoding="utf-8").strip()
        if name:
            return chroma_root / name

    index_dir = chroma_root / "index"
    marker.write_text(index_dir.name, encoding="utf-8")
    return index_dir


@st.cache_resource(show_spinner="Indicizzazione dei PDF in corso...")
def load_chain(pdf_dir: str, chroma_dir: str):
    vectorstore = build_or_load_vectorstore(pdf_dir, chroma_dir)
    return make_rag_chain(vectorstore)


def reset_index(chroma_root: Path) -> None:
    st.cache_resource.clear()
    index_dir = chroma_root / f"index_{int(time.time())}"
    index_dir.mkdir(parents=True, exist_ok=True)
    (chroma_root / ".active_index").write_text(index_dir.name, encoding="utf-8")


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="RAG Scuola", page_icon="📚", layout="wide")
    st.title("RAG Scuola")

    pdf_dir, chroma_root = get_paths()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    chroma_root.mkdir(parents=True, exist_ok=True)
    chroma_dir = get_active_index_dir(chroma_root)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    pdfs = sorted(pdf_dir.glob("*.pdf"))

    with st.sidebar:
        st.subheader("Configurazione")
        st.write(f"Modello: `{model}`")
        st.write(f"PDF trovati: `{len(pdfs)}`")

        if pdfs:
            st.caption("Documenti")
            for pdf in pdfs:
                st.write(pdf.name)

        if st.button("Ricostruisci indice", use_container_width=True):
            reset_index(chroma_root)
            st.rerun()

    if not api_key:
        st.error("MISTRAL_API_KEY manca nel file .env.")
        st.stop()

    if not pdfs:
        st.warning("Metti almeno un PDF nella cartella data, poi ricarica la pagina.")
        st.stop()

    try:
        rag = load_chain(str(pdf_dir), str(chroma_dir))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    question = st.text_area("Domanda", height=110, placeholder="Scrivi una domanda sui PDF...")

    if st.button("Chiedi", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Scrivi una domanda prima di inviare.")
            st.stop()

        with st.spinner("Generazione della risposta..."):
            answer = rag.invoke(question.strip())

        st.subheader("Risposta")
        st.write(answer)


if __name__ == "__main__":
    main()
