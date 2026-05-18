import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.ingest import has_vectorstore, index_pdfs, list_indexed_sources, load_vectorstore
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


@st.cache_resource(show_spinner="Caricamento dell'indice...")
def load_chain(chroma_dir: str):
    vectorstore = load_vectorstore(chroma_dir)
    return make_rag_chain(vectorstore)


def create_new_index(chroma_root: Path) -> Path:
    st.cache_resource.clear()
    index_dir = chroma_root / f"index_{int(time.time())}"
    index_dir.mkdir(parents=True, exist_ok=True)
    (chroma_root / ".active_index").write_text(index_dir.name, encoding="utf-8")
    return index_dir


def save_uploaded_pdfs(uploaded_files, pdf_dir: Path) -> list[Path]:
    saved = []
    for uploaded_file in uploaded_files:
        target = pdf_dir / Path(uploaded_file.name).name
        target.write_bytes(uploaded_file.getbuffer())
        saved.append(target)
    return saved


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
    indexed_sources = list_indexed_sources(str(chroma_dir))
    indexed_pdfs = [pdf for pdf in pdfs if pdf.name in indexed_sources]
    pending_pdfs = [pdf for pdf in pdfs if pdf.name not in indexed_sources]

    with st.sidebar:
        st.subheader("Configurazione")
        st.write(f"Modello: `{model}`")
        st.write(f"PDF trovati: `{len(pdfs)}`")
        st.write(f"PDF indicizzati: `{len(indexed_pdfs)}`")
        st.write(f"PDF da indicizzare: `{len(pending_pdfs)}`")

        if pdfs:
            st.caption("Documenti")
            for pdf in pdfs:
                status = "indicizzato" if pdf.name in indexed_sources else "da indicizzare"
                st.write(f"{pdf.name} - {status}")

    uploaded_files = st.file_uploader(
        "Aggiungi PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Trascina qui i PDF oppure selezionali dal computer. I file vengono copiati nella cartella data.",
    )

    if uploaded_files and st.button("Salva PDF in data", use_container_width=True):
        saved = save_uploaded_pdfs(uploaded_files, pdf_dir)
        st.success(f"PDF salvati: {len(saved)}")
        st.rerun()

    if not api_key:
        st.error("MISTRAL_API_KEY manca nel file .env.")
        st.stop()

    if not pdfs:
        st.warning("Aggiungi almeno un PDF usando il riquadro qui sopra.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Indicizza PDF da indicizzare", type="primary", use_container_width=True):
            if not pending_pdfs:
                st.info("Non ci sono nuovi PDF da indicizzare.")
            else:
                with st.spinner("Indicizzazione dei PDF nuovi..."):
                    index_pdfs([str(pdf) for pdf in pending_pdfs], str(chroma_dir))
                st.cache_resource.clear()
                st.success("Indicizzazione completata.")
                st.rerun()

    with col2:
        if st.button("Ricostruisci tutto l'indice", use_container_width=True):
            new_index_dir = create_new_index(chroma_root)
            with st.spinner("Ricostruzione completa dell'indice..."):
                index_pdfs([str(pdf) for pdf in pdfs], str(new_index_dir))
            st.cache_resource.clear()
            st.success("Indice ricostruito.")
            st.rerun()

    if not has_vectorstore(str(chroma_dir)):
        st.warning("Indice non ancora creato. Premi 'Indicizza PDF da indicizzare' per iniziare.")
        st.stop()

    try:
        rag = load_chain(str(chroma_dir))
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
