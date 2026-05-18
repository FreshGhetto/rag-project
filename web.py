import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.ingest import has_vectorstore, index_pdfs, list_indexed_sources, load_vectorstore
from app.rag import make_rag_chain


BASE_DIR = Path(__file__).resolve().parent


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 2rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            padding-top: 0.25rem;
        }
        .doc-row {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 8px;
            padding: 0.45rem 0.55rem;
            margin-bottom: 0.35rem;
            font-size: 0.88rem;
        }
        .doc-status {
            color: #8b949e;
            font-size: 0.76rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
def load_chain(chroma_dir: str, model: str):
    vectorstore = load_vectorstore(chroma_dir)
    return make_rag_chain(vectorstore, model=model)


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

    st.set_page_config(page_title="RAG", page_icon="📚", layout="wide")
    apply_styles()

    pdf_dir, chroma_root = get_paths()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    chroma_root.mkdir(parents=True, exist_ok=True)
    chroma_dir = get_active_index_dir(chroma_root)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    model_options = {
        "Small": "mistral-small-latest",
        "Medium": "mistral-medium-latest",
        "Large": "mistral-large-latest",
    }
    default_model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    default_label = next(
        (label for label, value in model_options.items() if value == default_model),
        "Small",
    )
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    indexed_sources = list_indexed_sources(str(chroma_dir))
    indexed_pdfs = [pdf for pdf in pdfs if pdf.name in indexed_sources]
    pending_pdfs = [pdf for pdf in pdfs if pdf.name not in indexed_sources]

    with st.sidebar:
        st.title("RAG")
        st.divider()

        st.subheader("Modello")
        model_label = st.selectbox(
            "Modello",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(default_label),
            label_visibility="collapsed",
        )
        model = model_options[model_label]
        st.caption(f"ID modello: `{model}`")

        st.divider()
        st.subheader("Documenti")
        st.caption(
            f"{len(pdfs)} PDF totali · {len(indexed_pdfs)} indicizzati · "
            f"{len(pending_pdfs)} da indicizzare"
        )

        uploaded_files = st.file_uploader(
            "Aggiungi PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="Trascina qui i PDF oppure selezionali dal computer. I file vengono copiati nella cartella data.",
        )

        if uploaded_files and st.button("Salva in data", use_container_width=True):
            saved = save_uploaded_pdfs(uploaded_files, pdf_dir)
            st.success(f"PDF salvati: {len(saved)}")
            st.rerun()

        selected_pending = st.multiselect(
            "PDF da indicizzare",
            options=[pdf.name for pdf in pending_pdfs],
            default=[pdf.name for pdf in pending_pdfs],
            help="Seleziona solo i nuovi documenti che vuoi indicizzare adesso.",
        )
        selected_pending_paths = [pdf for pdf in pending_pdfs if pdf.name in selected_pending]

        if st.button("Indicizza selezionati", type="primary", use_container_width=True):
            if not selected_pending_paths:
                st.info("Seleziona almeno un PDF non ancora indicizzato.")
            else:
                with st.spinner("Indicizzazione..."):
                    index_pdfs([str(pdf) for pdf in selected_pending_paths], str(chroma_dir))
                st.cache_resource.clear()
                st.success("Indicizzazione completata.")
                st.rerun()

        if st.button("Ricostruisci indice", use_container_width=True):
            new_index_dir = create_new_index(chroma_root)
            with st.spinner("Ricostruzione..."):
                index_pdfs([str(pdf) for pdf in pdfs], str(new_index_dir))
            st.cache_resource.clear()
            st.success("Indice ricostruito.")
            st.rerun()

        if pdfs:
            with st.expander("Elenco PDF", expanded=False):
                for pdf in pdfs:
                    status = "indicizzato" if pdf.name in indexed_sources else "da indicizzare"
                    st.markdown(
                        f"""
                        <div class="doc-row">
                            {pdf.name}<br>
                            <span class="doc-status">{status}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    if not api_key:
        st.error("MISTRAL_API_KEY manca nel file .env.")
        st.stop()

    if not pdfs:
        st.title("RAG")
        st.warning("Aggiungi almeno un PDF dalla barra laterale.")
        st.stop()

    if not has_vectorstore(str(chroma_dir)):
        st.title("RAG")
        st.warning("Indice non ancora creato. Usa la sezione Documenti nella barra laterale.")
        st.stop()

    try:
        rag = load_chain(str(chroma_dir), model)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("RAG")

    if not st.session_state.messages:
        st.caption("Fai una domanda sui documenti indicizzati.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Scrivi una domanda sui PDF")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Generazione della risposta..."):
                answer = rag.invoke(question)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
