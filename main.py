import os
from pathlib import Path
from dotenv import load_dotenv

from app.ingest import build_or_load_vectorstore
from app.rag import make_rag_chain


BASE_DIR = Path(__file__).resolve().parent


def main():
    load_dotenv()

    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError(
            "MISTRAL_API_KEY mancante. Inseriscila nel file .env prima di avviare."
        )

    pdf_dir = os.getenv("PDF_DIR", str(BASE_DIR / "data"))
    chroma_dir = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))

    vectorstore = build_or_load_vectorstore(pdf_dir, chroma_dir)
    rag = make_rag_chain(vectorstore)

    while True:
        q = input("\nDomanda (invio per uscire): ").strip()
        if not q:
            break
        print("\nRisposta:\n", rag.invoke(q))

if __name__ == "__main__":
    main()
