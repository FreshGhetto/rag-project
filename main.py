import os
from dotenv import load_dotenv

from app.ingest import build_or_load_vectorstore
from app.rag import make_rag_chain

def main():
    load_dotenv()

    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("MISTRAL_API_KEY mancante")

    vectorstore = build_or_load_vectorstore("/app/data", "/app/chroma_db")
    rag = make_rag_chain(vectorstore)

    while True:
        q = input("\nDomanda (invio per uscire): ").strip()
        if not q:
            break
        print("\nRisposta:\n", rag.invoke(q))

if __name__ == "__main__":
    main()
