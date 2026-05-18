# Rag_scuola
Creazione Rag

## Avvio facile con Docker

1. Inserisci la chiave Mistral in `.env`.
2. Fai doppio click su `avvia.bat`.
3. Apri la pagina web, trascina i PDF nel riquadro "Aggiungi PDF" e premi "Salva PDF in data".
4. Premi "Indicizza PDF da indicizzare".

Lo script avvia il progetto con Docker Compose e apre l'interfaccia web:

```text
http://localhost:8501
```

I dati Chroma vengono mantenuti nella cartella `chroma_db`.

L'indicizzazione non parte automaticamente a ogni avvio. L'app carica l'indice esistente; indicizza solo quando premi uno dei pulsanti dedicati nell'interfaccia.

## Avvio locale

```powershell
.\.venv\Scripts\Activate.ps1
```

Inserisci la chiave Mistral in `.env`:

```env
MISTRAL_API_KEY=la_tua_chiave
MISTRAL_MODEL=mistral-small-latest
```

Metti i PDF da indicizzare nella cartella `data`, poi avvia:

```powershell
python main.py
```

I dati Chroma vengono salvati in `chroma_db`.

## Interfaccia web locale

```powershell
streamlit run web.py
```
