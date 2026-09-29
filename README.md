# 🔐 Information Security RAG Chatbot

An educational chatbot that answers **information security** questions using
**Retrieval-Augmented Generation (RAG)**. Answers are grounded in a curated
security knowledge base, and the retrieved context is shown next to each answer.

## How it works

```
Question ──► Sentence embedding (all-MiniLM-L6-v2)
         ──► FAISS similarity search (top-3 passages from information_security.txt)
         ──► Prompt with retrieved context ──► FLAN-T5-base ──► Answer
```

- **Knowledge base:** `information_security.txt` (CIA triad, phishing, malware, access control, …)
- **Retriever:** Sentence-Transformers embeddings + FAISS `IndexFlatL2`
- **Generator:** `google/flan-t5-base` (runs locally, no API key needed)
- **UI:** Gradio

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860. The first run downloads the models (~1 GB).

## Run with Docker

```bash
docker build -t infosec-rag-chatbot .
docker run -p 7860:7860 infosec-rag-chatbot
```

## Tech stack

Python · FAISS · Sentence-Transformers · Hugging Face Transformers (FLAN-T5) · Gradio · Docker

## Notes

- `fastapi` and `pydantic` are pinned because newer releases are incompatible with
  Gradio 4.29 and caused an *Internal Server Error* on the main page.
- Developed as a team project.
