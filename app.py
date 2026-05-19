import gradio as gr
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# =========================
# Load knowledge base
# =========================
DATA_FILE = "information_security.txt"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("information_security.txt not found")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# =========================
# Retriever (Embeddings + FAISS)
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, show_progress_bar=True)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings).astype("float32"))

# =========================
# Generator (Local HF Model)
# =========================
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
)

# =========================
# RAG pipeline
# =========================
def answer_question(question):
    # Retrieve relevant context
    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb).astype("float32"), k=3)

    context = "\n".join([documents[i] for i in I[0]])

    prompt = f"""
You are an information security expert.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt)[0]["generated_text"]

    return result.strip(), context

# =========================
# Gradio UI (Friendly)
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔐 Information Security Assistant
        Ask questions about **Information Security concepts**  
        (CIA triad, phishing, malware, access control, etc.)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(
                label="💬 Your Question",
                placeholder="e.g. What is the CIA triad?",
                lines=2
            )
            ask_btn = gr.Button("🚀 Ask", variant="primary")

        with gr.Column(scale=1):
            answer = gr.Textbox(
                label="🧠 Answer",
                lines=6
            )

    with gr.Accordion("📄 Retrieved Security Context", open=False):
        context_box = gr.Textbox(lines=10, label="Context used by the model")

    ask_btn.click(
        fn=answer_question,
        inputs=question,
        outputs=[answer, context_box]
    )

if __name__ == "__main__":
    demo.launch()
