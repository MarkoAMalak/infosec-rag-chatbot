import os

import faiss
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

model.eval()


def generate(prompt):
    """Run FLAN-T5 directly (the text2text pipeline was removed in transformers 5)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# =========================
# RAG pipeline
# =========================
def answer_question(question):
    if not question or not question.strip():
        return "Please enter a question.", ""

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

    result = generate(prompt)

    return result.strip(), context


# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
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
                lines=2,
            )
            ask_btn = gr.Button("🚀 Ask", variant="primary")

        with gr.Column(scale=1):
            answer = gr.Textbox(label="🧠 Answer", lines=6)

    with gr.Accordion("📄 Retrieved Security Context", open=False):
        context_box = gr.Textbox(lines=10, label="Context used by the model")

    ask_btn.click(fn=answer_question, inputs=question, outputs=[answer, context_box])
    question.submit(fn=answer_question, inputs=question, outputs=[answer, context_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                theme=gr.themes.Soft())
