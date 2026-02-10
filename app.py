import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# =========================
# Load data from txt file
# =========================
DATA_FILE = "information_security.txt"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in project directory")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# =========================
# Load embedding model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts, show_progress_bar=True)

# =========================
# Build FAISS index
# =========================
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# =========================
# Search function
# =========================
def search(query):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb).astype("float32"), k=3)
    return [texts[i] for i in I[0]]

# =========================
# Gradio UI
# =========================
demo = gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Ask a question about Information Security"),
    outputs=gr.JSON(label="Top relevant chunks"),
    title="Information Security Semantic Search",
    description="Semantic search over information_security.txt using Sentence Transformers + FAISS"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
