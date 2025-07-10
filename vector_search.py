import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

# === SETTINGS ===
DATA_DIR = "."  # current directory
PREVIEW_LENGTH = 300
MODEL_NAME = "all-MiniLM-L6-v2"  # or your preferred model

# === Load model once ===
print("🧠 Loading local embedding model...")
model = SentenceTransformer(MODEL_NAME)

# === Read file content ===
def read_text_from_file(path):
    try:
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif path.endswith(".pdf"):
            return " ".join(page.extract_text() or "" for page in PdfReader(path).pages)
        elif path.endswith(".docx"):
            return " ".join(p.text for p in DocxDocument(path).paragraphs)
    except Exception as e:
        print(f"❌ Failed to read {path}: {e}")
    return ""

# === Load and Embed Documents ===
def load_documents_and_embeddings():
    data = []
    texts = []

    full_data_dir = os.path.abspath(DATA_DIR)
    if not os.path.exists(full_data_dir):
        raise ValueError(f"❌ Folder does not exist: {full_data_dir}")

    print(f"📂 Scanning folder: {full_data_dir}")
    for root, _, files in os.walk(full_data_dir):
        for file in files:
            if file.endswith((".txt", ".pdf", ".docx")):
                path = os.path.join(root, file)
                text = read_text_from_file(path)
                if not text.strip():
                    continue
                texts.append(text)
                data.append({
                    "file": file,
                    "path": path,
                    "text": text,
                    "preview": text[:PREVIEW_LENGTH] + "..."
                })

    if not texts:
        raise ValueError("❌ No valid documents found.")

    print(f"🧠 Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return data, embeddings

# === Build Index ===
documents, embeddings = load_documents_and_embeddings()

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === Search Function ===
def vector_search(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]
