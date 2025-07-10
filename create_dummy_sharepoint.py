import os
import numpy as np
from docx import Document
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

# --------------- CONFIG ---------------

ROOT_DIR = "DummySharePoint/Documents"
MODEL_NAME = 'all-MiniLM-L6-v2'  # Small and good for vector search

# --------------- UTILS ---------------

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def get_file_content(file_path):
    if file_path.endswith(".docx"):
        return read_docx(file_path)
    elif file_path.endswith(".txt"):
        return read_txt(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf(file_path)
    else:
        return ""

# --------------- EMBEDDING + INDEX ---------------

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

file_texts = []
file_metadata = []

print("Reading files and generating embeddings...")
for root, dirs, files in os.walk(ROOT_DIR):
    for fname in files:
        fpath = os.path.join(root, fname)
        text = get_file_content(fpath)
        if text.strip():  # Skip empty files
            file_texts.append(text)
            file_metadata.append({
                "file_path": fpath,
                "file_name": fname
            })

# Generate embeddings
embeddings = model.encode(file_texts)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print(f"Indexed {len(embeddings)} documents.")

# --------------- SEARCH FUNCTION ---------------

def vector_search(query, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    results = []
    for idx in I[0]:
        results.append({
            "file": file_metadata[idx]['file_name'],
            "path": file_metadata[idx]['file_path'],
            "preview": file_texts[idx][:200] + "..."
        })
    return results

# --------------- EXAMPLE SEARCH ---------------

if __name__ == "__main__":
    while True:
        query = input("\n🔎 Enter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = vector_search(query)
        for r in results:
            print(f"\n📄 {r['file']}")
            print(f"📂 {r['path']}")
            print(f"🔹 Preview: {r['preview']}")
