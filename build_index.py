import fitz  # PyMuPDF: For reading PDF files.
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
PDF_PATH = "docs/constitution.pdf"
INDEX_PATH = "storage/faiss_index.bin"
METADATA_PATH = "storage/index_metadata.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load PDF
doc = fitz.open(PDF_PATH)

# Initialize containers
text_chunks = []
metadatas = []

# Splitter config
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)

# Extract & chunk PDF
for i, page in enumerate(doc):
    page_text = page.get_text()
    if not page_text.strip():
        continue  # Skip empty pages

    chunks = splitter.split_text(page_text)
    for j, chunk in enumerate(chunks):
        text_chunks.append(chunk)
        metadatas.append({
            "page": i + 1,
            "chunk_id": j,
            "source": os.path.basename(PDF_PATH)
        })

# Embed text
embeddings = model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index and metadata
os.makedirs("storage", exist_ok=True)
faiss.write_index(index, INDEX_PATH)

with open(METADATA_PATH, "wb") as f:
    pickle.dump({"texts": text_chunks, "metadatas": metadatas}, f)

print("✅ FAISS index and metadata saved.")
