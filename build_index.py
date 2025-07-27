import fitz  # PyMuPDF: For reading PDF files.
import os     # (Not used here, but often useful for path operations)
import pickle # For serializing and saving data.
import faiss  # Facebook AI Similarity Search - efficient vector search library.
from sentence_transformers import SentenceTransformer  # To create embeddings.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Smart chunking.



model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast

# Load PDF
doc = fitz.open("docs/constitution.pdf")
text_chunks = []
metadatas = []


# Chunk the text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)


for i, page in enumerate(doc):
    page_text = page.get_text()
    chunks = splitter.split_text(page_text)

    for j, chunk in enumerate(chunks):
        text_chunks.append(chunk)
        metadatas.append({
            "page": i + 1,
            "chunk_id": j,
            "source": "constitution.pdf"
        })

embeddings = model.encode(text_chunks)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, "storage/faiss_index.bin")


# Save text chunks & metadata
with open("storage/index_metadata.pkl", "wb") as f:
    pickle.dump({"texts": text_chunks, "metadatas": metadatas}, f)

print("FAISS index and metadata saved.")
