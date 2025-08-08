import os
import faiss
import pickle
import threading
import shutil
import numpy as np
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from rag_utils import smart_chunk_text, extract_text_from_file

STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(STORAGE_DIR, "metadata.pkl")
# Reverted to LaBSE for faster performance as requested
EMBEDDING_MODEL = "sentence-transformers/LaBSE"

class RAGManager:
    """
    A thread-safe singleton to manage the RAG model, FAISS index, and
    document processing, now with source tracking for each chunk.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RAGManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            with self._lock:
                if not hasattr(self, 'initialized'):
                    print("🚀 Initializing RAG Manager...")
                    os.makedirs(STORAGE_DIR, exist_ok=True)
                    self.model = SentenceTransformer(EMBEDDING_MODEL)
                    self._load_data()
                    self.initialized = True
                    print("✅ RAG Manager Initialized.")

    def _load_data(self):
        """Loads the FAISS index and chunk metadata from storage."""
        print("Attempting to load index and metadata...")
        try:
            if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
                self.index = faiss.read_index(INDEX_PATH)
                with open(METADATA_PATH, "rb") as f:
                    # Metadata now contains chunks with their sources
                    self.chunk_metadata = pickle.load(f).get("chunk_metadata", [])
                print(f"✅ Loaded existing index with {self.index.ntotal} vectors.")
            else:
                self._initialize_new_index()
        except Exception as e:
            print(f"❌ Error loading data: {e}. Initializing a fresh index.")
            self._initialize_new_index()

    def _initialize_new_index(self):
        """Initializes a new, empty FAISS index and metadata list."""
        print("⚠️ No existing index found. Initializing a new one.")
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        # chunk_metadata will be a list of dicts, e.g., [{"text": str, "source": str}]
        self.chunk_metadata = []

    def _save_data(self):
        """Saves the current index and chunk metadata to disk."""
        print("💾 Saving index and metadata...")
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump({"chunk_metadata": self.chunk_metadata}, f)
        print("✅ Data saved successfully.")

    def add_document(self, file_content: bytes, filename: str):
        """Adds a new document to the knowledge base, tracking chunk sources."""
        with self._lock:
            print(f"🔄 Processing document: {filename}")
            text = extract_text_from_file(file_content, filename)
            if not text:
                print(f"⚠️ Could not extract text from {filename}. Skipping.")
                return

            new_chunks_text = smart_chunk_text(text)
            if not new_chunks_text:
                return

            # Create metadata for each new chunk
            new_metadata = [{"text": chunk, "source": filename} for chunk in new_chunks_text]

            print(f"Embedding {len(new_chunks_text)} chunks for {filename}...")
            new_embeddings = self.model.encode(new_chunks_text, convert_to_numpy=True, show_progress_bar=True)

            self.index.add(new_embeddings)
            self.chunk_metadata.extend(new_metadata)
            self._save_data()
            print(f"✅ Successfully added {filename}. Total vectors: {self.index.ntotal}")

    def retrieve(self, query: str, top_k: int = 15, score_threshold: Optional[float] = None) -> List[Dict[str, str]]:
        """
        Retrieves the most relevant document chunks for a given query,
        returning both the text and its source metadata.
        """
        if self.index.ntotal == 0:
            return []
            
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if score_threshold is None or dist <= score_threshold:
                if idx < len(self.chunk_metadata):
                    results.append(self.chunk_metadata[idx])
        
        return results

    def delete_index(self):
        """Deletes the entire FAISS index and metadata from disk."""
        with self._lock:
            print("🗑️ Deleting existing index and metadata...")
            if os.path.exists(STORAGE_DIR):
                try:
                    shutil.rmtree(STORAGE_DIR)
                    print("✅ Storage directory completely removed.")
                except OSError as e:
                    print(f"❌ Error removing storage directory: {e}")
            
            os.makedirs(STORAGE_DIR, exist_ok=True)
            self._initialize_new_index()
            print("✨ A new, empty index has been initialized.")


def get_rag_manager():
    """Factory function to get the RAGManager instance."""
    return RAGManager()