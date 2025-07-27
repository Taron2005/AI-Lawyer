import os
import faiss
import pickle
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(
        self,
        index_path: str = "storage/faiss_index.bin",
        metadata_path: str = "storage/index_metadata.pkl",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer(model_name)

        self.index = self._load_faiss_index()
        self.texts, self.metadatas = self._load_metadata()

    def _load_faiss_index(self):
        if not os.path.exists(self.index_path):
            logger.warning("FAISS index not found, creating a new one.")
            return faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        try:
            return faiss.read_index(self.index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")

    def _load_metadata(self):
        if not os.path.exists(self.metadata_path):
            logger.warning("Metadata file not found, initializing empty metadata.")
            return [], []
        try:
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
            return data.get("texts", []), data.get("metadatas", [])
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if not query.strip():
            raise ValueError("Query is empty.")

        query_embedding = self._normalize(self.model.encode([query]))
        D, I = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "score": float(dist)
            })
        return results

    def add_to_index(self, file_name: str, file_chunks: List[str]):
        if not file_chunks:
            logger.warning("No file chunks provided.")
            return

        new_embeddings = self._normalize(
            self.model.encode(file_chunks)
        ).astype(np.float32)

        # Deduplication
        unique_chunks = []
        unique_embeddings = []
        new_metadatas = []

        for chunk, embedding in zip(file_chunks, new_embeddings):
            if chunk not in self.texts:
                unique_chunks.append(chunk)
                unique_embeddings.append(embedding)
                new_metadatas.append({
                    "source": file_name,
                    "added_at": datetime.utcnow().isoformat()
                })

        if not unique_chunks:
            logger.info("No new unique chunks to add.")
            return

        self.index.add(np.array(unique_embeddings, dtype=np.float32))
        self.texts.extend(unique_chunks)
        self.metadatas.extend(new_metadatas)

        self._persist()
        logger.info(f"Added {len(unique_chunks)} new chunks to index.")

    def delete_by_source(self, file_name: str):
        """
        Delete all entries with metadata["source"] == file_name.
        This rebuilds the index from remaining entries.
        """
        logger.info(f"Deleting all entries from: {file_name}")

        # Filter out the entries that match the file_name
        kept_texts = []
        kept_metadatas = []

        for text, meta in zip(self.texts, self.metadatas):
            if meta.get("source") != file_name:
                kept_texts.append(text)
                kept_metadatas.append(meta)

        if len(kept_texts) == len(self.texts):
            logger.warning("No entries found for deletion.")
            return

        self.texts = kept_texts
        self.metadatas = kept_metadatas

        # Recompute embeddings and reinitialize index
        if kept_texts:
            kept_embeddings = self._normalize(self.model.encode(kept_texts)).astype(np.float32)
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
            self.index.add(kept_embeddings)
        else:
            # Reset to empty
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())

        self._persist()
        logger.info(f"Deleted file: {file_name}. Index rebuilt with {len(self.texts)} entries.")

    def _persist(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump({
                    "texts": self.texts,
                    "metadatas": self.metadatas
                }, f)
            logger.info("Index and metadata saved.")
        except Exception as e:
            logger.error(f"Failed to persist index and metadata: {e}")
            raise
