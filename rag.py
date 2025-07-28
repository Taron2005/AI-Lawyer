import os
import faiss
import pickle
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

    def _load_faiss_index(self) -> faiss.Index:
        if not os.path.exists(self.index_path):
            logger.warning("FAISS index not found, creating a new one.")
            return faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        try:
            return faiss.read_index(self.index_path)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise

    def _load_metadata(self) -> Tuple[List[str], List[Dict]]:
        if not os.path.exists(self.metadata_path):
            logger.warning("Metadata file not found. Initializing empty metadata.")
            return [], []
        try:
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
            return data.get("texts", []), data.get("metadatas", [])
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-10)

    def _encode(self, texts: List[str]) -> np.ndarray:
        try:
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_numpy=True)
        except Exception:
            embeddings = self.model.encode(texts)
        return self._normalize(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if not query.strip():
            raise ValueError("Query string is empty.")

        embedding = self._encode([query])
        D, I = self.index.search(embedding, k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(dist)
                })
        return results

    def add_to_index(self, file_name: str, file_chunks: List[str]) -> None:
        if not file_chunks:
            logger.warning("No file chunks provided.")
            return

        new_chunks, new_metadatas = [], []
        existing_text_set = set(self.texts)

        for chunk in file_chunks:
            if chunk not in existing_text_set:
                new_chunks.append(chunk)
                new_metadatas.append({
                    "source": file_name,
                    "added_at": datetime.utcnow().isoformat()
                })

        if not new_chunks:
            logger.info("No new unique chunks to add.")
            return

        new_embeddings = self._encode(new_chunks)
        self.index.add(new_embeddings)
        self.texts.extend(new_chunks)
        self.metadatas.extend(new_metadatas)

        self._persist()
        logger.info(f"Added {len(new_chunks)} new chunks from '{file_name}'.")

    def delete_by_source(self, file_name: str) -> None:
        logger.info(f"Deleting all entries from source: '{file_name}'")

        kept_texts, kept_metadatas = [], []
        for text, meta in zip(self.texts, self.metadatas):
            if meta.get("source") != file_name:
                kept_texts.append(text)
                kept_metadatas.append(meta)

        if len(kept_texts) == len(self.texts):
            logger.warning("No entries found for deletion.")
            return

        self.texts, self.metadatas = kept_texts, kept_metadatas

        if kept_texts:
            embeddings = self._encode(kept_texts)
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())

        self._persist()
        logger.info(f"Deleted source '{file_name}'. Rebuilt index with {len(self.texts)} entries.")

    def _persist(self) -> None:
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump({"texts": self.texts, "metadatas": self.metadatas}, f)
            logger.info("Index and metadata saved.")
        except Exception as e:
            logger.error(f"Failed to persist data: {e}")
            raise
