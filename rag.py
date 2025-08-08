import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llm import generate_with_groq

# NOTE: The main application (`api_backend.py` and `main.py`) uses a separate,
# more direct retrieval logic defined in the `rag_manager.py` file.
# This `AdvancedRAG` class is a self-contained implementation and is not
# currently used by the active API endpoints. The errors you are seeing
# originate from the `llm.py` file when it processes the context.

class AdvancedRAG:
    def __init__(self, documents, metadatas=None, embedding_model="sentence-transformers/LaBSE"):
        """
        documents: List of strings (knowledge base chunks)
        metadatas: List of dicts (metadata for each chunk, e.g., {"source": "doc1.pdf", "chunk_id": 0})
        embedding_model: HuggingFace model name for sentence embeddings
        """
        if not documents or not isinstance(documents, list):
            raise ValueError("Documents must be a non-empty list of strings.")
        self.documents = documents
        self.metadatas = metadatas if metadatas else [{} for _ in documents]
        self.model = SentenceTransformer(embedding_model)
        self.doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings)

    def retrieve(self, query, top_k=3, score_threshold=None, return_metadata=False):
        """
        Retrieve top_k most relevant documents for the query.
        Optionally filter by score_threshold (lower = more similar).
        Optionally return metadata.
        """
        if not query:
            raise ValueError("Query must be a non-empty string.")
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if score_threshold is not None and dist > score_threshold:
                continue
            item = {"text": self.documents[idx], "score": dist}
            if self.metadatas:
                item["metadata"] = self.metadatas[idx]
            results.append(item)
        if return_metadata:
            return results
        return [item["text"] for item in results]

    def batch_retrieve(self, queries, top_k=3):
        """
        Retrieve top_k docs for each query in a batch.
        Returns a list of lists.
        """
        query_vecs = self.model.encode(queries, convert_to_numpy=True)
        distances, indices = self.index.search(query_vecs, top_k)
        batch_results = []
        for idx_list in indices:
            batch_results.append([self.documents[i] for i in idx_list])
        return batch_results

    def format_context(self, docs_or_items, include_metadata=True):
        """
        Format retrieved docs/items as context for the LLM.
        If include_metadata, show metadata in the context.
        """
        formatted = []
        for i, item in enumerate(docs_or_items):
            if isinstance(item, dict):
                meta = item.get("metadata", {})
                meta_str = f" | Metadata: {meta}" if meta and include_metadata else ""
                formatted.append(f"Context chunk {i+1}:{meta_str}\n{item['text']}")
            else:
                formatted.append(f"Context chunk {i+1}:\n{item}")
        return "\n\n".join(formatted)

    def answer(self, question, top_k=5, model="llama3-70b-8192", score_threshold=None):
        """
        Retrieve context and generate an answer using the LLM.
        This method now passes raw chunks to the LLM function for better
        token management, aligning with the fixes made elsewhere.
        """
        # **FIX**: Retrieve raw text chunks instead of pre-formatted items.
        # This allows the `generate_with_groq` function to handle token budgeting.
        retrieved_chunks = self.retrieve(question, top_k=top_k, score_threshold=score_threshold)
        
        # Pass the list of chunks directly to the LLM function.
        return generate_with_groq(question, retrieved_chunks=retrieved_chunks, model=model)

