import os
import pickle
import logging
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import igraph as ig
import leidenalg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GRAPH_PATH = "storage/kg_graph.pkl"

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists(GRAPH_PATH):
            try:
                with open(GRAPH_PATH, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded existing KG from {GRAPH_PATH}")
            except Exception as e:
                logger.warning(f"Could not load KG pickle: {e}. Starting fresh.")
        else:
            logger.info("No existing KG found; starting with an empty graph.")

    def add_fact(self, subject: str, relation: str, obj: str, source: str = None):
        self.graph.add_edge(subject, obj, relation=relation, source=source)
        logger.info(f"Added triple: ({subject}) -[{relation}]-> ({obj})")

        for node in [subject, obj]:
            if "embedding" not in self.graph.nodes[node]:
                self.graph.nodes[node]["embedding"] = self.model.encode(node)

    def query(self, keyword: str, k: int = 5) -> List[Dict]:
        nodes = [n for n in self.graph.nodes if "embedding" in self.graph.nodes[n]]
        if not nodes:
            logger.warning("KG has no embedded nodes.")
            return []

        node_embs = np.array([self.graph.nodes[n]["embedding"] for n in nodes])
        q_emb = self.model.encode([keyword])[0]

        node_embs = node_embs / np.linalg.norm(node_embs, axis=1, keepdims=True)
        q_emb = q_emb / np.linalg.norm(q_emb)
        scores = node_embs @ q_emb

        top_idxs = np.argsort(scores)[-k:][::-1]
        results = []

        for idx in top_idxs:
            node = nodes[idx]
            for nbr in self.graph.successors(node):
                ed = self.graph.get_edge_data(node, nbr)
                results.append({
                    "subject": node,
                    "relation": ed["relation"],
                    "object": nbr,
                    "source": ed.get("source", "Unknown"),
                    "score": float(scores[idx])
                })
        return results

    def detect_communities(self):
        logger.info("Detecting communities using Leiden algorithm...")
        undirected = self.graph.to_undirected()
        ig_graph = ig.Graph.TupleList(undirected.edges(), directed=False)

        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
        for community_id, community in enumerate(partition):
            for ig_node in community:
                node_name = ig_graph.vs[ig_node]["name"]
                if node_name in self.graph.nodes:
                    self.graph.nodes[node_name]["community"] = community_id

        logger.info(f"Assigned {len(partition)} communities to graph nodes.")

    def persist(self):
        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info(f"KG persisted to {GRAPH_PATH}")
