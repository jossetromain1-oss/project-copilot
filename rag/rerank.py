from typing import List, Dict
from sentence_transformers import CrossEncoder

class LocalReranker:
    """Rerank local des passages pour une requête donnée"""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs).tolist()
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]