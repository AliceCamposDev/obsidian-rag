
import requests
from langchain_core.embeddings import Embeddings
from typing import List
import src.utils.utils as utils

config = utils.load_config()

class OllamaEmbeddingWrapper(Embeddings):
    def __init__(self, endpoint: str = "http://localhost:11434/api/embeddings"):
        self.model = config["embedding"]["embedding_model"]
        self.endpoint = endpoint

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, chunk in enumerate(texts):
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": chunk},
                timeout=60
            )
            res = response.json()
            if "embedding" not in res:
                raise RuntimeError(f"Missing embedding in response: {res}")
            embedding = res["embedding"]
            # if len(embedding) != 768:
            #     raise ValueError(f"Embedding at index {i} has length {len(embedding)}, expected 768")
            embeddings.append(embedding)
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "prompt": text},
            timeout=60
        )
        res = response.json()
        if "embedding" not in res:
            raise RuntimeError(f"Missing embedding in query response: {res}")
        return res["embedding"]