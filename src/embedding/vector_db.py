from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os

from src.embedding.ollama_embedding_wrapper import OllamaEmbeddingWrapper
from src.utils.utils import load_vault
def generate_vector_db (vault_path: Path) -> Any:
    docs: List[Document] = load_vault(vault_path)
    
    #TODO: Filter and preprocess docs properly later
    
    print("Filtering empty docs")
    filtered_docs = []
    for doc in docs:
        if doc.page_content.strip():
            filtered_docs.append(doc)
    
    print("starting embedding")
    
    embedder = OllamaEmbeddingWrapper() #i guess i created a new word
    vector_db = FAISS.from_documents(filtered_docs, embedder)
    
    try:
        vector_db_path: Path = Path("/data/vector_db")
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
        vector_db.save_local(str(vector_db_path))
        print("vector_db saved")
    except Exception as e:
        print("failed while saving vector db")
    
    return vector_db

#TODO: loadvector

def load_vector_db() -> Any:
    print ("aaawwa")