from pathlib import Path
from src.utils.utils import load_vault
from typing import Any, List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
import os

def generate_bm25_index(vault_path: Path) -> Any:
    """
    Generate Okapi BM25 indexes
    
    Args:
        vault_path (Path): vault path

    Returns:
        Any: _description_
    """    
    docs: List[Document] = load_vault(vault_path)
    content = [doc.page_content for doc in docs]
    tokenized_content = [c.split() for c in content]
    bm25_index = BM25Okapi(tokenized_content)
    save_bm25_index_to_file(bm25_index, docs)
    return bm25_index

def save_bm25_index_to_file(bm25_index: Any, docs: List[Document]) -> None:
    try:
        bm25_path: Path = Path("/data/bm25/bm25.pkl")
        
        os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
        
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)
        
    except Exception as e:
        print ("failed to save bm25 index to files")
    
def load_bm25_index(vault_path: Path) -> None:
    bm25_path: Path = Path("/data/bm25/bm25.pkl")
    docs_path: Path = Path("/data/docs.pkl")
    
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

