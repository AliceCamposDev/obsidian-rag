import yaml
from pathlib import Path
from typing import Any, List

import sys
print("importing langchain_community DirectoryLoader")
#TODO: taking too long to import
from langchain_community.document_loaders import DirectoryLoader
    
from langchain_core.documents import Document
import pickle
import os

def load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    

def load_vault(vault_path: Path) -> List[Document]:
    """
        Load Vault from VaultPath

    Args:
        vault_path (Path): vault path

    Returns:
       List[Document]: loaded docs list
    """    
    print("loading docs from ", vault_path)
    loader = DirectoryLoader(str(vault_path), glob="**/*.md")
    docs = loader.load()
    save_docs(docs)
    return docs


def load_docs() -> List[Document]:
    docs_path: Path = Path("/data/docs.pkl")
    try:  
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
    except Exception as e:
        print("failed to load docs from files")
    
    return docs

def save_docs(docs: List[Document]) -> None:
    try:
        docs_path: Path = Path("/data/docs.pkl")
        
        os.makedirs(os.path.dirname(docs_path), exist_ok=True)

        with open(docs_path, "wb") as f: #TODO: Save docs file only when necessary
            pickle.dump(docs, f)

    except Exception as e:
        print("failed to save docs to files")
    
