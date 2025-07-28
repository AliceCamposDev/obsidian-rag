import os
import re
import json
import pathlib
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import json
from langchain_core.embeddings import Embeddings
from typing import List
import pickle
import random
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


#TODO: import and use configs in a better way
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

EMBEDDING_MODEL = config["embedding"]["embedding_model"]
OUTPUT_PATH = config["embedding"]["output_path"]
OLLAMA_ENDPOINT = config["general"]["ollama_endpoint"]+"/embeddings"
VAULT_PATH = config["general"]["vault_path"]


class OllamaEmbeddingWrapper(Embeddings):
    def __init__(self, model: str = "nomic-embed-text", endpoint: str = "http://localhost:11434/api/embeddings"):
        self.model = model
        self.endpoint = endpoint

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, chunk in enumerate(texts):
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": "[BRAZILIAN PORTUGUESE]"+chunk},
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
            json={"model": self.model, "prompt": "[BRAZILIAN PORTUGUESE]"+text},
            timeout=60
        )
        res = response.json()
        if "embedding" not in res:
            raise RuntimeError(f"Missing embedding in query response: {res}")
        return res["embedding"]
    
def setup_indexes(vault_path):
    loader = DirectoryLoader(vault_path, glob="**/*.md")
    docs = loader.load()

    embeddings = OllamaEmbeddingWrapper(model = EMBEDDING_MODEL)

    print("Filtering docs")
    
    filtered_docs = []
    for doc in docs:
        if doc.page_content.strip():
            filtered_docs.append(doc)
    
    #TODO: this will be used in the future for better query
    # print("Processing vault")
    # process_vault(vault_path)
    
    
    vector_store = FAISS.from_documents(filtered_docs, embeddings)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH+ "./faiss_index/files"), exist_ok=True)
    vector_store.save_local(OUTPUT_PATH+"./faiss_index/files")
    print(f"Vector index saved to {OUTPUT_PATH}/faiss_index/files")

    texts = [doc.page_content for doc in docs]
    tokenized_texts = [text.split() for text in texts]
    bm25_index = BM25Okapi(tokenized_texts)
    documents = docs
    
    with open(OUTPUT_PATH+"./faiss_index/files/bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    with open(OUTPUT_PATH+"./faiss_index/files/documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    
    return (embeddings, vector_store, bm25_index, documents)


def clean_markdown(content):
    """Versão mais eficiente de limpeza"""
    patterns = [
        (r'^---\n.*?\n---\n', ''),  # Front matter
        (r'```.*?```', ''),  # Code blocks
        (r'!?\[\[.*?\]\]', ''),  # Links/embeds
        (r'[#*_~`]', ''), 
        (r'\s+', ' ')  # Espaços extras
    ]
    for pattern, repl in patterns:
        content = re.sub(pattern, repl, content, flags=re.DOTALL)
    return content.strip()



def process_file(filepath):
    """Processa um único arquivo com tratamento de erro"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        clean_content = clean_markdown(content)
        if not clean_content or len(clean_content) < 50:
            return None
            
        chunks = [clean_content[i:i+256] 
                 for i in range(0, len(clean_content), 256 - 30)]
    except Exception as e:
        print(f"Erro em {filepath.name}: {str(e)}")
        
    embeddings = []

    for chunk in chunks:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": EMBEDDING_MODEL, "prompt": chunk},
            timeout=60
        )

        try:
            res = response.json()
            
            if "embedding" not in res or not res["embedding"]:
                raise ValueError(f"Resposta inválida: {res}")
            embeddings.append(res["embedding"])
        except Exception as err:
            raise ValueError(f"Falha ao processar chunk com resposta: {response.text[:200]} - Erro: {err}")
        try:
            return {
                "embedding": embeddings,
                "chunks": chunks
            }
        except Exception as err:
            raise ValueError(f"Falha ao processar chunk com resposta: {response.text[:200]} - Erro: {err}")
        return None


def process_vault(vault_path):
    os.makedirs(OUTPUT_PATH+"./vector_db", exist_ok=True)
    
    md_files = list(Path(vault_path).rglob("*.md"))
    
    all_embeddings = []
    all_metadata = []
    
    for i in tqdm(range(0, len(md_files), 1), desc="Processando lotes"):
        batch = md_files[i:i+1]
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_file, f) for f in batch]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_embeddings.extend(result["embedding"])
                    for j, chunk in enumerate(result["chunks"]):
                        all_metadata.append({
                            "source": "path",
                            "chunk_id": j,
                            "text_preview": chunk[:150] + "..."
                        })
        
        if all_embeddings:
            np.save(os.path.join(OUTPUT_PATH, "embeddings_temp.npy"), np.array(all_embeddings))
            with open(os.path.join(OUTPUT_PATH, "metadata_temp.json"), 'w') as f:
                json.dump(all_metadata, f)
    
    # Salvar final
    np.save(os.path.join(OUTPUT_PATH, "embeddings.npy"), np.array(all_embeddings))
    with open(os.path.join(OUTPUT_PATH, "metadata.json"), 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nConcluído! {len(all_embeddings)} chunks processados")
    return all_embeddings
    



