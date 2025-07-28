
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import random
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import TFIDFRetriever
from sentence_transformers import CrossEncoder
import requests
from src.embedding.generate_enbeddings import OllamaEmbeddingWrapper, setup_indexes
import json
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


VECTOR_DB_PATH = "vector_db_optimized"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
EMBEDDING_ENDPOINT = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "bge-m3:567m"

MODEL_NAME = "mistral:latest"

TOP_K = 5
LOG_FILE = "chat_history.log"


class LocalRAGSystem:
    def __init__(self, vault_path, update_vault = False):
        self.vault_path = vault_path
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        if update_vault:
            self.embeddings, self.vector_store, self.bm25_index, self.documents = setup_indexes(vault_path)
        else:
            self.vector_store = FAISS.load_local("./data/faiss_index/files", OllamaEmbeddingWrapper(), allow_dangerous_deserialization=True)
            with open("./data/faiss_index/files/bm25.pkl", "rb") as f:
                self.bm25_index = pickle.load(f)

            with open("./data/faiss_index/files/documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
        
        

    
    def retrieve_context(self, query):
        TOP_K_INITIAL = 20
        TOP_K_FINAL = 5
    
        vector_results = self.vector_store.similarity_search(query = query, k=TOP_K_INITIAL)
        
        tokenized_query = query.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:TOP_K_INITIAL]
        bm25_results = [self.documents[i] for i in top_indices]
        
        all_results = []
        seen_ids = set()
        top_docs = []
        for doc in vector_results + bm25_results:
            doc_id = doc.metadata.get('source', '') + str(hash(doc.page_content[:100]))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append(doc)
        
        if hasattr(self, 'cross_encoder') and self.cross_encoder:
            pairs = [(query, doc.page_content) for doc in all_results]
            try:
                scores = self.cross_encoder.predict(pairs)
                
                scored_docs = list(zip(all_results, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                top_docs = [doc for doc, _ in scored_docs[:TOP_K_FINAL]]
            except Exception as e:
                print(f"Erro no re-ranking: {e}")
                top_docs = all_results[:TOP_K_FINAL]
        else:
            top_docs = all_results[:TOP_K_FINAL]
            
        
        

        context = ""
        for i, doc in enumerate(top_docs):
            context += f"\n**Document {i+1}:** {doc.page_content[:2000]}...\n"

        return context
    
    def generate_response(self, query, context):
        
        with open('./prompts/querry_prompt.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        prompt = prompts["phi3-rpg-pt"]["template"].format(context=context, query=query)

        try:
            response = requests.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},
                    "top_p": 0.9,
                    "repeat_penalty": 1.1, 
                    "top_k": 40,
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            return f"Error generating response: {e}"

        
    def process_query(self, session_id, query):
        context = self.retrieve_context(query)
        response = self.generate_response(query, context)
        return response, context
