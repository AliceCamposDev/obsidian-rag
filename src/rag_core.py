
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import requests
from src.embedding.generate_enbeddings import OllamaEmbeddingWrapper, setup_indexes
import json
import pickle
import src.utils.utils as utils

config = utils.load_config()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

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
        topK_pool = config["context"]["topK_files_pool"]
        topK = config["context"]["topK_files"]
    
        vector_results = self.vector_store.similarity_search(query = query, k=topK_pool)
        
        tokenized_query = query.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:topK_pool]
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
                
                top_docs = [doc for doc, _ in scored_docs[:topK]]
            except Exception as e:
                print(f"Erro no re-ranking: {e}")
                top_docs = all_results[:topK]
        else:
            top_docs = all_results[:topK]
            
        
        

        context = ""
        for i, doc in enumerate(top_docs):
            context += f"\n**Document {i+1}:** {doc.page_content[:2000]}...\n"

        return context
    
    def generate_response(self, query, context):
        
        with open('./prompts/query_prompt.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        prompt_template = config["query"]["prompt"]
        
        prompt = prompts[prompt_template]["template"].format(context=context, query=query)

        try:
            response = requests.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": config["query"]["query_model"],
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
