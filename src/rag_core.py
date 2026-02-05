from pathlib import Path
from typing import Any
from src.utils.utils import load_docs, load_vault, load_config
from src.embedding.bm25 import generate_bm25_index, load_bm25_index
from src.embedding.vector_db import generate_vector_db, load_vector_db
import numpy as np
import json
import requests
from src.retrieve_context.hybrid_retriever import HybridRetriever
from src.retrieve_context.retrieve_context import gen_vector_results, gen_bm25_results

config = load_config()

class RAGSystem:
    def __init__(self: Any, vault_path: Path, update_vault: bool = False) -> None:
        self.vault_path = vault_path
        if update_vault:
            self.update_vault()

        else:
            try:
                self.load_embeddings()
            except Exception as e:
                print("could not load files, generating embeddings again")
                self.update_vault()
                
                
                
                

                
    #TODO: THIS PART WILL BE REFACTORED

        
    def generate_response(self: Any, query: Any, context: Any) -> Any :
        
        with open('./prompts/query_prompt.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        prompt_template = config["query"]["prompt"]
        
        prompt = prompts[prompt_template]["template"].format(context=context, query=query)

        try:
            ollama_endpoint = config["general"]["ollama_endpoint"]
            response = requests.post(
                ollama_endpoint + '/generate/',
                json={
                    "model": config["query"]["query_model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},
                    "top_p": 0.5,
                    "repeat_penalty": 1.2, 
                    "top_k": 20,
                }
            )
            response.raise_for_status()
            print("prompt tokens: ", response.json()["prompt_eval_count"])
            print("response tokens: ", response.json()["eval_count"])
            return response.json()["response"].strip()
        except Exception as e:
            return f"Error generating response: {e}"

        
    def process_query(self: Any, session_id: Any, query: Any) -> Any:
        # context = self.retrieve_context(query)
        vector_results = gen_vector_results(self.vector_db, query)
        bm25_results =  gen_bm25_results(self.docs, self.bm25_index, query)
        retriever = HybridRetriever ()
        top_docs = retriever.retrieve(query, vector_results,  bm25_results )
        context = ""
        for i, doc in enumerate(top_docs):
            context += f"\n**Document {i+1}:** {doc.page_content[:2000]}...\n"
            
        response = self.generate_response(query, context)
        return response, context   
    
    
    
    #TODO: REFAC CODE ABOVE
     
                
                
                
                
                
    def update_vault(self) -> bool:
        """
            Update embeddings from all files inside the vault

        Returns:
            bool: True if updated, False otherwise
        """       
        self.bm25_index = generate_bm25_index(self.vault_path)
        self.vector_db = generate_vector_db(self.vault_path)
        self.docs = load_vault(self.vault_path)  #TODO: change func name
        return True                              #TODO: remove redundant load/save   
    

    def load_embeddings(self) -> bool:
        self.bm25_index = load_bm25_index(self.vault_path)
        self.vector_db = load_vector_db(self.vault_path)
        self.docs = load_docs()
        return True
    