from src.utils.utils import load_config
from typing import Any, List
import numpy as np
from langchain_core.documents import Document
config = load_config()

#TODO: type hint 4 vector db
def gen_vector_results(vector_db: Any, query: str) -> Any:
    topK_pool = config["context"]["topK_files_pool"]
    topK = config["context"]["topK_files"]
    vector_results = vector_db.similarity_search(query = query, k=topK_pool)
    return vector_results
    
def gen_bm25_results(docs: List[Document], bm25_index: Any, query: str) -> Any:
    tokenized_query = query.split() #TODO: preprocess 
    bm25_scores = bm25_index.get_scores(tokenized_query)
    topK_pool = config["context"]["topK_files_pool"]
    top_indices = np.argsort(bm25_scores)[::-1][:topK_pool]
    bm25_results = [docs[i] for i in top_indices]
    return bm25_results

