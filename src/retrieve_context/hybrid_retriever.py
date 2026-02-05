from sentence_transformers import CrossEncoder
from src.utils.utils import load_config
from typing import Any, Union, List, Tuple, Optional, Dict
from langchain_core.documents import Document
from src.retrieve_context.retrieve_context import gen_vector_results, gen_bm25_results
config = load_config()

class HybridRetriever:
    def __init__(self, cross_encoder_model: str = config["cross_encoder"]["model"], device: str = config["general"]["device"]):
        try:
            self.model: Any = CrossEncoder(
                cross_encoder_model, 
                device = device
            )
            print(f"[SUCESSO] Modelo carregado")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo: {e}")
            raise e
        
        # vector_results
        # b25_results
        
    def retrieve(
        self,
        query: str,
        vector_results: List[Document] ,
        bm25_results: List[Document] ,
        top_k: int = config["context"]["topK_files_pool"],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        use_reranking: bool = True
    ) -> List[Document]:
        
        # Normalizar pesos
        total_weight = vector_weight + bm25_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            bm25_weight = bm25_weight / total_weight
        
        # Criar dicionário para combinar resultados
        combined_scores: Dict[Any, Any] = {}
        
        # Adicionar scores dos resultados vetoriais
        for i, doc in enumerate(vector_results):
            score = 1.0 - (i / len(vector_results)) 
            if doc.page_content in combined_scores:
                combined_scores[doc.page_content]['score'] += score * vector_weight
                combined_scores[doc.page_content]['sources'].append('vector')
            else:
                combined_scores[doc.page_content] = {
                    'document': doc,
                    'score': score * vector_weight,
                    'sources': ['vector']
                }
        
        # Adicionar scores dos resultados BM25
        for i, doc in enumerate(bm25_results):
            score = 1.0 - (i / len(bm25_results))  # Score normalizado baseado na posição
            if doc.page_content in combined_scores:
                combined_scores[doc.page_content]['score'] += score * bm25_weight
                combined_scores[doc.page_content]['sources'].append('bm25')
            else:
                combined_scores[doc.page_content] = {
                    'document': doc,
                    'score': score * bm25_weight,
                    'sources': ['bm25']
                }
        
        # Ordenar por score combinado
        sorted_docs = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Selecionar top_k documentos
        top_docs = [item['document'] for item in sorted_docs[:top_k]]
        
        # Aplicar reranking com cross-encoder se solicitado
        if use_reranking and self.model and top_docs:
            return self._rerank_with_cross_encoder(query, top_docs)
        
        return top_docs

    def _rerank_with_cross_encoder(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Aplica reranking usando o cross-encoder.
        
        Args:
            query: Consulta do usuário
            documents: Lista de documentos para reranking
            
        Returns:
            Lista de documentos rerankeada
        """
        try:
            # Preparar pares (query, documento) para o cross-encoder
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Obter scores do cross-encoder
            scores = self.model.predict(pairs)
            
            # Combinar scores com documentos
            scored_docs = list(zip(documents, scores))
            
            # Ordenar por score do cross-encoder
            reranked_docs = sorted(
                scored_docs,
                key=lambda x: x[1],
                reverse=True
            )
            
            # Retornar apenas os documentos (sem os scores)
            return [doc for doc, _ in reranked_docs]
            
        except Exception as e:
            print(f"[AVISO] Erro no reranking com cross-encoder: {e}")
            return documents