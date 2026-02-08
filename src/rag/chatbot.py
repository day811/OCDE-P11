# src/rag/chatbot.py
"""
Interactive chatbot interface using LangChain RAG
"""
from src.rag.langchain_bridge import LangChainRAG
from src.rag.retriever import RAGRetriever
from config import Config
from src.rag.query_parser import QueryParser
from src.utils.token_accounting import get_accounting
from src.llm.factory import get_llm
import logging

logger = logging.getLogger(__name__)

class ChatBot:
    """Chatbot for event recommendations"""
    
    def __init__(self, snapshot_date=Config.DEV_SNAPSHOT_DATE):
#        import faiss
#        import json
        from pathlib import Path

        self.snapshot_date = snapshot_date
        # Load Faiss index and metadata
        
        self.llm = get_llm()
 
        # Initialize LangChain RAG
        self.rag = LangChainRAG(embed_function= self._embed_query )
        
        logger.info("ChatBot initialized with LangChain RAG")
    
    def _embed_query(self, query_text: str):
        """Embed query using Mistral"""
        
        return self.llm.embed(query_text)
        
   
    def chat(self, user_question: str, top_k: int = 5, temperature: float = 0.7) -> dict:
        """Process user question through LangChain RAG"""
        logger.info(f"User question: {user_question}")
        
        result = self.rag.answer(user_question, top_k=top_k, temperature=temperature)
        

        return {
            'question': user_question,
            'answer': result['answer'],
            'sources': result['sources'],
            'success': result['success'],
            'total_tokens': result['total_tokens']
        }
    