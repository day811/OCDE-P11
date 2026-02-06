# src/core/search_engine.py
"""
Unified Search Engine Interface
Abstraction commune pour RAGEngine et ChatBot
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from src.rag.rag_engine import RAGEngine
from src.rag.chatbot import ChatBot
from src.utils.token_accounting import get_accounting
from config import Config

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Interface unifiée pour recherche et chat.
    Abstrait les détails d'implémentation de RAGEngine et ChatBot.
    """
    
    def __init__(self, snapshot_date: str = "", mode: str = 'search'):
        """
        Initialize SearchEngine.
        
        Args:
            snapshot_date: Date du snapshot (YYYY-MM-DD). Default: Config.DEV_SNAPSHOT_DATE
            mode: 'search' pour recherche simple, 'chat' pour chat interactif
        """
        self.snapshot_date = snapshot_date or Config.DEV_SNAPSHOT_DATE
        self.mode = mode
        self.start_time = datetime.now()
        
        logger.info(f"Initializing SearchEngine (mode={mode}, snapshot={self.snapshot_date})")
        
        try:
            if mode == 'search':
                self.engine = RAGEngine(snapshot_date=self.snapshot_date)
            elif mode == 'chat':
                self.engine = ChatBot(snapshot_date=self.snapshot_date)
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'search' or 'chat'.")
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise
    
    def query(self, question: str, top_k: int = 5, 
              temperature: float = 0.7) -> Dict:
        """
        Execute unified query (compatible with both RAGEngine and ChatBot).
        
        Args:
            question: User question/query
            top_k: Number of top results to retrieve
            temperature: LLM temperature (0.0-1.0), ignored for pure RAG
            
        Returns:
            Dictionary with keys:
            - 'answer': str - Main answer/response
            - 'sources': List[Dict] - Source events
            - 'total_tokens': int - Tokens used
            - 'execution_time': float - Query execution time in seconds
            - 'constraints': Dict - Extracted constraints (date, city)
        """
        
        query_start = datetime.now()
        
        try:
            if self.mode == 'search':
                result = self._handle_search(question, top_k, temperature)
            elif self.mode == 'chat':
                result = self._handle_chat(question, top_k, temperature)
            
            # Calculate execution time
            execution_time = (datetime.now() - query_start).total_seconds()
            result['execution_time'] = execution_time
            
            logger.info(f"Query completed in {execution_time:.3f}s | Tokens: {result.get('total_tokens', 0)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise
    
    def _handle_search(self, question: str, top_k: int, 
                       temperature: float) -> Dict:
        """Handle search mode query (RAGEngine)."""
        
        logger.debug(f"Handling search query: {question}")
        
        result = self.engine.answer_question(
            question=question,
            top_k=top_k,
            snapshot_date=self.snapshot_date
        )
        
        return {
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'total_tokens': result.get('total_tokens', 0),
            'execution_time': result.get('execution_time', 0),
            'constraints': result.get('constraints', {}),
            'mode': 'search'
        }
    
    def _handle_chat(self, question: str, top_k: int, 
                     temperature: float) -> Dict:
        """Handle chat mode query (ChatBot with LangChain)."""
        
        logger.debug(f"Handling chat query: {question}")
        
        result = self.engine.chat(
            user_question=question,
            top_k=top_k,
            temperature=temperature
        )
        
        return {
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'total_tokens': result.get('total_tokens', 0),
            'execution_time': result.get('execution_time', 0),
            'constraints': result.get('constraints', {}),
            'mode': 'chat',
            'conversation_id': getattr(self.engine, 'conversation_id', None)
        }
    
    def get_session_accounting(self) -> Dict:
        """Get token accounting for current session."""
        accounting = get_accounting()
        return accounting.get_session_report()
    
    def reset_session(self):
        """Reset session accounting."""
        from src.utils.token_accounting import reset_accounting
        reset_accounting()
        logger.info("Session accounting reset")
