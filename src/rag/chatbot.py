# src/rag/chatbot.py
"""
Interactive chatbot interface using LangChain RAG
"""
from src.rag.langchain_bridge import LangChainRAG
from src.rag.retriever import RAGRetriever
from config import Config
from src.rag.query_parser import QueryParser
from src.utils.token_accounting import get_accounting
import logging

logger = logging.getLogger(__name__)

class ChatBot:
    """Chatbot for event recommendations"""
    
    def __init__(self, snapshot_date=Config.DEV_SNAPSHOT_DATE, mode: str = 'CLI'):
        import faiss
        import json
        from pathlib import Path

        # Load Faiss index and metadata
        
        index_path = Config.get_index_path(snapshot_date)
        metadata_path = Config.get_metadata_path(snapshot_date)
        
        self.faiss_index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Initialize components
        self.query_parser = QueryParser()
        self.retriever = RAGRetriever(
            faiss_index=self.faiss_index,
            metadata=self.metadata,
            embed_function=self._embed_query
        )
        
        output_format = "HTML" if mode == "UI" else "Markdown"
        
        # Initialize LangChain RAG
        self.rag = LangChainRAG(
            rag_retriever=self.retriever,
            query_parser=self.query_parser,
            mistral_api_key=Config.MISTRAL_API_KEY
        )
        
        logger.info("ChatBot initialized with LangChain RAG")
    
    def _embed_query(self, query_text: str):
        """Embed query using Mistral"""
        from src.vector.vectorization import EventVectorizer
        
        vectorizer = EventVectorizer(
            model_name='mistral-embed',
            api_key=Config.MISTRAL_API_KEY
        )
        
        embeddings = vectorizer.vectorize_chunks(
            [{'chunk_id': 0, 'event_id': 'query', 'text': query_text}],
            batch_size=1
        )
        
        return embeddings[0].tolist()
    
    def chat(self, user_question: str) -> dict:
        """Process user question through LangChain RAG"""
        logger.info(f"User question: {user_question}")
        
        result = self.rag.answer(user_question)
        

        return {
            'question': user_question,
            'answer': result['answer'],
            'sources': result['sources'],
            'success': result['success'],
            'total_tokens': result['total_tokens']
        }
    
    def interactive_session(self):
        """Run interactive chat session"""
        print("\n" + "="*80)
        print("CHATBOT ÉVÉNEMENTS CULTURELS - Tapez 'quit' pour quitter")
        print("="*80 + "\n")
        
        while True:
            try:
                question = input("Vous: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Au revoir!")
                    break
                
                if not question:
                    continue
                
                result = self.chat(question)
                
                print(f"\nBot: {result['answer']}\n")
                
                if result['sources']:
                    print(f"Sources ({len(result['sources'])}):")
                    for doc in result['sources']:
                        print(f"  - {doc.metadata.get('source', 'N/A')}")
                
                print("-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\nSession fermée.")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Erreur: {e}\n")
