# src/rag/rag_engine.py
import logging
import time
import json
import faiss
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from src.rag.query_parser import QueryParser
from src.rag.retriever import RAGRetriever
from src.rag.context_builder import ContextBuilder
from config import Config


logger = logging.getLogger(__name__)

class RAGEngine:
    """Orchestrates RAG pipeline: parsing -> retrieval -> context -> LLM"""
    
    def __init__(self, snapshot_date: Optional[str] = None, environment: str = 'prod'):
        """Initialize RAG Engine"""
        
        self.snapshot_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
        self.environment = environment
        
        # Load Faiss index
        index_path = Config.get_index_path(self.snapshot_date)
        self.faiss_index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = Config.get_metadata_path(self.snapshot_date)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize components
        self.query_parser = QueryParser
        self.retriever = RAGRetriever(
            faiss_index=self.faiss_index,
            metadata=self.metadata,
            embed_function=self._embed_query
        )
        self.context_builder = ContextBuilder
        
        # Initialize Mistral LLM
        from mistralai import Mistral, UserMessage
        self.llm = Mistral(api_key=Config.MISTRAL_API_KEY)
    
    def _embed_query(self, query_text: str):
        """Embed query using Mistral"""
        from src.vector.vectorization import EventVectorizer
        
        vectorizer = EventVectorizer(
            model_name="mistral-embed",
            api_key=Config.MISTRAL_API_KEY
        )
        embeddings = vectorizer.vectorize_chunks(
            [{'chunk_id': 0, 'event_id': 'query', 'text': query_text}],
            batch_size=1
        )
        return embeddings[0].tolist()
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        snapshot_date: Optional[str] = None
    ) -> Dict:
        """Answer a question using RAG"""
        start_time = time.time()
        
        try:
            # Step 1: Parse constraints
            constraints = self.query_parser.parse_constraints(question)
            
            # Step 2: Retrieve chunks
            chunks = self.retriever.retrieve(
                query_text=question,
                k=top_k * 2,
                date_constraint=constraints['date'],
                city_constraint=constraints['city']
            )
            chunks = chunks[:top_k]
            
            # Step 3: Build context
            context = self.context_builder.build_context(chunks)
            
            # Step 4: Generate answer with LLM
            prompt = self._build_prompt(question, context)
            answer = self._generate_answer(prompt)
            
            # Step 5: Format response
            sources = [
                {
                    'event_id': chunk.get('event_id'),
                    'title': chunk.get('title'),
                    'city': chunk.get('city'),
                    'date': chunk.get('date'),
                    'url': chunk.get('url'),
                    'distance': chunk.get('distance')
                }
                for chunk in chunks
            ]
            
            execution_time = time.time() - start_time
            
            return {
                'answer': answer,
                'sources': sources,
                'constraints': {
                    'date': constraints['date'].isoformat() if constraints['date'] else None,
                    'city': constraints['city']
                },
                'execution_time': execution_time
            }
        
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            raise
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM"""
        return f"""Tu es un assistant pour recommander des événements.

Contexte (événements trouvés):
{context}

Question: {question}

Basé sur le contexte, fournis une réponse concise recommandant les événements pertinents."""
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM"""
        try:
            llm_response = self.llm.chat.complete(
                model="mistral-small",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return llm_response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "Désolé, je n'ai pas pu générer une réponse."
