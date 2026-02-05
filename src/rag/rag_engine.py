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
from src.utils.token_accounting import get_accounting
from config import Config
from src.llm.factory import get_llm


logger = logging.getLogger(__name__)

class RAGEngine:
    """Orchestrates RAG pipeline: parsing -> retrieval -> context -> LLM"""
    
    def __init__(self, snapshot_date: Optional[str] = None, environment: str = 'prod'):
        self.snapshot_date = snapshot_date or datetime.now().strftime('%Y-%m-%d')
        self.environment = environment
        
        # ✅ INITIALIZE LLM FROM CONFIG
        self.llm = get_llm(
            provider=Config.LLM_PROVIDER,
            chat_model=Config.get_chat_model(),
            embed_model=Config.get_embed_model(),
            temperature=Config.LLM_TEMPERATURE
        )
            
        
        self.snapshot_date = snapshot_date or Config.DEV_SNAPSHOT_DATE
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
            embed_function=self.embed_query
        )
        
    
    def embed_query(self, query_text: str) -> list:
        """Embed query using LLM"""
        return self.llm.embed(query_text)
        
    
    def build_context(self,chunks: list[Dict], constraints:dict) -> str:
        """Build formatted context from chunks"""
        if not chunks:
            return "Aucun événement n'a été trouvé pour votre recherche."
        
        context = "Voici les événements pertinents trouvés :\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get('title', 'Sans titre')
            city = chunk.get('city', 'Lieu non spécifié')
            dates = self.retriever._matches_date(chunk, constraints['date'],only_first=False)
            dates = " + ".join(date.strftime("%d/%m/%Y, %H:%M:%S") for date in dates)
            text = chunk.get('text', 'Description non disponible')
            url = chunk.get('url', '')
            distance = chunk.get('distance')
            
            context += f"{i}. **{title}**\n"
            context += f"   📍 Lieu: {city}\n"
            context += f"   📅 Dates: {dates}\n"
            
            if distance:
                relevance = int(distance * 100)
                context += f"   ⭐ Pertinence: {relevance}%\n"
            
            context += f"\n   {text}\n"
            
            if url:
                context += f"   🔗 {url}\n"
            
            context += "\n"
        
        return context



    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        snapshot_date: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict:
        """Answer a question using RAG"""
        start_time = time.time()
        
        try:
            # Step 1: Parse constraints
            constraints = self.query_parser.parse_constraints(question, snapshot_date = self.snapshot_date)
            accounting = get_accounting()
           
            # Step 2: Retrieve chunks
            chunks = self.retriever.retrieve(
                query_text=question,
                k=top_k * 2,
                date_constraint=constraints['date'],
                city_constraint=constraints['city'],
                dept_constraint=constraints['dept'],
            )
            chunks = chunks[:top_k]
            
            # Step 3: Build context
            context = self.build_context(chunks, constraints)
            
            # Step 4: Generate answer with LLM
            prompt = self._build_prompt(question, context)
            answer = self._generate_answer(prompt,temperature = temperature)

            # Step 5: Format response
            sources = []
            for chunk in chunks:
                dates = self.retriever._matches_date(chunk, constraints['date'],only_first=False)
                dates = " et ".join(date.strftime("%d/%m/%Y, %H:%M:%S") for date in dates)
                sources .append(
                    {
                        'event_id': chunk.get('event_id'),
                        'title': chunk.get('title'),
                        'city': chunk.get('city'),
                        'dept': chunk.get('dept'),
                        'dates': dates,
                        'url': chunk.get('url'),
                        'distance': chunk.get('distance')
                    }
                )
            
            execution_time = time.time() - start_time
            
            # ✅ LOG TOKENS
            query_tokens = len(question.split()) * 1.3
            context_tokens = len(context.split()) * 1.3
            llm_tokens = len(answer.split()) * 1.3
            
            get_accounting().log_search(
                query_tokens=int(query_tokens),
                context_tokens=int(context_tokens),
                llm_tokens=int(llm_tokens),
                operation='search'
            )
            
            return {
                'answer': answer,
                'sources': sources,
                'constraints': {
                    'date': constraints['date'],
                    'city': constraints['city'],
                    'dept': constraints['dept']
                },
                'total_tokens': int(query_tokens + context_tokens + llm_tokens),
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
    
    def _generate_answer(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate answer using LLM"""
        try:
            return self.llm.generate(prompt, temperature=temperature)

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "Désolé, je n'ai pas pu générer une réponse."
