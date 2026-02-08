# src/rag/langchain_bridge.py
"""
LangChain integration bridge for RAG pipeline
Connects Faiss retriever + Mistral LLM
"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.utils.token_accounting import get_accounting
from src.llm.factory import get_langchain_llm
from src.rag.retriever import RAGRetriever
from src.rag.query_parser import QueryParser

from typing import List, Callable
import logging
from config import Config
import faiss
import json
from pathlib import Path



logger = logging.getLogger(__name__)

class FaissRetrieverAdapter(BaseRetriever):
    """Adapter to use RAGRetriever as LangChain Retriever"""
    
    def __init__(self, rag_retriever, query_parser,top_k):
        super().__init__()
        # Lazy init pour éviter inspection prématurée
        object.__setattr__(self, '_rag_retriever', rag_retriever)  # Force set même avec __slots__
        object.__setattr__(self, '_query_parser', query_parser)
        object.__setattr__(self, '_top_k', top_k)

    @property
    def rag_retriever(self):
        return getattr(self, '_rag_retriever', None)

    @property
    def query_parser(self):
        return getattr(self, '_query_parser', None)

    @property
    def top_k(self):
        return getattr(self, '_top_k', None)

   
    def _get_relevant_documents(self, query: str) -> List[Document]: # type: ignore
        """Get relevant documents from Faiss through RAGRetriever"""
        # Parse constraints from query
        constraints = self.query_parser.parse_constraints(query) # type: ignore
        
        # Retrieve chunks using your existing RAGRetriever
        chunks = self.rag_retriever.retrieve( # type: ignore
            query_text=query,
            date_constraint=constraints['date'],
            city_constraint=constraints['city'],
            dept_constraint=constraints['dept']
        )
        
        # Convert chunks to LangChain Documents
        chunks = chunks[:self.top_k]
        documents = []
        for chunk in chunks:
            dates = self.rag_retriever._matches_date(chunk, constraints['date'],only_first=False) # type: ignore
            dates = " et ".join(date.strftime("%d/%m/%Y, %H:%M:%S") for date in dates)
            doc_text = f"""
            Title: {chunk.get('title', 'N/A')}
            City: {chunk.get('city', 'N/A')}
            Dates: {dates}
            Description: {chunk.get('text', '')}
            URL: {chunk.get('url', 'N/A')}
            """
            documents.append(Document(
                page_content=chunk.get('text'),
                metadata={
                    'source': chunk.get('url'),
                    'event_id': chunk.get('event_id'),
                    'city': chunk.get('city'),
                    'dates': dates
                }
            ))
        
        logger.info(f"Retrieved {len(documents)} documents from Faiss")
        return documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version"""
        return self._get_relevant_documents(query)


class LangChainRAG:
    """RAG system using LangChain orchestration"""
    
    def __init__(self, embed_function: Callable, snapshot_date=Config.DEV_SNAPSHOT_DATE):
        
        self.snapshot_date = snapshot_date
        self.embed_function  = embed_function
        self.qa_chain = None
        
        # Define RAG prompt
        index_path = Config.get_index_path(self.snapshot_date)
        metadata_path = Config.get_metadata_path(self.snapshot_date)
        
        self.faiss_index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)


        # Initialize components
        self.query_parser = QueryParser()
        self.prompt = PromptTemplate(
            input_variables=['context', 'question', ''],
            template="""Tu es un assistant expert en événements culturels en Occitanie.

Voici les événements pertinents trouvés :
{context}

Question de l'utilisateur : {question}

Fournis une recommandation personnalisée basée sur ces événements.
Sois concis et pertinent.
""" 
        )
        
        # Create RAG chain with LangChain

    def make_chain_retriever(self, top_k: int =5, temperature:float=0.7):
        """
        Docstring for get_chain_retriever
        """
        # Initialize Mistral LLM
        self.llm = get_langchain_llm(temperature)
        self.temperature=temperature
        
        self.rag_retriever = RAGRetriever(
            faiss_index=self.faiss_index,
            metadata=self.metadata,
            embed_function=self.embed_function,
            top_k=top_k
        )

        self.retriever = FaissRetrieverAdapter(self.rag_retriever, self.query_parser, top_k=top_k)
        
        # Initialize LangChain RAG
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',  # Simple context concatenation
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.prompt}
        )
         



    def answer(self, question: str, top_k: int =5, temperature: float = 0.7 ) -> dict:
        """Answer a question using LangChain RAG"""
        
        if self.qa_chain is None or temperature !=  self.temperature:
            self.make_chain_retriever(top_k=top_k, temperature=temperature)

        total_tokens = 0
        try:
            result = self.qa_chain({'query': question}) # type: ignore
            
            # Estimate tokens (query embedding + context + generation)
            query_tokens = len(question.split()) * 1.3  # rough estimate
            context_tokens = len(str(result.get('source_documents', '')).split()) * 1.3
            llm_tokens = len(result['result'].split()) * 1.3
            total_tokens = int(query_tokens + context_tokens + llm_tokens)

            accounting = get_accounting()
            accounting.log_search(
                query_tokens=int(query_tokens),
                context_tokens=int(context_tokens),
                llm_tokens=int(llm_tokens),
                operation='chat'
            )
            return {
                'answer': result['result'],
                'sources': result.get('source_documents', []),
                'total_tokens': total_tokens,
                'success': True
            }
        except Exception as e:
            logger.error(f"LangChain RAG error: {e}")
            return {
                'answer': f"Erreur: {str(e)}",
                'sources': [],
                'total_tokens': total_tokens,
                'success': False
            }
