# src/rag/langchain_bridge.py
"""
LangChain integration bridge for RAG pipeline
Connects Faiss retriever + Mistral LLM
"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.utils.token_accounting import get_accounting

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class FaissRetrieverAdapter(BaseRetriever):
    """Adapter to use RAGRetriever as LangChain Retriever"""
    
    def __init__(self, rag_retriever, query_parser):
        super().__init__()
        # Lazy init pour éviter inspection prématurée
        object.__setattr__(self, '_rag_retriever', rag_retriever)  # Force set même avec __slots__
        object.__setattr__(self, '_query_parser', query_parser)

    @property
    def rag_retriever(self):
        return getattr(self, '_rag_retriever', None)

    @property
    def query_parser(self):
        return getattr(self, '_query_parser', None)

    
    def _get_relevant_documents(self, query: str) -> List[Document]: # type: ignore
        """Get relevant documents from Faiss through RAGRetriever"""
        # Parse constraints from query
        constraints = self.query_parser.parse_constraints(query) # type: ignore
        
        # Retrieve chunks using your existing RAGRetriever
        chunks = self.rag_retriever.retrieve( # type: ignore
            query_text=query,
            k=10,
            date_constraint=constraints['date'],
            city_constraint=constraints['city'],
            dept_constraint=constraints['dept']
        )
        
        # Convert chunks to LangChain Documents
        documents = []
        for chunk in chunks:
            dates = self.rag_retriever._matches_date(chunk, constraints['date'],only_first=False)
            dates = " et ".join(date.strftime("%d/%m/%Y, %H:%M:%S") for date in dates)
            doc_text = f"""
            Title: {chunk.get('title', 'N/A')}
            City: {chunk.get('city', 'N/A')}
            Dates: {dates}
            Description: {chunk.get('text', '')}
            URL: {chunk.get('url', 'N/A')}
            """
            documents.append(Document(
                page_content=doc_text,
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
    
    def __init__(self, rag_retriever, query_parser, mistral_api_key, output_fromat:str = "Markdown"):
        
        self.retriever = FaissRetrieverAdapter(rag_retriever, query_parser)
        
        # Initialize Mistral LLM
        self.llm = ChatMistralAI(
            model='mistral-small', # type: ignore
            api_key=mistral_api_key
        )
        
        # Define RAG prompt
        format = f"Tous les résultats doivent être formatés en {output_fromat}."
        self.prompt = PromptTemplate(
            input_variables=['context', 'question', ''],
            template="""Tu es un assistant expert en événements culturels en Occitanie.

Voici les événements pertinents trouvés :
{context}

Question de l'utilisateur : {question}

Fournis une recommandation personnalisée basée sur ces événements.
Sois concis et pertinent.
""" + format
        )
        
        # Create RAG chain with LangChain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',  # Simple context concatenation
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.prompt}
        )
    
    def answer(self, question: str) -> dict:
        """Answer a question using LangChain RAG"""
        total_tokens = 0
        try:
            result = self.qa_chain({'query': question})
            
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
