# src/rag/retriever.py
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime, timedelta
from src.utils.utils import normalize_str, flat_date_constraints
from src.utils.token_accounting import get_accounting

class RAGEngine:
    """Retrieve and filter chunks from Faiss index"""
    
    def __init__(self, faiss_index, metadata: Dict, embed_function: Callable, top_k:int = 5):
        """Initialize retriever"""
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embed_function = embed_function
        self.top_k = top_k
    

    def retrieve(
        self,
        query_text: str,
        k: int = 0,
        date_constraint: Optional[tuple[datetime,int]] = None,
        city_constraint: Optional[str] = None,
        dept_constraint: Optional[str] = None,
    ) -> Tuple[List[Dict], int] :
        """Retrieve and filter chunks"""
        
        if k == 0:
            k = self.top_k
        # Embed query
        query_text += flat_date_constraints(date_constraint)
        query_embedding = self.embed_function(query_text)

        embed_tokens = int(len(query_text.split()) * 1.3) # Approximate

        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search Faiss
        search_k = max(k * 70,500)
        distances, indices = self.faiss_index.search(query_vector, search_k)
        
        # Build results with distances
        results = []
        seen= []
        for idx, (dist, chunk_id) in enumerate(zip(distances[0], indices[0])):
            if chunk_id >= 0:
                chunk_metadata = self.metadata.get(str(chunk_id), {})
                if chunk_metadata['event_id'] not in seen:
                    chunk_metadata['distance'] = float(dist)
                    chunk_metadata['rank'] = idx
                    results.append(chunk_metadata)
                    seen.append(chunk_metadata['event_id'])
        
        # Apply filters
        filtered_results = self._filter_chunks(
            results,
            date_constraint=date_constraint,
            city_constraint=city_constraint,
            dept_constraint=dept_constraint,
       )
        
        return (filtered_results[:k], embed_tokens)
    
    def _filter_chunks(
        self,
        chunks: List[Dict],
        date_constraint: Optional[tuple[datetime,int]] = None,
        city_constraint: Optional[str] = None,
        dept_constraint: Optional[str] = None,
    ) -> List[Dict]:
        """Filter chunks by date and city"""
        filtered = []
        for chunk in chunks:
        # Filter by date
            if date_constraint and len(self._matches_date(chunk, date_constraint)) <= 0:
                continue
        
            # Filter by city
            if city_constraint and not self._matches_city(chunk, city_constraint):
                continue
            
            # Filter by department
            if dept_constraint and not self._matches_dept(chunk, dept_constraint):
                continue
            filtered.append(chunk)
            
        return filtered
    

    @staticmethod
    def _matches_date(chunk: Dict, date_range: Optional[tuple[datetime,int]],only_first=True) -> list:
        """Check if chunk date matches target date"""
        selected_dates = []
        try:
            timings = chunk.get('dates', '')
            if date_range:
                for timing in timings:
                    begin = timing['begin']
                    chunk_date = datetime.fromisoformat(begin)
                    target = date_range[0].date()
                    tolerance_days = date_range[1]
                    delta = (chunk_date.date() - target).days
                    if delta >=0 and delta <= tolerance_days :
                        selected_dates.append(chunk_date)
                        if only_first : return selected_dates
            return selected_dates
        except:
            return []
    
    @staticmethod
    def _matches_city(chunk: Dict, target_city: str) -> bool:
        """Check if chunk city matches target city"""
        try:
            chunk_city = normalize_str(chunk.get('city', ''))
            chunk_address = normalize_str(chunk.get('address', ''))
            target_city = normalize_str(target_city)
            return target_city in chunk_city or target_city in chunk_address
        except:
            return False
    
    @staticmethod
    def _matches_dept(chunk: Dict, target_dept: str) -> bool:
        """Check if chunk department matches target city"""
        try:
            chunk_dept = normalize_str(chunk.get('dept', ''))
            target_dept = normalize_str(target_dept)
            return chunk_dept == target_dept
        except:
            return False

    def build_context(self,chunks: list[Dict], constraints:dict) -> str:
        """Build formatted context from chunks"""
        if not chunks:
            return "Aucun événement n'a été trouvé pour votre recherche."
        
        context = "Voici les événements pertinents trouvés :\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get('title', 'Sans titre')
            city = chunk.get('city', 'Lieu non spécifié')
            dates = self._matches_date(chunk, constraints['date'],only_first=False)
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

