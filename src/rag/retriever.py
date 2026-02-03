# src/rag/retriever.py
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime

class RAGRetriever:
    """Retrieve and filter chunks from Faiss index"""
    
    def __init__(self, faiss_index, metadata: Dict, embed_function: Callable):
        """Initialize retriever"""
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embed_function = embed_function
    
    def retrieve(
        self,
        query_text: str,
        k: int = 5,
        date_constraint: Optional[tuple[datetime,int]] = None,
        city_constraint: Optional[str] = None,
        dept_constraint: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve and filter chunks"""
        
        # Embed query
        query_embedding = self.embed_function(query_text)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search Faiss
        search_k = k * 50
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
        
        return filtered_results[:k]
    
    def _filter_chunks(
        self,
        chunks: List[Dict],
        date_constraint: Optional[tuple[datetime,int]] = None,
        city_constraint: Optional[str] = None,
        dept_constraint: Optional[str] = None,
    ) -> List[Dict]:
        """Filter chunks by date and city"""
        filtered = chunks
        
        # Filter by date
        if date_constraint:
            filtered = [
                chunk for chunk in filtered
                if len(self._matches_date(chunk, date_constraint)) > 0
            ]
        
        # Filter by city
        if city_constraint:
            filtered = [
                chunk for chunk in filtered
                if self._matches_city(chunk, city_constraint)
            ]
        
        # Filter by department
        if dept_constraint:
            filtered = [
                chunk for chunk in filtered
                if self._matches_dept(chunk, dept_constraint)
            ]
        
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
            chunk_city = chunk.get('city', '').lower()
            chunk_address = chunk.get('address', '').lower()
            return target_city in chunk_city.lower() or target_city in chunk_address.lower()
        except:
            return False
    
    @staticmethod
    def _matches_dept(chunk: Dict, target_city: str) -> bool:
        """Check if chunk department matches target city"""
        try:
            chunk_dept = chunk.get('dept', '').lower()
            return chunk_dept == target_city.lower()
        except:
            return False
