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
        date_constraint: Optional[datetime] = None,
        city_constraint: Optional[str] = None,
        date_tolerance_days: int = 0
    ) -> List[Dict]:
        """Retrieve and filter chunks"""
        
        # Embed query
        query_embedding = self.embed_function(query_text)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search Faiss
        search_k = k * 10
        distances, indices = self.faiss_index.search(query_vector, search_k)
        
        # Build results with distances
        results = []
        for idx, (dist, chunk_id) in enumerate(zip(distances[0], indices[0])):
            if chunk_id >= 0:
                chunk_metadata = self.metadata.get(str(chunk_id), {})
                chunk_metadata['distance'] = float(dist)
                chunk_metadata['rank'] = idx
                results.append(chunk_metadata)
        
        # Apply filters
        filtered_results = self._filter_chunks(
            results,
            date_constraint=date_constraint,
            city_constraint=city_constraint,
            date_tolerance_days=date_tolerance_days
        )
        
        return filtered_results[:k]
    
    def _filter_chunks(
        self,
        chunks: List[Dict],
        date_constraint: Optional[datetime] = None,
        city_constraint: Optional[str] = None,
        date_tolerance_days: int = 0
    ) -> List[Dict]:
        """Filter chunks by date and city"""
        filtered = chunks
        
        # Filter by date
        if date_constraint:
            filtered = [
                chunk for chunk in filtered
                if self._matches_date(chunk, date_constraint, date_tolerance_days)
            ]
        
        # Filter by city
        if city_constraint:
            filtered = [
                chunk for chunk in filtered
                if self._matches_city(chunk, city_constraint)
            ]
        
        return filtered
    

    @staticmethod
    def _matches_date(chunk: Dict, target_date: datetime, tolerance_days: int = 0) -> bool:
        """Check if chunk date matches target date"""
        try:
            timings = chunk.get('dates', '')
            
            for timing in timings:
                begin = timing['begin']
                chunk_date = datetime.fromisoformat(begin).date()
                target = target_date.date()
                delta = abs((chunk_date - target).days)
                if delta <= tolerance_days :
                    return True
            return False
        except:
            return False
    
    @staticmethod
    def _matches_city(chunk: Dict, target_city: str) -> bool:
        """Check if chunk city matches target city"""
        try:
            chunk_city = chunk.get('city', '').lower()
            return chunk_city == target_city.lower()
        except:
            return False
