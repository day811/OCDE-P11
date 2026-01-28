# src/vectorization.py
"""
Vectorization Module : Chunking, Embedding, Indexation Faiss
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from mistralai import Mistral
import faiss

logger = logging.getLogger(__name__)

class EventVectorizer:
    """
    Convertit les événements textes en vecteurs indexés dans Faiss.
    """
    
    def __init__(self, model_name: str = "mistral-embed", api_key: str = ""):
        self.model_name = model_name
        self.client = Mistral(api_key=api_key)
        self.chunks_to_metadata = {}
    
    def chunk_events(self, events: List[Dict], chunk_size: int = 500) -> List[Dict]:
        """
        Découper les descriptions d'événements en chunks.
        """
        chunks = []
        chunk_id = 0
        
        for event in events:
            # Combiner tous les textes de l'événement
            full_text = f"{event['titlefr']}. {event['descriptionfr']}. {event['conditionsfr']}"
            
            # Chunking intelligent (par phrases)
            sentences = full_text.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append({
                            'chunk_id': chunk_id,
                            'event_id': event['uid'],
                            'text': current_chunk.strip(),
                            'event': event  # Garde l'événement original
                        })
                        chunk_id += 1
                    current_chunk = sentence + ". "
            
            # Dernier chunk
            if current_chunk:
                chunks.append({
                    'chunk_id': chunk_id,
                    'event_id': event['uid'],
                    'text': current_chunk.strip(),
                    'event': event
                })
                chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks from {len(events)} events")
        return chunks
    
    def vectorize_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Créer les embeddings pour chaque chunk via Mistral.
        """
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Vectorizing {len(texts)} chunks with {self.model_name}...")
        
        # Appel à l'API Mistral
        response = self.client.embeddings.create(
            model=self.model_name,
            inputs=texts
        )
        
        embeddings = np.array([
            embed.embedding for embed in response.data
        ]).astype('float32')
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexIVFFlat:
        """
        Créer un index Faiss optimisé.
        """
        dimension = embeddings.shape[1]  # 1024 pour Mistral
        
        # IVF : optimisé pour medium-sized datasets
        nlist = min(100, max(10, len(embeddings) // 100))
        
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        logger.info(f"Training index with {len(embeddings)} vectors...")
        index.train(embeddings)
        index.add(embeddings)
        
        logger.info(f"✅ Faiss index created (nlist={nlist}, vectors={len(embeddings)})")
        return index
    
    def save_index(self, index: faiss.IndexIVFFlat, snapshot_date: str) -> str:
        """
        Sauvegarder l'index Faiss.
        """
        from config import Config
        
        index_path = Config.get_index_path(snapshot_date)
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, index_path)
        logger.info(f"Index saved to {index_path}")
        return index_path
    
    def save_metadata(self, chunks: List[Dict], snapshot_date: str) -> str:
        """
        Sauvegarder les métadonnées des chunks.
        """
        from config import Config
        
        metadata_path = Config.get_metadata_path(snapshot_date)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Format : {index: {event_id, chunk_id, text, ...}}
        metadata = {}
        for i, chunk in enumerate(chunks):
            metadata[i] = {
                'event_id': chunk['event_id'],
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'title': chunk['event']['titlefr'],
                'city': chunk['event']['locationcity'],
                'date': chunk['event']['timings'][0]['begin'] if chunk['event']['timings'] else None,
                'url': chunk['event']['canonicalurl']
            }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to {metadata_path}")
        return metadata_path
    
    
    def load_processed_events(self, processed_path: str) -> List[Dict]:
            """
            Load processed events from JSON file.
            
            Args:
                processed_path: Path to processed events JSON
                
            Returns:
                List of event dictionaries
            """
            logger.info(f"Loading processed events from {processed_path}...")
            
            with open(processed_path, 'r', encoding='utf-8') as f:
                events = json.load(f)
            
            logger.info(f"Loaded {len(events)} events")
            return events