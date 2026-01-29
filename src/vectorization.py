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
from config import Config
import re

logger = logging.getLogger(__name__)

class EventVectorizer:
    """
    Convertit les événements textes en vecteurs indexés dans Faiss.
    """
    
    def __init__(self, model_name: str = "mistral-embed", api_key: str = ""):
        self.model_name = model_name
        self.client = Mistral(api_key=api_key)
        self.chunks_to_metadata = {}
    

    def split_text(self, input_text, chunk_size: int = 500, level=0):
        """
        Eclate le texte en fragments de taille <= chunksize
        Divise sur ". " et " " en cas d'insuffisance
        """
        sep = (". ", ", ", " ")
        sentences = input_text.split(sep[level])
        output_text_list=[]
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence)>chunk_size:
                if level < len(sep)-1:
                    mini_sentences =  self.split_text(sentence, chunk_size=chunk_size, level = level + 1)
                    output_text_list.extend(mini_sentences)
                else:
                    logger.error(f"\n❌ Chunks splitting - FAILED")
                    raise
            else:
                if sentence : output_text_list.append(sentence + sep[level] )   
        return output_text_list
    

    def chunk_events(self, events: List[Dict], chunk_size: int = 500) -> List[Dict]:
        """
        Découper les descriptions d'événements en chunks.
        """
        chunks = []
        chunk_id = 0
        
        for event in events:
            # Combiner tous les textes de l'événement
            full_text = ". ".join(event[field] for field in Config.CHUNK_FIELDS)
            full_text = re.sub(r"<.*?>", " ", full_text)
            # Chunking intelligent (par phrases puis mots)
            sentences = self.split_text(full_text,chunk_size=chunk_size)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append({
                            'chunk_id': chunk_id,
                            'event_id': event['uid'],
                            'text': current_chunk,
                        })
                        chunk_id += 1
                    current_chunk = sentence
        
            # Dernier chunk
            if current_chunk:
                chunks.append({
                    'chunk_id': chunk_id,
                    'event_id': event['uid'],
                    'text': current_chunk.strip(),
                })
                chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks from {len(events)} events")
        return chunks
    
    def vectorize_chunks(self, chunks: List[Dict], batch_size: int = 100) -> np.ndarray:
        """
        Create embeddings for each chunk via Mistral API.
        Processes chunks in batches to respect API limits and improve reliability.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks per API call (default: 100)
            
        Returns:
            Numpy array of embeddings (float32)
        """
        texts = [chunk['text'] for chunk in chunks]
        total_chunks = len(texts)
        
        logger.info(f"Vectorizing {total_chunks} chunks with {self.model_name}...")
        logger.info(f"Batch size: {batch_size} chunks per API call")
        
        all_embeddings = []
        
        try:
            # Process in batches
            num_batches = (total_chunks + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_chunks)
                batch_texts = texts[start_idx:end_idx]
                
                logger.info(f"Batch {batch_idx + 1}/{num_batches}: Vectorizing chunks {start_idx}-{end_idx}...")
                
                # Call Mistral API for this batch
                response = self.client.embeddings.create(
                    model=self.model_name,
                    inputs=batch_texts
                )
                
                # Extract embeddings from response
                batch_embeddings = np.array([
                    embed.embedding for embed in response.data
                ]).astype('float32')
                
                all_embeddings.append(batch_embeddings)
                
                logger.info(f"  ✓ Batch {batch_idx + 1}: {len(batch_embeddings)} embeddings created")
            
            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"✅ Vectorization complete: shape={embeddings.shape}, dtype={embeddings.dtype}")
            return embeddings
        
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            raise  


    def create_faiss_index(self, embeddings: np.ndarray, n_vectors: int) -> faiss.IndexIVFFlat:
        """
        Créer un index Faiss optimisé.
        """
        dimension = embeddings.shape[1]  # 1024 pour Mistral
        
        # IVF : optimisé pour medium-sized datasets
        nlist = min(100, max(10, len(embeddings) // 100))
        
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        logger.info(f"Training index with {len(embeddings)} vectors...")
        index.train(n_vectors, embeddings)
        index.add(n_vectors,embeddings)
        
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
    
    def save_metadata(self, chunks: List[Dict], events: List[Dict], snapshot_date: str) -> str:
        """
        Sauvegarder les métadonnées des chunks.
        
        Args:
            chunks: List of chunks with chunk_id, event_id, text
            events: List of events to lookup metadata by uid
            snapshot_date: Date for path generation
        """
        from config import Config
        
        metadata_path = Config.get_metadata_path(snapshot_date)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build event lookup by uid
        event_lookup = {event['uid']: event for event in events}
        
        # Format: {index: {event_id, chunk_id, text, title, city, date, url}}
        metadata = {}
        for i, chunk in enumerate(chunks):
            event = event_lookup.get(chunk['event_id'])
            if event:
                metadata[i] = {
                    'event_id': chunk['event_id'],
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'title': event['title_fr'],
                    'city': event['location_city'],
                    'date': event['timings'][0]['begin'] if event['timings'] else None,
                    'url': event['canonical_url']
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
    
    def run_full_vectorization_pipeline(self, processed_path: str, snapshot_date: str, chunk_size: int = 500) -> Tuple[str, str]:
        """Execute complete vectorization pipeline: Load → Chunk → Vectorize → Index → Save"""
        logger.info(f"\n{'='*70}")
        logger.info("VECTORIZATION PIPELINE - START")
        logger.info(f"{'='*70}")
        
        try:
            logger.info("\n[3a] Loading processed events...")
            events = self.load_processed_events(processed_path)
            
            logger.info(f"\n[3b] Chunking events (chunk_size={chunk_size})...")
            chunks = self.chunk_events(events, chunk_size=chunk_size)
            
            logger.info(f"\n[3c] Vectorizing {len(chunks)} chunks...")
            embeddings = self.vectorize_chunks(chunks)
            
            logger.info(f"\n[3d] Creating Faiss index...")
            index = self.create_faiss_index(embeddings, len(chunks))
            
            logger.info(f"\n[3e] Saving Faiss index...")
            index_path = self.save_index(index, snapshot_date)
            
            logger.info(f"\n[3f] Saving metadata...")
            metadata_path = self.save_metadata(chunks, snapshot_date)
            
            logger.info(f"\n{'='*70}")
            logger.info("✅ VECTORIZATION PIPELINE - COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"Index: {index_path}")
            logger.info(f"Metadata: {metadata_path}")
            
            return index_path, metadata_path
        
        except Exception as e:
            logger.error(f"\n❌ VECTORIZATION PIPELINE - FAILED")
            logger.error(f"Error: {e}")
            raise
            print(f"✅ Indexed {len(chunks)} chunks in Faiss")

