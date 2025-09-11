"""
Embedder for generating text embeddings in batches

Uses SentenceTransformer models for generating embeddings.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Iterator
import pickle

# Optional import for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from config import EMBEDDING_MODEL, INGEST_BATCH_SIZE

logger = logging.getLogger(__name__)


class Embedder:
    """Handles batched embedding generation"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.error("sentence-transformers not available. Embeddings will not work.")
            return
            
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[1]
            
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            
    def is_available(self) -> bool:
        """Check if embedder is available"""
        return self.model is not None
        
    def encode(self, texts: List[str], batch_size: int = INGEST_BATCH_SIZE) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if not self.is_available():
            raise RuntimeError("Embedding model not available")
            
        if not texts:
            return np.array([])
            
        try:
            # Process in batches
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 100
                )
                embeddings.append(batch_embeddings)
                
            # Concatenate all batches
            all_embeddings = np.vstack(embeddings)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            all_embeddings = all_embeddings / (norms + 1e-8)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
            
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        embeddings = self.encode([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
        
    def get_dimension(self) -> Optional[int]:
        """Get embedding dimension"""
        return self.embedding_dim
        
    def serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding for database storage"""
        return pickle.dumps(embedding.astype(np.float32))
        
    def deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Deserialize embedding from database"""
        return pickle.loads(data)


class EmbeddingProcessor:
    """Processes chunks and generates embeddings"""
    
    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder or Embedder()
        
    def process_unembedded_chunks(self, db_manager, batch_size: int = INGEST_BATCH_SIZE) -> Dict[str, int]:
        """
        Process all chunks that don't have embeddings yet
        
        Args:
            db_manager: Database manager instance
            batch_size: Batch size for processing
            
        Returns:
            Statistics about processing
        """
        if not self.embedder.is_available():
            raise RuntimeError("Embedder not available")
            
        stats = {
            'chunks_processed': 0,
            'chunks_failed': 0,
            'batches_processed': 0
        }
        
        try:
            # Get unembedded chunks in batches
            conn = db_manager.connect()
            
            while True:
                # Fetch batch of unembedded chunks
                cursor = conn.execute("""
                    SELECT id, text FROM chunks 
                    WHERE embedding IS NULL 
                    ORDER BY id 
                    LIMIT ?
                """, (batch_size,))
                
                batch = cursor.fetchall()
                if not batch:
                    break
                    
                # Extract texts and IDs
                chunk_ids = [row['id'] for row in batch]
                texts = [row['text'] for row in batch]
                
                try:
                    # Generate embeddings
                    embeddings = self.embedder.encode(texts)
                    
                    # Store embeddings
                    with db_manager.transaction() as trans_conn:
                        for chunk_id, embedding in zip(chunk_ids, embeddings):
                            serialized = self.embedder.serialize_embedding(embedding)
                            trans_conn.execute("""
                                UPDATE chunks SET embedding = ? WHERE id = ?
                            """, (serialized, chunk_id))
                            
                    stats['chunks_processed'] += len(chunk_ids)
                    stats['batches_processed'] += 1
                    
                    logger.info(f"Processed batch of {len(chunk_ids)} embeddings")
                    
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    stats['chunks_failed'] += len(chunk_ids)
                    continue
                    
        except Exception as e:
            logger.error(f"Embedding processing failed: {e}")
            raise
            
        return stats
        
    def get_embeddings_for_chunks(self, db_manager, chunk_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Get embeddings for specific chunks
        
        Args:
            db_manager: Database manager instance
            chunk_ids: List of chunk IDs
            
        Returns:
            Dictionary mapping chunk ID to embedding
        """
        conn = db_manager.connect()
        
        # Build query with proper parameterization
        placeholders = ','.join('?' * len(chunk_ids))
        cursor = conn.execute(f"""
            SELECT id, embedding FROM chunks 
            WHERE id IN ({placeholders}) AND embedding IS NOT NULL
        """, chunk_ids)
        
        embeddings = {}
        for row in cursor.fetchall():
            chunk_id = row['id']
            embedding_data = row['embedding']
            if embedding_data:
                embedding = self.embedder.deserialize_embedding(embedding_data)
                embeddings[chunk_id] = embedding
                
        return embeddings
        
    def get_all_embeddings(self, db_manager) -> Iterator[tuple[int, np.ndarray, str]]:
        """
        Get all embeddings from database
        
        Args:
            db_manager: Database manager instance
            
        Yields:
            Tuples of (chunk_id, embedding, text)
        """
        conn = db_manager.connect()
        cursor = conn.execute("""
            SELECT id, embedding, text FROM chunks 
            WHERE embedding IS NOT NULL 
            ORDER BY id
        """)
        
        for row in cursor:
            chunk_id = row['id']
            embedding_data = row['embedding']
            text = row['text']
            
            if embedding_data:
                embedding = self.embedder.deserialize_embedding(embedding_data)
                yield chunk_id, embedding, text