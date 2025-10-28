"""
FAISS index management for persistent similarity search

Handles building, loading, and querying FAISS indexes.
"""
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import os

logger = logging.getLogger(__name__)

# --- FIXED FAISS IMPORT HANDLING ---
try:
    import faiss
    HAS_FAISS = True
    logger.info("FAISS successfully imported.")
except Exception as e:
    HAS_FAISS = False
    logger.error(f"FAISS import failed: {e}")
# ----------------------------------

from config import INDEX_PATH, INDEX_TYPE, INDEX_DIMENSION


class FaissIndex:
    """Manages FAISS index for similarity search"""
    
    def __init__(self, index_dir: str = INDEX_PATH, index_type: str = INDEX_TYPE):
        self.index_dir = Path(index_dir)
        self.index_type = index_type
        self.index = None
        self.chunk_mapping = {}  # Maps FAISS index position to chunk_id
        self.metadata = {}
        
        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_file = self.index_dir / 'faiss.index'
        self.mapping_file = self.index_dir / 'chunk_mapping.pkl'
        self.metadata_file = self.index_dir / 'metadata.json'
        
    def build_index(self, embeddings: np.ndarray, chunk_ids: List[int], 
                   dimension: int = None) -> bool:
        """
        Build a new FAISS index from embeddings
        
        Args:
            embeddings: Array of embeddings (N x D)
            chunk_ids: List of chunk IDs corresponding to embeddings
            dimension: Embedding dimension (auto-detected if None)
            
        Returns:
            True if successful
        """
        if not HAS_FAISS:
            logger.error("FAISS not available")
            return False
            
        if len(embeddings) == 0:
            logger.warning("No embeddings provided for index building")
            return False
            
        try:
            # Validate inputs
            if len(embeddings) != len(chunk_ids):
                raise ValueError("Embeddings and chunk_ids must have same length")
                
            # Detect dimension
            if dimension is None:
                dimension = embeddings.shape[1]
                
            logger.info(f"Building FAISS index: {len(embeddings)} vectors, dim={dimension}")
            
            # Create index based on type
            if self.index_type == 'IndexFlatIP':
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == 'IndexFlatL2':
                index = faiss.IndexFlatL2(dimension)
            elif self.index_type.startswith('IndexIVF'):
                # For larger datasets, could use IVF indexes
                quantizer = faiss.IndexFlatIP(dimension)
                nlist = min(100, len(embeddings) // 10)  # Reasonable default
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                index.train(embeddings.astype(np.float32))
            else:
                logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
                index = faiss.IndexFlatIP(dimension)
                
            # Add vectors
            embeddings_f32 = embeddings.astype(np.float32)
            index.add(embeddings_f32)
            
            # Store index and mapping
            self.index = index
            self.chunk_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
            
            # Update metadata
            self.metadata = {
                'index_type': self.index_type,
                'dimension': dimension,
                'num_vectors': len(embeddings),
                'chunk_count': len(chunk_ids),
                'build_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now())
            }
            
            # Save to disk
            self._save_index()
            
            logger.info(f"FAISS index built successfully: {index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False
            
    def load_index(self) -> bool:
        """
        Load existing FAISS index from disk
        
        Returns:
            True if loaded successfully
        """
        if not HAS_FAISS:
            logger.error("FAISS not available")
            return False
            
        try:
            if not self.index_file.exists():
                logger.info("No existing FAISS index found")
                return False
                
            # Load index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load chunk mapping
            if self.mapping_file.exists():
                with open(self.mapping_file, 'rb') as f:
                    self.chunk_mapping = pickle.load(f)
            else:
                logger.warning("Chunk mapping file not found")
                self.chunk_mapping = {}
                
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
                
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
            
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        if not self.is_available():
            self.load_index()
            if not self.is_available():
                logger.error("FAISS index not loaded; cannot perform search")
                return []

            
        try:
            # Ensure query is 2D and float32
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype(np.float32)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self.chunk_mapping:
                    chunk_id = self.chunk_mapping[idx]
                    results.append((chunk_id, float(score)))
                    
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
            
    def add_vectors(self, embeddings: np.ndarray, chunk_ids: List[int]) -> bool:
        """
        Add new vectors to existing index (incremental update)
        
        Args:
            embeddings: New embeddings to add
            chunk_ids: Corresponding chunk IDs
            
        Returns:
            True if successful
        """
        if not self.is_available():
            logger.warning("Index not available for incremental update")
            return False
            
        try:
            if len(embeddings) != len(chunk_ids):
                raise ValueError("Embeddings and chunk_ids must have same length")
                
            # Add to index
            embeddings_f32 = embeddings.astype(np.float32)
            current_size = self.index.ntotal
            self.index.add(embeddings_f32)
            
            # Update chunk mapping
            for i, chunk_id in enumerate(chunk_ids):
                self.chunk_mapping[current_size + i] = chunk_id
                
            # Update metadata
            self.metadata['num_vectors'] = self.index.ntotal
            self.metadata['chunk_count'] = len(self.chunk_mapping)
            
            # Save updates
            self._save_index()
            
            logger.info(f"Added {len(embeddings)} vectors to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to index: {e}")
            return False
            
    def remove_chunks(self, chunk_ids: List[int]) -> bool:
        """
        Remove chunks from index (requires rebuild)
        
        Args:
            chunk_ids: Chunk IDs to remove
            
        Returns:
            True if successful
        """
        # FAISS doesn't support efficient deletion, so we need to rebuild
        logger.warning("Chunk removal requires index rebuild")
        
        # Mark for rebuild - could implement lazy deletion here
        removed_indices = []
        for idx, chunk_id in self.chunk_mapping.items():
            if chunk_id in chunk_ids:
                removed_indices.append(idx)
                
        if removed_indices:
            logger.info(f"Marked {len(removed_indices)} vectors for removal. Index rebuild recommended.")
            # Could set a flag here to trigger rebuild on next search
            
        return True
        
    def is_available(self) -> bool:
        """Check if FAISS is available and the in-memory index object is loaded"""
        return HAS_FAISS and self.index is not None

            
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            'available': self.is_available(),
            'has_faiss': HAS_FAISS,
            'index_type': self.index_type,
            'index_dir': str(self.index_dir)
        }
        
        if self.is_available() and self.index is not None:
            stats.update({
                'num_vectors': self.index.ntotal,
                'dimension': self.index.d,
                'chunk_count': len(self.chunk_mapping)
            })
            
        stats.update(self.metadata)
        return stats
        
    def _save_index(self):
        """Save index, mapping, and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save chunk mapping
            with open(self.mapping_file, 'wb') as f:
                pickle.dump(self.chunk_mapping, f)
                
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.debug("FAISS index saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
            
    def clear(self):
        """Clear index and remove files"""
        self.index = None
        self.chunk_mapping = {}
        self.metadata = {}
        
        # Remove files
        for file_path in [self.index_file, self.mapping_file, self.metadata_file]:
            if file_path.exists():
                file_path.unlink()
                
        logger.info("FAISS index cleared")


# Fix import issue
try:
    import pandas as pd
except ImportError:
    from datetime import datetime