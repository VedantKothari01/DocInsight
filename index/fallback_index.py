"""
Fallback index implementation using numpy

Provides similarity search when FAISS is not available.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from config import INDEX_PATH

logger = logging.getLogger(__name__)


class FallbackIndex:
    """Numpy-based fallback similarity search index"""
    
    def __init__(self, index_dir: str = INDEX_PATH):
        self.index_dir = Path(index_dir)
        self.embeddings = None
        self.chunk_mapping = {}  # Maps array index to chunk_id
        self.metadata = {}
        
        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.embeddings_file = self.index_dir / 'fallback_embeddings.npy'
        self.mapping_file = self.index_dir / 'fallback_mapping.pkl'
        self.metadata_file = self.index_dir / 'fallback_metadata.json'
        
        logger.warning("Using numpy fallback index - FAISS not available. Performance will be slower.")
        
    def build_index(self, embeddings: np.ndarray, chunk_ids: List[int], 
                   dimension: int = None) -> bool:
        """
        Build a new fallback index from embeddings
        
        Args:
            embeddings: Array of embeddings (N x D)
            chunk_ids: List of chunk IDs corresponding to embeddings
            dimension: Embedding dimension (auto-detected if None)
            
        Returns:
            True if successful
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings provided for index building")
            return False
            
        try:
            # Validate inputs
            if len(embeddings) != len(chunk_ids):
                raise ValueError("Embeddings and chunk_ids must have same length")
                
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / (norms + 1e-8)
            
            # Store embeddings and mapping
            self.embeddings = normalized_embeddings.astype(np.float32)
            self.chunk_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
            
            # Update metadata
            self.metadata = {
                'index_type': 'numpy_fallback',
                'dimension': embeddings.shape[1],
                'num_vectors': len(embeddings),
                'chunk_count': len(chunk_ids),
                'build_timestamp': str(datetime.now())
            }
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Fallback index built successfully: {len(embeddings)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build fallback index: {e}")
            return False
            
    def load_index(self) -> bool:
        """
        Load existing fallback index from disk
        
        Returns:
            True if loaded successfully
        """
        try:
            if not self.embeddings_file.exists():
                logger.info("No existing fallback index found")
                return False
                
            # Load embeddings
            self.embeddings = np.load(self.embeddings_file)
            
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
                
            logger.info(f"Fallback index loaded: {len(self.embeddings)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fallback index: {e}")
            return False
            
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors using cosine similarity
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        if not self.is_available():
            return []
            
        try:
            # Normalize query
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            else:
                return []
                
            # Compute cosine similarities
            similarities = np.dot(self.embeddings, query_embedding.reshape(-1, 1)).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                idx = int(idx)
                if idx in self.chunk_mapping:
                    chunk_id = self.chunk_mapping[idx]
                    score = float(similarities[idx])
                    results.append((chunk_id, score))
                    
            return results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
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
                
            # Normalize new embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / (norms + 1e-8)
            
            # Concatenate with existing embeddings
            current_size = len(self.embeddings)
            self.embeddings = np.vstack([self.embeddings, normalized_embeddings.astype(np.float32)])
            
            # Update chunk mapping
            for i, chunk_id in enumerate(chunk_ids):
                self.chunk_mapping[current_size + i] = chunk_id
                
            # Update metadata
            self.metadata['num_vectors'] = len(self.embeddings)
            self.metadata['chunk_count'] = len(self.chunk_mapping)
            
            # Save updates
            self._save_index()
            
            logger.info(f"Added {len(embeddings)} vectors to fallback index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to fallback index: {e}")
            return False
            
    def remove_chunks(self, chunk_ids: List[int]) -> bool:
        """
        Remove chunks from index
        
        Args:
            chunk_ids: Chunk IDs to remove
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
            
        try:
            # Find indices to remove
            indices_to_remove = []
            for idx, chunk_id in self.chunk_mapping.items():
                if chunk_id in chunk_ids:
                    indices_to_remove.append(idx)
                    
            if not indices_to_remove:
                return True
                
            # Remove from embeddings array
            mask = np.ones(len(self.embeddings), dtype=bool)
            mask[indices_to_remove] = False
            self.embeddings = self.embeddings[mask]
            
            # Rebuild chunk mapping with updated indices
            old_mapping = self.chunk_mapping.copy()
            self.chunk_mapping = {}
            new_idx = 0
            for old_idx in range(len(mask)):
                if mask[old_idx]:  # Keep this embedding
                    if old_idx in old_mapping:
                        self.chunk_mapping[new_idx] = old_mapping[old_idx]
                        new_idx += 1
                        
            # Update metadata
            self.metadata['num_vectors'] = len(self.embeddings)
            self.metadata['chunk_count'] = len(self.chunk_mapping)
            
            # Save updates
            self._save_index()
            
            logger.info(f"Removed {len(indices_to_remove)} vectors from fallback index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove chunks from fallback index: {e}")
            return False
            
    def is_available(self) -> bool:
        """Check if index is loaded and ready"""
        return self.embeddings is not None and len(self.embeddings) > 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            'available': self.is_available(),
            'index_type': 'numpy_fallback',
            'index_dir': str(self.index_dir)
        }
        
        if self.is_available():
            stats.update({
                'num_vectors': len(self.embeddings),
                'dimension': self.embeddings.shape[1],
                'chunk_count': len(self.chunk_mapping)
            })
            
        stats.update(self.metadata)
        return stats
        
    def _save_index(self):
        """Save index, mapping, and metadata to disk"""
        try:
            # Save embeddings
            np.save(self.embeddings_file, self.embeddings)
            
            # Save chunk mapping
            with open(self.mapping_file, 'wb') as f:
                pickle.dump(self.chunk_mapping, f)
                
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.debug("Fallback index saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save fallback index: {e}")
            raise
            
    def clear(self):
        """Clear index and remove files"""
        self.embeddings = None
        self.chunk_mapping = {}
        self.metadata = {}
        
        # Remove files
        for file_path in [self.embeddings_file, self.mapping_file, self.metadata_file]:
            if file_path.exists():
                file_path.unlink()
                
        logger.info("Fallback index cleared")


# Import fix
from datetime import datetime