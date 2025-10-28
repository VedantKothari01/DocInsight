"""
Index manager for coordinating index building and updates

Manages the lifecycle of similarity search indexes.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .faiss_index import FaissIndex
from .fallback_index import FallbackIndex
from embeddings import Embedder, EmbeddingProcessor
from db import DatabaseManager

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages similarity search indexes"""
    
    def __init__(self, db_manager: DatabaseManager = None, prefer_faiss: bool = True):
        self.db_manager = db_manager or DatabaseManager()
        self.embedder = Embedder()
        self.embedding_processor = EmbeddingProcessor(self.embedder)
        
        # Choose index implementation
        if prefer_faiss:
            # Import HAS_FAISS flag directly from faiss_index
            from .faiss_index import HAS_FAISS
            
            if HAS_FAISS:
                self.index = FaissIndex()
                logger.info("Using FAISS index (faiss-cpu detected).")
            else:
                logger.warning("FAISS not available — using numpy fallback.")
                self.index = FallbackIndex()
        else:
            self.index = FallbackIndex()

            
        self.index_type = type(self.index).__name__
        
    def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build similarity search index from database
        
        Args:
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            Build statistics
        """
        stats = {
            'index_type': self.index_type,
            'build_started': str(datetime.now()),
            'chunks_indexed': 0,
            'embeddings_generated': 0,
            'build_successful': False,
            'errors': []
        }
        
        try:
            # Check if index already exists
            if not force_rebuild and self.index.load_index():
                logger.info("Using existing index. Use force_rebuild=True to rebuild.")
                stats['build_successful'] = True
                stats['chunks_indexed'] = self.index.get_stats().get('num_vectors', 0)
                return stats
                
            # Generate embeddings for any chunks that don't have them
            if self.embedder.is_available():
                embedding_stats = self.embedding_processor.process_unembedded_chunks(self.db_manager)
                stats['embeddings_generated'] = embedding_stats['chunks_processed']
                
                if embedding_stats['chunks_failed'] > 0:
                    stats['errors'].append(f"Failed to generate {embedding_stats['chunks_failed']} embeddings")
            else:
                error_msg = "Embedder not available - cannot generate embeddings"
                stats['errors'].append(error_msg)
                logger.error(error_msg)
                return stats
                
            # Collect all embeddings from database
            embeddings_list = []
            chunk_ids = []
            
            for chunk_id, embedding, text in self.embedding_processor.get_all_embeddings(self.db_manager):
                embeddings_list.append(embedding)
                chunk_ids.append(chunk_id)
                
            if not embeddings_list:
                error_msg = "No embeddings found in database"
                stats['errors'].append(error_msg)
                logger.warning(error_msg)
                return stats
                
            # Build index
            import numpy as np
            embeddings_array = np.vstack(embeddings_list)
            
            success = self.index.build_index(embeddings_array, chunk_ids)
            
            if success:
                stats['build_successful'] = True
                stats['chunks_indexed'] = len(chunk_ids)
                
                # Update database settings
                self.db_manager.set_setting('last_index_build', str(datetime.now()))
                self.db_manager.set_setting('index_version', str(self.get_next_version()))
                
                logger.info(f"Index built successfully: {len(chunk_ids)} chunks indexed")
            else:
                stats['errors'].append("Index building failed")
                
        except Exception as e:
            error_msg = f"Index building failed: {e}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
            
        stats['build_finished'] = str(datetime.now())
        return stats
        
    def update_index_incremental(self, chunk_ids: List[int] = None) -> Dict[str, Any]:
        """
        Incrementally update index with new chunks
        
        Args:
            chunk_ids: Specific chunk IDs to add (if None, adds all unindexed)
            
        Returns:
            Update statistics
        """
        stats = {
            'update_started': str(datetime.now()),
            'chunks_added': 0,
            'update_successful': False,
            'errors': []
        }
        
        try:
            # Load existing index
            if not self.index.load_index():
                error_msg = "No existing index found. Use build_index() first."
                stats['errors'].append(error_msg)
                logger.error(error_msg)
                return stats
                
            # Get embeddings for new chunks
            if chunk_ids is None:
                # Find chunks not in current index
                existing_chunks = set(self.index.chunk_mapping.values())
                conn = self.db_manager.connect()
                cursor = conn.execute("SELECT id FROM chunks WHERE embedding IS NOT NULL")
                all_chunks = {row['id'] for row in cursor.fetchall()}
                chunk_ids = list(all_chunks - existing_chunks)
                
            if not chunk_ids:
                logger.info("No new chunks to add to index")
                stats['update_successful'] = True
                return stats
                
            # Get embeddings for these chunks
            chunk_embeddings = self.embedding_processor.get_embeddings_for_chunks(
                self.db_manager, chunk_ids
            )
            
            if not chunk_embeddings:
                error_msg = "No embeddings found for specified chunks"
                stats['errors'].append(error_msg)
                return stats
                
            # Prepare arrays
            embeddings_list = []
            valid_chunk_ids = []
            
            for chunk_id in chunk_ids:
                if chunk_id in chunk_embeddings:
                    embeddings_list.append(chunk_embeddings[chunk_id])
                    valid_chunk_ids.append(chunk_id)
                    
            if embeddings_list:
                import numpy as np
                embeddings_array = np.vstack(embeddings_list)
                
                success = self.index.add_vectors(embeddings_array, valid_chunk_ids)
                
                if success:
                    stats['update_successful'] = True
                    stats['chunks_added'] = len(valid_chunk_ids)
                    logger.info(f"Index updated: {len(valid_chunk_ids)} chunks added")
                else:
                    stats['errors'].append("Index update failed")
                    
        except Exception as e:
            error_msg = f"Index update failed: {e}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
            
        stats['update_finished'] = str(datetime.now())
        return stats
        
    def search_similar(self, query_text: str, top_k: int = 10, 
                      min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_text: Text to search for
            top_k: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List of similar chunks with metadata
        """
        if not self.embedder.is_available():
            logger.error("Embedder not available")
            return []
            
        try:
            # Load index if not already loaded
            if not self.index.is_available() and not self.index.load_index():
                logger.warning("No search index available")
                return []
                
            # Generate query embedding
            query_embedding = self.embedder.encode_single(query_text)
            if len(query_embedding) == 0:
                return []
                
            # Search index
            search_results = self.index.search(query_embedding, top_k)
            
            # Get chunk details from database
            if not search_results:
                return []
                
            chunk_ids = [chunk_id for chunk_id, score in search_results]
            chunk_details = self._get_chunk_details(chunk_ids)
            
            # Combine results with metadata
            results = []
            for chunk_id, score in search_results:
                if score >= min_score and chunk_id in chunk_details:
                    result = {
                        'chunk_id': chunk_id,
                        'score': score,
                        **chunk_details[chunk_id]
                    }
                    results.append(result)
                    
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
            
    def _get_chunk_details(self, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get chunk details from database"""
        conn = self.db_manager.connect()
        
        placeholders = ','.join('?' * len(chunk_ids))
        cursor = conn.execute(f"""
            SELECT c.id, c.text, c.ordinal, c.char_count,
                   d.title, d.url, d.raw_path,
                   s.type as source_type, s.locator as source_locator
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN sources s ON d.source_id = s.id
            WHERE c.id IN ({placeholders})
        """, chunk_ids)
        
        details = {}
        for row in cursor.fetchall():
            details[row['id']] = {
                'text': row['text'],
                'ordinal': row['ordinal'],
                'char_count': row['char_count'],
                'document_title': row['title'],
                'document_url': row['url'],
                'document_path': row['raw_path'],
                'source_type': row['source_type'],
                'source_locator': row['source_locator']
            }
            
        return details
        
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        stats = self.index.get_stats()
        
        # Add database stats
        db_stats = self.db_manager.get_corpus_stats()
        stats.update({
            'database_chunks': db_stats['total_chunks'],
            'embedded_chunks': db_stats['embedded_chunks'],
            'total_documents': db_stats['total_documents']
        })
        
        # Add coverage info
        if stats.get('num_vectors', 0) > 0 and db_stats['embedded_chunks'] > 0:
            coverage = stats['num_vectors'] / db_stats['embedded_chunks']
            stats['index_coverage'] = min(coverage, 1.0)
        else:
            stats['index_coverage'] = 0.0
            
        return stats
    
    def load_index(self) -> bool:
        """ Load the FAISS or fallback index from disk.Returns True if successfully loaded. """
        try:
            if hasattr(self.index, "load_index"):
                success = self.index.load_index()
                if success:
                    logger.info(f"Index loaded successfully ({type(self.index).__name__})")
                    return True
                else:
                    logger.warning("Index load_index() returned False")
                    return False
            else:
                logger.warning("This index type does not support load_index()")
                return False
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False



        
    def clear_index(self):
        """Clear the search index"""
        self.index.clear()
        logger.info("Search index cleared")
        
    def get_next_version(self) -> int:
        """Get next index version number"""
        current_version = self.db_manager.get_setting('index_version', '0')
        try:
            return int(current_version) + 1
        except (ValueError, TypeError):
            return 1