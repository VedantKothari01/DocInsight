"""
Retrieval API for DocInsight Phase 2

Provides unified interface for similarity search and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from index import IndexManager
from db import DatabaseManager
from config import RETRIEVAL_TOP_K, MIN_SIM_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    """Represents a ranked similarity search result"""
    chunk_id: int
    text: str
    score: float
    ordinal: int
    document_title: str
    document_url: Optional[str] = None
    document_path: Optional[str] = None
    source_type: str = 'unknown'
    source_locator: str = ''
    char_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'score': self.score,
            'ordinal': self.ordinal,
            'document_title': self.document_title,
            'document_url': self.document_url,
            'document_path': self.document_path,
            'source_type': self.source_type,
            'source_locator': self.source_locator,
            'char_count': self.char_count
        }


class RetrievalEngine:
    """Main retrieval engine for similarity search"""
    
    def __init__(self, db_manager: DatabaseManager = None, index_manager: IndexManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.index_manager = index_manager or IndexManager(self.db_manager)
        
    def retrieve_similar_chunks(self, query_chunks: List[str], 
                              top_k: int = RETRIEVAL_TOP_K,
                              min_score: float = MIN_SIM_THRESHOLD) -> List[RankedChunk]:
        """
        Retrieve similar chunks for a list of query chunks
        
        Args:
            query_chunks: List of text chunks to search for
            top_k: Maximum results per query chunk
            min_score: Minimum similarity score threshold
            
        Returns:
            List of ranked similar chunks
        """
        if not query_chunks:
            return []
            
        all_results = []
        
        for query_chunk in query_chunks:
            if not query_chunk.strip():
                continue
                
            try:
                # Search for similar chunks
                search_results = self.index_manager.search_similar(
                    query_chunk, 
                    top_k=top_k, 
                    min_score=min_score
                )
                
                # Convert to RankedChunk objects
                for result in search_results:
                    ranked_chunk = RankedChunk(
                        chunk_id=result['chunk_id'],
                        text=result['text'],
                        score=result['score'],
                        ordinal=result['ordinal'],
                        document_title=result['document_title'],
                        document_url=result.get('document_url'),
                        document_path=result.get('document_path'),
                        source_type=result.get('source_type', 'unknown'),
                        source_locator=result.get('source_locator', ''),
                        char_count=result.get('char_count', 0)
                    )
                    all_results.append(ranked_chunk)
                    
            except Exception as e:
                logger.error(f"Search failed for query chunk: {e}")
                continue
                
        # Remove duplicates and sort by score
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results
        
    def retrieve_for_document(self, document_text: str, 
                             chunk_size: int = 500,
                             top_k: int = RETRIEVAL_TOP_K,
                             min_score: float = MIN_SIM_THRESHOLD) -> List[RankedChunk]:
        """
        Retrieve similar chunks for an entire document
        
        Args:
            document_text: Full document text
            chunk_size: Size for splitting document into chunks
            top_k: Maximum results per chunk
            min_score: Minimum similarity score
            
        Returns:
            List of ranked similar chunks
        """
        # Split document into chunks
        chunks = self._split_document(document_text, chunk_size)
        
        # Retrieve for all chunks
        return self.retrieve_similar_chunks(chunks, top_k, min_score)
        
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus and index statistics"""
        stats = {}
        
        # Database stats
        db_stats = self.db_manager.get_corpus_stats()
        stats.update(db_stats)
        
        # Index stats
        index_stats = self.index_manager.get_index_stats()
        stats.update(index_stats)
        
        return stats
        
    def is_ready(self) -> bool:
        """Check if retrieval system is ready for queries"""
        try:
            # Try to load index if not already loaded
            if not self.index_manager.index.is_available():
                self.index_manager.index.load_index()
                
            index_stats = self.index_manager.get_index_stats()
            return (
                index_stats.get('available', False) and 
                index_stats.get('num_vectors', 0) > 0
            )
        except Exception:
            return False
            
    def _deduplicate_results(self, results: List[RankedChunk]) -> List[RankedChunk]:
        """Remove duplicate chunks, keeping highest scoring"""
        seen_chunks = {}
        
        for result in results:
            chunk_id = result.chunk_id
            if chunk_id not in seen_chunks or result.score > seen_chunks[chunk_id].score:
                seen_chunks[chunk_id] = result
                
        return list(seen_chunks.values())
        
    def _split_document(self, text: str, chunk_size: int) -> List[str]:
        """Split document into chunks for retrieval"""
        chunks = []
        
        # Simple sentence-based splitting
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to paragraph splitting
            sentences = [s.strip() for s in text.split('\n\n') if s.strip()]
            
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 50:  # Minimum chunk size
                    chunks.append(chunk_text)
                    
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 50:
                chunks.append(chunk_text)
                
        return chunks


class HybridRetrieval:
    """Hybrid retrieval with multiple scoring methods"""
    
    def __init__(self, retrieval_engine: RetrievalEngine = None):
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        
        # Optional cross-encoder for reranking
        self.cross_encoder = None
        self._load_cross_encoder()
        
    def _load_cross_encoder(self):
        """Load cross-encoder for reranking (optional)"""
        try:
            from sentence_transformers import CrossEncoder
            from config import CROSS_ENCODER_MODEL_NAME
            
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            logger.info("Cross-encoder loaded for reranking")
            
        except Exception as e:
            logger.debug(f"Cross-encoder not available: {e}")
            self.cross_encoder = None
            
    def retrieve_with_reranking(self, query_chunks: List[str],
                               top_k: int = RETRIEVAL_TOP_K,
                               rerank_top_k: Optional[int] = None) -> List[RankedChunk]:
        """
        Retrieve with optional cross-encoder reranking
        
        Args:
            query_chunks: Query text chunks
            top_k: Initial retrieval count
            rerank_top_k: Final count after reranking (defaults to top_k)
            
        Returns:
            Reranked results
        """
        if rerank_top_k is None:
            rerank_top_k = top_k
            
        # Initial retrieval with higher top_k for reranking
        initial_k = top_k * 2 if self.cross_encoder else top_k
        
        results = self.retrieval_engine.retrieve_similar_chunks(
            query_chunks, 
            top_k=initial_k
        )
        
        if not results or not self.cross_encoder:
            return results[:rerank_top_k]
            
        # Rerank with cross-encoder
        try:
            reranked_results = self._rerank_results(query_chunks, results)
            return reranked_results[:rerank_top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:rerank_top_k]
            
    def _rerank_results(self, query_chunks: List[str], 
                       results: List[RankedChunk]) -> List[RankedChunk]:
        """Rerank results using cross-encoder"""
        if not query_chunks or not results:
            return results
            
        # Create pairs for cross-encoder scoring
        pairs = []
        result_indices = []
        
        for query_chunk in query_chunks:
            for i, result in enumerate(results):
                pairs.append([query_chunk, result.text])
                result_indices.append(i)
                
        # Score pairs
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Aggregate scores for each result
        result_scores = {}
        for idx, score in zip(result_indices, cross_scores):
            if idx not in result_scores:
                result_scores[idx] = []
            result_scores[idx].append(float(score))
            
        # Create new results with fused scores
        reranked = []
        for i, result in enumerate(results):
            if i in result_scores:
                # Combine semantic and cross-encoder scores
                semantic_score = result.score
                cross_score = max(result_scores[i])  # Use best cross-encoder score
                
                # Weighted fusion (configurable weights)
                from config import FUSION_WEIGHTS
                fused_score = (
                    semantic_score * FUSION_WEIGHTS.get('semantic', 0.7) +
                    cross_score * FUSION_WEIGHTS.get('cross_encoder', 0.3)
                )
                
                # Create new result with fused score
                new_result = RankedChunk(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=fused_score,
                    ordinal=result.ordinal,
                    document_title=result.document_title,
                    document_url=result.document_url,
                    document_path=result.document_path,
                    source_type=result.source_type,
                    source_locator=result.source_locator,
                    char_count=result.char_count
                )
                reranked.append(new_result)
            else:
                reranked.append(result)
                
        # Sort by fused score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


# Global retrieval engine instance
_retrieval_engine = None

def get_retrieval_engine() -> RetrievalEngine:
    """Get global retrieval engine instance"""
    global _retrieval_engine
    if _retrieval_engine is None:
        _retrieval_engine = RetrievalEngine()
    return _retrieval_engine