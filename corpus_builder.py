"""
Corpus builder for DocInsight - Manages corpus building, indexing, and storage
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import logging
from dataset_loaders import get_default_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorpusBuilder:
    """Manages corpus building, embedding, indexing, and storage."""
    
    def __init__(self, 
                 cache_dir: str = "./corpus_cache",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize corpus builder.
        
        Args:
            cache_dir: Directory to store cached embeddings and indices
            model_name: SentenceTransformer model to use for embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model_name = model_name
        self.sbert_model = None
        self.corpus_sentences = []
        self.embeddings = None
        self.index = None
        
    def load_model(self):
        """Load the SentenceTransformer model."""
        if self.sbert_model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.sbert_model = SentenceTransformer(self.model_name)
        return self.sbert_model
    
    def build_corpus(self, 
                    sentences: Optional[List[str]] = None,
                    target_size: int = 50000,
                    force_rebuild: bool = False) -> List[str]:
        """
        Build or load corpus.
        
        Args:
            sentences: Optional list of sentences to use
            target_size: Target number of sentences
            force_rebuild: Force rebuilding even if cached version exists
            
        Returns:
            List of corpus sentences
        """
        corpus_file = self.cache_dir / "corpus_sentences.json"
        
        # Try to load existing corpus if not forcing rebuild
        if not force_rebuild and corpus_file.exists():
            try:
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    self.corpus_sentences = json.load(f)
                logger.info(f"Loaded cached corpus: {len(self.corpus_sentences)} sentences")
                
                # Check if we need more sentences
                if len(self.corpus_sentences) >= target_size * 0.8:
                    return self.corpus_sentences
            except Exception as e:
                logger.warning(f"Error loading cached corpus: {e}")
        
        # Build new corpus
        if sentences is None:
            logger.info("Downloading datasets to build corpus...")
            sentences = get_default_corpus(target_size=target_size)
        
        # Deduplicate and filter
        self.corpus_sentences = self._process_sentences(sentences)
        
        # Save corpus
        try:
            with open(corpus_file, 'w', encoding='utf-8') as f:
                json.dump(self.corpus_sentences, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached corpus saved: {len(self.corpus_sentences)} sentences")
        except Exception as e:
            logger.warning(f"Error saving corpus: {e}")
        
        return self.corpus_sentences
    
    def _process_sentences(self, sentences: List[str]) -> List[str]:
        """Process and clean sentences."""
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if sent and sent not in seen:
                # Basic quality filters
                if (10 <= len(sent) <= 300 and
                    len(sent.split()) >= 3 and
                    any(c.isalpha() for c in sent) and
                    not sent.startswith(('http', 'www', '[', '{', '<'))):
                    unique_sentences.append(sent)
                    seen.add(sent)
        
        logger.info(f"Processed sentences: {len(sentences)} -> {len(unique_sentences)}")
        return unique_sentences
    
    def build_embeddings(self, batch_size: int = 64) -> np.ndarray:
        """
        Build embeddings for the corpus.
        
        Args:
            batch_size: Batch size for encoding
            
        Returns:
            Normalized embeddings array
        """
        if not self.corpus_sentences:
            raise ValueError("No corpus sentences available. Call build_corpus() first.")
        
        embeddings_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        # Try to load cached embeddings
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cached embeddings match current corpus
                if (len(cached_data['sentences']) == len(self.corpus_sentences) and
                    cached_data['sentences'][:100] == self.corpus_sentences[:100]):
                    self.embeddings = cached_data['embeddings']
                    logger.info(f"Loaded cached embeddings: {self.embeddings.shape}")
                    return self.embeddings
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {e}")
        
        # Build new embeddings
        logger.info(f"Building embeddings for {len(self.corpus_sentences)} sentences...")
        model = self.load_model()
        
        self.embeddings = model.encode(
            self.corpus_sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # Cache embeddings
        try:
            with open(embeddings_file, 'wb') as f:
                pickle.dump({
                    'sentences': self.corpus_sentences,
                    'embeddings': self.embeddings,
                    'model_name': self.model_name
                }, f)
            logger.info(f"Cached embeddings saved: {self.embeddings.shape}")
        except Exception as e:
            logger.warning(f"Error saving embeddings: {e}")
        
        return self.embeddings
    
    def build_index(self, index_type: str = "flat") -> faiss.Index:
        """
        Build FAISS index for similarity search.
        
        Args:
            index_type: Type of index ("flat", "ivf", "hnsw")
            
        Returns:
            FAISS index
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call build_embeddings() first.")
        
        index_file = self.cache_dir / f"faiss_index_{index_type}_{self.model_name.replace('/', '_')}.index"
        
        # Try to load cached index
        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                if self.index.ntotal == len(self.corpus_sentences):
                    logger.info(f"Loaded cached FAISS index: {self.index.ntotal} vectors")
                    return self.index
            except Exception as e:
                logger.warning(f"Error loading cached index: {e}")
        
        # Build new index
        logger.info(f"Building {index_type} FAISS index...")
        d = self.embeddings.shape[1]
        
        if index_type == "flat":
            # Exact search using inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "ivf":
            # Approximate search using IVF (Inverted File)
            nlist = min(4096, max(64, len(self.corpus_sentences) // 39))
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.index.train(self.embeddings)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World graphs
            M = 16  # Number of bi-directional links for every node
            self.index = faiss.IndexHNSWFlat(d, M)
            self.index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors to index
        self.index.add(self.embeddings)
        
        # Cache index
        try:
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Cached FAISS index saved: {self.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Error saving index: {e}")
        
        return self.index
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar sentences in the corpus.
        
        Args:
            query: Query sentence
            top_k: Number of results to return
            
        Returns:
            List of search results with sentences and scores
        """
        if self.index is None:
            raise ValueError("No index available. Call build_index() first.")
        
        if self.sbert_model is None:
            self.load_model()
        
        # Encode query
        query_embedding = self.sbert_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append({
                    'sentence': self.corpus_sentences[idx],
                    'score': float(score)
                })
        
        return results
    
    def get_corpus_stats(self) -> Dict:
        """Get statistics about the corpus."""
        if not self.corpus_sentences:
            return {}
        
        sentence_lengths = [len(sent) for sent in self.corpus_sentences]
        word_counts = [len(sent.split()) for sent in self.corpus_sentences]
        
        return {
            'total_sentences': len(self.corpus_sentences),
            'avg_sentence_length': np.mean(sentence_lengths),
            'avg_word_count': np.mean(word_counts),
            'min_sentence_length': min(sentence_lengths),
            'max_sentence_length': max(sentence_lengths),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
            'index_type': type(self.index).__name__ if self.index is not None else None,
            'index_total': self.index.ntotal if self.index is not None else None
        }
    
    def build_complete_corpus(self,
                            target_size: int = 50000,
                            index_type: str = "flat",
                            force_rebuild: bool = False) -> Tuple[List[str], faiss.Index]:
        """
        Complete corpus building pipeline.
        
        Args:
            target_size: Target number of sentences
            index_type: Type of FAISS index to build
            force_rebuild: Force rebuilding even if cached versions exist
            
        Returns:
            Tuple of (corpus_sentences, faiss_index)
        """
        logger.info("Starting complete corpus building pipeline...")
        
        # Build corpus
        self.build_corpus(target_size=target_size, force_rebuild=force_rebuild)
        
        # Build embeddings
        self.build_embeddings()
        
        # Build index
        self.build_index(index_type=index_type)
        
        # Print stats
        stats = self.get_corpus_stats()
        logger.info(f"Corpus building complete: {stats}")
        
        return self.corpus_sentences, self.index

def get_default_corpus_and_index(target_size: int = 50000, 
                                index_type: str = "flat",
                                force_rebuild: bool = False) -> Tuple[List[str], faiss.Index, SentenceTransformer]:
    """
    Convenience function to get a complete corpus, index, and model.
    
    Args:
        target_size: Target number of sentences in corpus
        index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        force_rebuild: Force rebuilding even if cached versions exist
        
    Returns:
        Tuple of (corpus_sentences, faiss_index, sbert_model)
    """
    builder = CorpusBuilder()
    corpus_sentences, index = builder.build_complete_corpus(
        target_size=target_size,
        index_type=index_type,
        force_rebuild=force_rebuild
    )
    
    return corpus_sentences, index, builder.sbert_model