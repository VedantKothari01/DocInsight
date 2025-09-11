"""
Real dataset corpus builder for DocInsight - Production-ready with online datasets only
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Defensive imports
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Import our real dataset loader
from dataset_loaders import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorpusIndex:
    """Production-ready corpus index using real datasets only - no hardcoded fallbacks."""
    
    def __init__(self, target_size: int = 50000, cache_dir: str = "corpus_cache"):
        self.target_size = target_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.sentences: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[object] = None
        self.model: Optional[object] = None
        self._is_loaded = False
        
        # Initialize sentence transformer model (lazy loading)
        self._init_model()
    
    def _init_model(self):
        """Initialize sentence transformer model with caching."""
        if HAS_SENTENCE_TRANSFORMERS and not self.model:
            try:
                logger.info("Loading sentence transformer model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence transformer model loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None
        elif not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("SentenceTransformers not available - semantic search disabled")
    
    def is_ready_for_production(self) -> bool:
        """Check if corpus is ready for production use (all assets cached)."""
        return (
            self._is_fully_cached() and 
            Path(self.cache_dir / ".docinsight_ready").exists()
        )
    
    def _is_fully_cached(self) -> bool:
        """Check if all necessary files are cached."""
        cache_files = [
            self.cache_dir / f"corpus_{self.target_size}.json",
            self.cache_dir / f"embeddings_{self.target_size}.pkl",
            self.cache_dir / f"faiss_index_{self.target_size}.bin"
        ]
        return all(f.exists() for f in cache_files)
    
    def load_for_production(self) -> bool:
        """Fast loading for production use - assumes all assets are cached."""
        if self._is_loaded:
            return True
            
        logger.info("Loading DocInsight for production use...")
        
        try:
            # Load corpus
            if not self._load_cached_corpus():
                logger.error("No cached corpus found - run setup first")
                return False
            
            # Load embeddings
            if not self._load_cached_embeddings():
                logger.error("No cached embeddings found - run setup first")
                return False
            
            # Load FAISS index
            if not self._load_cached_index():
                logger.warning("No cached FAISS index - will use fallback search")
            
            # Initialize model
            self._init_model()
            
            self._is_loaded = True
            logger.info(f"✅ Production ready: {len(self.sentences)} sentences loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load for production: {e}")
            return False
    def load_or_build(self) -> bool:
        """Load existing corpus or build new one from real datasets."""
        if self._is_loaded:
            return True
        
        # Fast path: try production loading first
        if self.is_ready_for_production():
            logger.info("Found cached assets - loading for production...")
            return self.load_for_production()
        
        # Slow path: build from scratch
        logger.info("No cached assets found - building from real datasets...")
        return self._build_from_scratch()
    
    def _build_from_scratch(self) -> bool:
        """Build corpus from scratch (setup/training mode)."""
        try:
            # Build new corpus using real datasets
            logger.info("Building new corpus from real datasets...")
            dataset_loader = DatasetLoader()
            
            # Load real corpus - NO FALLBACKS
            self.sentences = dataset_loader.load_combined_corpus(self.target_size)
            
            if not self.sentences:
                raise RuntimeError("Failed to load any real datasets - cannot proceed")
            
            logger.info(f"Built corpus with {len(self.sentences)} sentences from real datasets")
            
            # Cache the corpus
            self._save_cached_corpus()
            
            # Build and cache embeddings and index
            self.build_embeddings()
            self.build_index()
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to build corpus: {e}")
            return False
    
    def _load_cached_corpus(self) -> bool:
        """Load corpus from cache if available."""
        cache_file = self.cache_dir / f"corpus_{self.target_size}.json"
        
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.sentences = json.load(f)
            return len(self.sentences) > 0
        except Exception as e:
            logger.warning(f"Failed to load cached corpus: {e}")
            return False
    
    def _save_cached_corpus(self) -> None:
        """Save corpus to cache."""
        if not self.sentences:
            return
            
        cache_file = self.cache_dir / f"corpus_{self.target_size}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.sentences, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached corpus to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache corpus: {e}")
    
    def _load_cached_embeddings(self) -> bool:
        """Load embeddings from cache."""
        embeddings_file = self.cache_dir / f"embeddings_{self.target_size}.pkl"
        
        if not embeddings_file.exists():
            return False
        
        try:
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            logger.info(f"Loaded cached embeddings: {self.embeddings.shape}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return False
    
    def _load_cached_index(self) -> bool:
        """Load FAISS index from cache."""
        if not HAS_FAISS:
            return False
            
        index_file = self.cache_dir / f"faiss_index_{self.target_size}.bin"
        
        if not index_file.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded cached FAISS index: {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            return False
    def build_embeddings(self) -> bool:
        """Build sentence embeddings."""
        if not self.sentences:
            logger.error("No sentences available for embedding")
            return False
        
        if not self.model:
            logger.warning("No sentence transformer model available")
            return False
        
        # Check for cached embeddings first
        if self._load_cached_embeddings():
            return True
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(self.sentences)} sentences...")
        try:
            self.embeddings = self.model.encode(self.sentences, convert_to_numpy=True)
            
            # Normalize embeddings for cosine similarity
            if HAS_FAISS:
                faiss.normalize_L2(self.embeddings)
            
            # Cache embeddings
            embeddings_file = self.cache_dir / f"embeddings_{self.target_size}.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            logger.info(f"Generated and cached embeddings: {self.embeddings.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return False
    
    def build_index(self) -> bool:
        """Build FAISS index for fast similarity search."""
        if not HAS_FAISS:
            logger.warning("FAISS not available - using fallback search")
            return False
        
        if self.embeddings is None:
            if not self.build_embeddings():
                return False
        
        # Check for cached index first
        if self._load_cached_index():
            return True
        
        # Build new index
        try:
            logger.info("Building FAISS index...")
            dimension = self.embeddings.shape[1]
            
            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            # Cache index
            index_file = self.cache_dir / f"faiss_index_{self.target_size}.bin"
            faiss.write_index(self.index, str(index_file))
            
            logger.info(f"Built and cached FAISS index: {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar sentences."""
        if not self.sentences:
            logger.error("No corpus loaded")
            return []
        
        if not self.model:
            logger.warning("No model available for similarity search")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            if HAS_FAISS:
                faiss.normalize_L2(query_embedding)
            
            results = []
            
            if self.index and HAS_FAISS:
                # Use FAISS for fast search
                scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
                
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self.sentences):
                        results.append((self.sentences[idx], float(score)))
            
            else:
                # Fallback: manual similarity computation
                if self.embeddings is None:
                    if not self.build_embeddings():
                        return []
                
                # Compute similarities manually
                similarities = np.dot(self.embeddings, query_embedding.T).flatten()
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    results.append((self.sentences[idx], float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_random_sample(self, n: int = 10) -> List[str]:
        """Get random sample of sentences."""
        if not self.sentences:
            return []
        
        import random
        return random.sample(self.sentences, min(n, len(self.sentences)))
    
    def __len__(self) -> int:
        """Return corpus size."""
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> str:
        """Get sentence by index."""
        return self.sentences[idx]