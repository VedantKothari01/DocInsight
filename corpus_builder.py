"""
Clean corpus builder for DocInsight - Defensive implementation with fallbacks
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback mini corpus for offline operation
FALLBACK_CORPUS = [
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks can approximate complex non-linear functions.",
    "Deep learning requires large amounts of training data.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information.",
    "Artificial intelligence aims to create intelligent machines.",
    "Data science combines statistics, programming, and domain expertise.",
    "Algorithm optimization improves computational efficiency and performance.",
    "Cloud computing provides scalable infrastructure for applications.",
    "Cybersecurity protects digital systems from malicious attacks.",
    "Climate change affects global weather patterns significantly.",
    "Renewable energy sources help reduce carbon emissions.",
    "Sustainable development balances economic growth with environmental protection.",
    "Biodiversity loss threatens ecosystem stability worldwide.",
    "Ocean acidification impacts marine life and food chains.",
    "Education technology transforms traditional learning methods.",
    "Digital literacy skills are essential in modern society.",
    "Remote work changes organizational culture and practices.",
    "Social media influences public opinion and behavior.",
    "Healthcare innovation improves patient outcomes and treatment options.",
]

class CorpusIndex:
    """Clean corpus management with fallback capabilities."""
    
    def __init__(self, 
                 target_size: int = 5000,
                 cache_dir: str = "./cache",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize corpus index.
        
        Args:
            target_size: Target number of sentences in corpus
            cache_dir: Directory for caching (not committed to git)
            model_name: SentenceTransformer model name
        """
        self.target_size = target_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        
        self.sentences: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._model = None
        
    def _load_model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading SentenceTransformer: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load model {self.model_name}: {e}")
                self._model = None
        return self._model
    
    def _build_fallback_corpus(self) -> List[str]:
        """Build corpus using fallback data."""
        logger.info("Using fallback mini-corpus (offline mode)")
        sentences = FALLBACK_CORPUS.copy()
        
        # Expand with variations if needed
        while len(sentences) < min(self.target_size, 100):
            for base in FALLBACK_CORPUS:
                if len(sentences) >= min(self.target_size, 100):
                    break
                # Add simple variations
                variations = [
                    f"The concept of {base.lower()}",
                    f"Research shows that {base.lower()}",
                    f"Studies indicate that {base.lower()}",
                ]
                sentences.extend(variations[:1])  # Add one variation
        
        return sentences[:self.target_size]
    
    def _try_load_datasets(self) -> List[str]:
        """Attempt to load from dataset_loaders, fallback gracefully."""
        try:
            from dataset_loaders import get_default_corpus
            logger.info("Attempting to load real datasets...")
            sentences = get_default_corpus(target_size=self.target_size)
            if sentences and len(sentences) > 50:
                logger.info(f"Loaded {len(sentences)} sentences from real datasets")
                return sentences
        except Exception as e:
            logger.warning(f"Failed to load real datasets: {e}")
        
        return self._build_fallback_corpus()
    
    def load_or_build(self) -> 'CorpusIndex':
        """Load existing corpus or build new one."""
        cache_file = self.cache_dir / f"corpus_{self.target_size}.json"
        
        # Try to load cached corpus
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.sentences = json.load(f)
                logger.info(f"Loaded {len(self.sentences)} sentences from cache")
                return self
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Build new corpus
        self.sentences = self._try_load_datasets()
        
        # Cache the corpus
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.sentences, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(self.sentences)} sentences")
        except Exception as e:
            logger.warning(f"Failed to cache corpus: {e}")
        
        return self
    
    def build_index(self) -> bool:
        """Build FAISS index if possible."""
        if not HAS_FAISS or not self.sentences:
            logger.warning("Cannot build FAISS index (missing dependencies or sentences)")
            return False
        
        model = self._load_model()
        if not model:
            logger.warning("Cannot build index without SentenceTransformer model")
            return False
        
        try:
            logger.info("Building embeddings and FAISS index...")
            # Generate embeddings
            self.embeddings = model.encode(self.sentences, convert_to_numpy=True)
            
            # Normalize embeddings
            faiss.normalize_L2(self.embeddings)
            
            # Build FAISS index
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
            self.index.add(self.embeddings)
            
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar sentences."""
        if not self.sentences:
            return []
        
        # If no FAISS index, use simple string matching as fallback
        if not self.index:
            results = []
            query_lower = query.lower()
            for sentence in self.sentences[:k*2]:  # Search in first k*2 sentences
                if any(word in sentence.lower() for word in query_lower.split()):
                    # Simple word overlap scoring
                    score = len(set(query_lower.split()) & set(sentence.lower().split())) / max(len(query_lower.split()), 1)
                    results.append((sentence, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
        
        model = self._load_model()
        if not model:
            return []
        
        try:
            # Encode query
            query_embedding = model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.sentences):
                    results.append((self.sentences[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []