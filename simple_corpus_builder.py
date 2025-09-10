"""
Simple corpus builder for DocInsight - Works offline with fallback data
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataset_loaders import get_default_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCorpusBuilder:
    """Simple corpus builder that works offline."""
    
    def __init__(self, cache_dir: str = "./corpus_cache"):
        """Initialize simple corpus builder."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.corpus_sentences = []
        
    def build_corpus(self, target_size: int = 50000) -> List[str]:
        """Build corpus using offline fallback data."""
        logger.info(f"Building corpus with target size: {target_size}")
        
        corpus_file = self.cache_dir / "corpus_sentences.json"
        
        # Try to load existing corpus first
        if corpus_file.exists():
            try:
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    self.corpus_sentences = json.load(f)
                logger.info(f"Loaded cached corpus: {len(self.corpus_sentences)} sentences")
                
                if len(self.corpus_sentences) >= target_size * 0.8:
                    return self.corpus_sentences
            except Exception as e:
                logger.warning(f"Error loading cached corpus: {e}")
        
        # Build new corpus
        logger.info("Building new corpus with fallback data...")
        try:
            self.corpus_sentences = get_default_corpus(target_size=target_size)
        except Exception as e:
            logger.warning(f"Error building corpus: {e}")
            # Final fallback - create a minimal corpus
            self.corpus_sentences = self._create_minimal_corpus()
        
        # Save corpus
        try:
            with open(corpus_file, 'w', encoding='utf-8') as f:
                json.dump(self.corpus_sentences, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved corpus: {len(self.corpus_sentences)} sentences")
        except Exception as e:
            logger.warning(f"Error saving corpus: {e}")
        
        return self.corpus_sentences
    
    def _create_minimal_corpus(self) -> List[str]:
        """Create a minimal corpus for basic testing."""
        return [
            "Climate change is a critical global issue that affects agriculture and health.",
            "The effects of global warming include rising sea levels and more extreme weather.",
            "Machine learning improves many real world tasks such as image recognition and language modeling.", 
            "Neural networks can approximate complex functions and are widely used in deep learning.",
            "The French Revolution began in 1789 and led to major political changes in Europe.",
            "Photosynthesis is the process by which green plants convert sunlight into energy.",
            "The mitochondrion is the powerhouse of the cell.",
            "In 1969, Neil Armstrong became the first person to walk on the Moon.",
            "The capital of France is Paris.",
            "SQL stands for Structured Query Language and is used to manage relational databases.",
            "The Internet has revolutionized how people communicate and share information globally.",
            "Renewable energy sources like solar and wind power are becoming increasingly important.",
            "Artificial intelligence systems can process and analyze large amounts of data quickly.",
            "Education plays a vital role in personal development and societal progress.",
            "The human genome contains approximately 3 billion base pairs of DNA.",
            "Democracy allows citizens to participate in decision-making through voting and representation.",
            "Scientific research follows systematic methods to understand natural phenomena and processes.",
            "Economic growth is influenced by factors such as technology, education, and infrastructure.",
            "Cultural diversity enriches societies by bringing together different perspectives and traditions.",
            "Environmental conservation efforts aim to protect ecosystems and preserve biodiversity for future generations.",
        ]
    
    def get_corpus_stats(self) -> Dict:
        """Get basic statistics about the corpus."""
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
        }

def get_simple_corpus(target_size: int = 50000) -> List[str]:
    """Get a simple corpus for offline usage."""
    builder = SimpleCorpusBuilder()
    return builder.build_corpus(target_size)

if __name__ == "__main__":
    # Test the simple corpus builder
    builder = SimpleCorpusBuilder()
    corpus = builder.build_corpus(1000)
    stats = builder.get_corpus_stats()
    
    print("Simple Corpus Builder Test")
    print("=" * 30)
    print(f"Corpus size: {len(corpus)}")
    print(f"Statistics: {stats}")
    if corpus:
        print(f"Sample sentence: {corpus[0]}")
    print("✓ Simple corpus builder working correctly")