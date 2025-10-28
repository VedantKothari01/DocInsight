"""
Corpus Builder - Temporary module for Phase 1

This module provides basic corpus building functionality and will be replaced 
in Phase 2 by a DB-backed ingestion & indexing system.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

from enhanced_pipeline import TextExtractor, SentenceProcessor

logger = logging.getLogger(__name__)


class SimpleCorpusBuilder:
    """
    Simple corpus builder for Phase 1
    
    NOTE: This will be replaced in Phase 2 with a sophisticated 
    DB-backed ingestion & indexing system using SQLite/FAISS hybrid storage.
    """
    
    def __init__(self, corpus_dir: str = "corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.text_extractor = TextExtractor()
        self.sentence_processor = SentenceProcessor()
        
    def build_corpus_from_directory(self, source_dir: str) -> List[str]:
        """
        Build corpus by extracting sentences from all documents in a directory
        
        Args:
            source_dir: Directory containing source documents
            
        Returns:
            List of sentences from all documents
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ValueError(f"Source directory {source_dir} does not exist")
        
        all_sentences = []
        processed_files = 0
        
        # Supported file extensions from config
        from config import SUPPORTED_EXTENSIONS
        
        for file_path in source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    logger.info(f"Processing {file_path}")
                    
                    # Extract text
                    text = self.text_extractor.extract_text(str(file_path))
                    
                    # Split into sentences
                    sentences = self.sentence_processor.split_sentences(text)
                    
                    # Add to corpus with metadata
                    for sentence in sentences:
                        all_sentences.append(sentence)
                    
                    processed_files += 1
                    logger.info(f"Extracted {len(sentences)} sentences from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(f"Corpus building complete. Processed {processed_files} files, extracted {len(all_sentences)} sentences.")
        return all_sentences
    
    def save_corpus(self, sentences: List[str], corpus_name: str = "default") -> str:
        """
        Save corpus sentences to file
        
        Args:
            sentences: List of corpus sentences
            corpus_name: Name for the corpus file
            
        Returns:
            Path to saved corpus file
        """
        os.makedirs(self.corpus_dir, exist_ok=True)
        
        corpus_file = self.corpus_dir / f"{corpus_name}_corpus.json"
        
        corpus_data = {
            "corpus_name": corpus_name,
            "total_sentences": len(sentences),
            "sentences": sentences
        }
        
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Corpus saved to {corpus_file}")
        return str(corpus_file)
    
    def load_corpus(self, corpus_name: str = "default") -> List[str]:
        """
        Load corpus sentences from file
        
        Args:
            corpus_name: Name of the corpus to load
            
        Returns:
            List of corpus sentences
        """
        corpus_file = self.corpus_dir / f"{corpus_name}_corpus.json"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file {corpus_file} not found")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        sentences = corpus_data.get("sentences", [])
        logger.info(f"Loaded corpus with {len(sentences)} sentences from {corpus_file}")
        return sentences
    
    def get_demo_corpus(self) -> List[str]:
        """
        Get the demo corpus used for testing
        
        Returns:
            List of demo corpus sentences
        """
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
            "Python is a high-level programming language used for web development and data science.",
            "The DNA double helix structure was discovered by Watson and Crick in 1953.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "The Internet has revolutionized communication and information sharing globally.",
            "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence."
        ]


def build_demo_corpus() -> List[str]:
    """
    Convenience function to get demo corpus
    
    Returns:
        Demo corpus sentences
    """
    builder = SimpleCorpusBuilder()
    return builder.get_demo_corpus()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Build demo corpus
    builder = SimpleCorpusBuilder()
    demo_sentences = builder.get_demo_corpus()
    
    # Save demo corpus
    corpus_file = builder.save_corpus(demo_sentences, "demo")
    print(f"Demo corpus saved to: {corpus_file}")
    
    # Load and verify
    loaded_sentences = builder.load_corpus("demo")
    print(f"Loaded {len(loaded_sentences)} sentences from corpus")