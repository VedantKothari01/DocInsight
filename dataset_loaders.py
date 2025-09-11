"""
Dataset loaders for DocInsight - Download and process real datasets
Real dataset integration: PAWS, Wikipedia, arXiv, academic sources
"""
import os
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Defensive imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logging.warning("requests not available - some dataset downloads may fail")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logging.warning("datasets library not available - will try alternative downloads")

try:
    import wikipedia
    HAS_WIKIPEDIA = True
except ImportError:
    HAS_WIKIPEDIA = False
    logging.warning("wikipedia library not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads real datasets from various sources for plagiarism detection."""
    
    def __init__(self, cache_dir: str = "dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_paws_dataset(self, split: str = "train", max_samples: int = 10000) -> List[str]:
        """Load PAWS (Paraphrase Adversaries from Word Scrambling) dataset."""
        logger.info(f"Loading PAWS dataset (split: {split}, max_samples: {max_samples})")
        
        cache_file = self.cache_dir / f"paws_{split}_{max_samples}.json"
        if cache_file.exists():
            logger.info("Loading PAWS from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        sentences = []
        
        if HAS_DATASETS:
            try:
                logger.info("Downloading PAWS dataset...")
                dataset = load_dataset("paws", "labeled_final", split=split)
                
                for i, example in enumerate(dataset):
                    if i >= max_samples:
                        break
                    
                    # Add both sentences from paraphrase pairs
                    if example['sentence1'] and len(example['sentence1'].strip()) > 10:
                        sentences.append(example['sentence1'].strip())
                    if example['sentence2'] and len(example['sentence2'].strip()) > 10:
                        sentences.append(example['sentence2'].strip())
                
                # Remove duplicates and very short sentences
                sentences = list(set([s for s in sentences if len(s.split()) >= 5]))
                logger.info(f"Loaded {len(sentences)} unique sentences from PAWS")
                
                # Cache the results
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(sentences, f, indent=2, ensure_ascii=False)
                
                return sentences
                
            except Exception as e:
                logger.error(f"Failed to load PAWS dataset: {e}")
        
        return []
    
    def load_wikipedia_articles(self, topics: List[str] = None, sentences_per_topic: int = 100) -> List[str]:
        """Load sentences from Wikipedia articles on various topics."""
        if topics is None:
            topics = [
                "Machine learning", "Artificial intelligence", "Natural language processing",
                "Computer vision", "Deep learning", "Neural network", "Data science",
                "Climate change", "Renewable energy", "Biodiversity", "Sustainability",
                "Scientific method", "Research methodology", "Academic writing",
                "Literature", "Philosophy", "Psychology", "Economics", "Education",
                "Healthcare", "Medicine", "Technology", "Innovation", "Mathematics"
            ]
        
        logger.info(f"Loading Wikipedia articles for {len(topics)} topics")
        
        cache_file = self.cache_dir / f"wikipedia_{len(topics)}_{sentences_per_topic}.json"
        if cache_file.exists():
            logger.info("Loading Wikipedia content from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        sentences = []
        
        if HAS_WIKIPEDIA:
            import wikipedia
            wikipedia.set_lang("en")  # Set language to English
            
            for topic in topics:
                try:
                    logger.info(f"Fetching Wikipedia article: {topic}")
                    # Try to get the page with auto-suggest to handle typos
                    page = wikipedia.page(topic, auto_suggest=True)
                    content = page.content
                    
                    # Split into sentences and clean
                    topic_sentences = self._extract_sentences(content)
                    
                    # Filter and limit sentences
                    filtered_sentences = [
                        s for s in topic_sentences 
                        if len(s.split()) >= 5 and len(s) <= 500
                    ][:sentences_per_topic]
                    
                    sentences.extend(filtered_sentences)
                    logger.info(f"Added {len(filtered_sentences)} sentences from {topic}")
                    
                    # Small delay to be respectful to Wikipedia
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to load Wikipedia article for {topic}: {e}")
                    continue
        
        elif HAS_REQUESTS:
            # Fallback: try Wikipedia API directly
            logger.info("Using Wikipedia API directly...")
            for topic in topics:
                try:
                    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'extract' in data:
                            topic_sentences = self._extract_sentences(data['extract'])
                            sentences.extend(topic_sentences[:sentences_per_topic // 10])  # Smaller extract
                            time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Failed to fetch {topic} via API: {e}")
                    continue
        
        # Remove duplicates
        sentences = list(set(sentences))
        logger.info(f"Loaded {len(sentences)} unique sentences from Wikipedia")
        
        if sentences:
            # Cache the results
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, indent=2, ensure_ascii=False)
        
        return sentences
    
    def load_arxiv_abstracts(self, categories: List[str] = None, max_papers: int = 1000) -> List[str]:
        """Load abstracts from arXiv papers in specified categories."""
        if categories is None:
            categories = ["cs.AI", "cs.CL", "cs.LG", "cs.CV", "stat.ML", "physics", "math"]
        
        logger.info(f"Loading arXiv abstracts from {len(categories)} categories")
        
        cache_file = self.cache_dir / f"arxiv_{len(categories)}_{max_papers}.json"
        if cache_file.exists():
            logger.info("Loading arXiv abstracts from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        sentences = []
        
        if HAS_REQUESTS:
            import xml.etree.ElementTree as ET
            
            for category in categories:
                try:
                    logger.info(f"Fetching arXiv papers from category: {category}")
                    
                    # arXiv API query
                    papers_per_category = max(1, max_papers // len(categories))  # Ensure at least 1 paper per category
                    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results={papers_per_category}"
                    
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        # Parse XML response
                        root = ET.fromstring(response.content)
                        
                        # Extract abstracts
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        entries = root.findall('atom:entry', ns)
                        
                        category_sentences = []
                        for entry in entries:
                            summary = entry.find('atom:summary', ns)
                            if summary is not None and summary.text:
                                abstract = summary.text.strip()
                                if abstract:  # Make sure abstract is not empty
                                    # Split abstract into sentences
                                    abstract_sentences = self._extract_sentences(abstract)
                                    category_sentences.extend(abstract_sentences)
                        
                        sentences.extend(category_sentences)
                        logger.info(f"Added {len(category_sentences)} sentences from {len(entries)} papers in {category}")
                    
                    # Rate limiting for arXiv API
                    time.sleep(3)
                    
                except Exception as e:
                    logger.warning(f"Failed to load arXiv papers from {category}: {e}")
                    continue
        
        # Remove duplicates and filter
        sentences = list(set([s for s in sentences if len(s.split()) >= 8]))
        logger.info(f"Loaded {len(sentences)} unique sentences from arXiv")
        
        if sentences:
            # Cache the results
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, indent=2, ensure_ascii=False)
        
        return sentences
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using simple rules."""
        if not text:
            return []
        
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        import re
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence.split()) >= 5:
                # Remove citations, references, etc.
                sentence = re.sub(r'\[[\d\s,]+\]', '', sentence)  # Remove citations like [1,2,3]
                sentence = re.sub(r'\([^)]*\d{4}[^)]*\)', '', sentence)  # Remove year citations
                sentence = sentence.strip()
                if sentence and not sentence.startswith(('Figure', 'Table', 'References')):
                    cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def load_combined_corpus(self, target_size: int = 50000) -> List[str]:
        """Load and combine sentences from all available sources."""
        logger.info(f"Building combined corpus with target size: {target_size}")
        
        all_sentences = []
        
        # Load PAWS dataset (paraphrase detection)
        paws_sentences = self.load_paws_dataset(max_samples=target_size // 3)
        all_sentences.extend(paws_sentences)
        logger.info(f"Added {len(paws_sentences)} sentences from PAWS")
        
        # Load Wikipedia articles
        wiki_sentences = self.load_wikipedia_articles(sentences_per_topic=50)
        all_sentences.extend(wiki_sentences)
        logger.info(f"Added {len(wiki_sentences)} sentences from Wikipedia")
        
        # Load arXiv abstracts
        arxiv_sentences = self.load_arxiv_abstracts(max_papers=min(1000, target_size // 10))
        all_sentences.extend(arxiv_sentences)
        logger.info(f"Added {len(arxiv_sentences)} sentences from arXiv")
        
        # Remove duplicates and shuffle
        unique_sentences = list(set(all_sentences))
        random.shuffle(unique_sentences)
        
        # Limit to target size
        if len(unique_sentences) > target_size:
            unique_sentences = unique_sentences[:target_size]
        
        logger.info(f"Final corpus size: {len(unique_sentences)} sentences")
        
        if not unique_sentences:
            logger.error("No sentences loaded from any dataset! Check network connection and API access.")
            raise RuntimeError("Failed to load any dataset - cannot proceed without real data sources")
        
        return unique_sentences


def get_default_corpus(target_size: int = 50000) -> List[str]:
    """Get default corpus from real datasets only - no hardcoded fallbacks."""
    loader = DatasetLoader()
    return loader.load_combined_corpus(target_size)