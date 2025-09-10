"""
Dataset loaders for DocInsight - Download and process real datasets with fallbacks
"""
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Defensive imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Extended fallback corpus for when datasets can't be loaded
EXTENDED_FALLBACK_CORPUS = [
    # Technology & AI
    "Machine learning algorithms can identify patterns in large datasets.",
    "Neural networks use interconnected nodes to process information.",
    "Deep learning requires substantial computational resources and data.",
    "Natural language processing enables computers to understand human text.",
    "Computer vision systems can analyze and interpret digital images.",
    "Artificial intelligence aims to replicate human cognitive abilities.",
    "Data science combines statistical analysis with programming skills.",
    "Algorithm optimization improves software performance and efficiency.",
    "Cloud computing provides scalable and flexible IT infrastructure.",
    "Cybersecurity measures protect against digital threats and attacks.",
    
    # Science & Environment
    "Climate change significantly impacts global weather patterns.",
    "Renewable energy sources help reduce greenhouse gas emissions.",
    "Sustainable development balances economic needs with environmental protection.",
    "Biodiversity loss threatens the stability of natural ecosystems.",
    "Ocean acidification affects marine organisms and food webs.",
    "Scientific research relies on systematic observation and experimentation.",
    "Genetic engineering techniques can modify organism characteristics.",
    "Quantum mechanics describes the behavior of subatomic particles.",
    "Evolution explains the diversity of life through natural selection.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    
    # Education & Society  
    "Education technology transforms traditional teaching methods.",
    "Digital literacy skills are essential in modern society.",
    "Remote work arrangements change organizational dynamics.",
    "Social media platforms influence public opinion and behavior.",
    "Healthcare innovations improve patient outcomes and quality of life.",
    "Economic policies affect market stability and growth.",
    "Cultural diversity enriches communities and perspectives.",
    "Democracy depends on informed citizen participation.",
    "Human rights principles ensure dignity and equality for all.",
    "Globalization connects economies and cultures worldwide.",
    
    # Research & Academia
    "Academic research contributes to knowledge advancement.",
    "Peer review ensures the quality of scientific publications.",
    "Literature reviews synthesize existing research findings.",
    "Hypothesis testing follows rigorous scientific methodology.",
    "Data analysis reveals trends and correlations in research.",
    "Statistical significance indicates reliable research results.",
    "Interdisciplinary studies combine multiple academic fields.",
    "Research ethics protect participants and maintain integrity.",
    "Publication standards ensure transparency and reproducibility.",
    "Collaborative research accelerates scientific discovery.",
    
    # Business & Innovation
    "Innovation drives competitive advantage in business markets.",
    "Entrepreneurship creates new ventures and economic opportunities.",
    "Strategic planning guides organizational decision making.",
    "Customer satisfaction influences business success and growth.",
    "Supply chain management optimizes product delivery systems.",
    "Digital transformation modernizes business operations.",
    "Market research identifies consumer needs and preferences.",
    "Financial analysis evaluates investment opportunities.",
    "Project management coordinates resources and timelines.",
    "Quality assurance ensures products meet standards.",
]

def get_fallback_corpus(target_size: int = 1000) -> List[str]:
    """Generate fallback corpus with variations."""
    logger.info(f"Generating fallback corpus with target size: {target_size}")
    
    base_sentences = EXTENDED_FALLBACK_CORPUS.copy()
    corpus = base_sentences.copy()
    
    # Add variations to reach target size
    while len(corpus) < target_size:
        for sentence in base_sentences:
            if len(corpus) >= target_size:
                break
            
            # Add simple variations
            variations = [
                f"Research indicates that {sentence.lower()}",
                f"Studies show that {sentence.lower()}",
                f"Evidence suggests that {sentence.lower()}",
                f"Analysis reveals that {sentence.lower()}",
                f"Observations confirm that {sentence.lower()}",
            ]
            
            # Add one random variation
            if variations:
                corpus.append(random.choice(variations))
    
    # Shuffle and return target size
    random.shuffle(corpus)
    return corpus[:target_size]

def try_load_paws_dataset(target_size: int = 10000) -> List[str]:
    """Attempt to load PAWS dataset."""
    if not HAS_DATASETS:
        logger.warning("datasets library not available")
        return []
    
    try:
        logger.info("Attempting to load PAWS dataset...")
        dataset = load_dataset("paws", "labeled_final", split="train", cache_dir="./cache")
        
        sentences = []
        for item in dataset:
            if len(sentences) >= target_size:
                break
            
            # Extract valid sentences
            for sent in [item.get('sentence1', ''), item.get('sentence2', '')]:
                if sent and len(sent.strip()) > 15 and len(sent.strip()) < 200:
                    sentences.append(sent.strip())
                    if len(sentences) >= target_size:
                        break
        
        logger.info(f"Loaded {len(sentences)} sentences from PAWS")
        return sentences
    except Exception as e:
        logger.warning(f"Failed to load PAWS dataset: {e}")
        return []

def try_load_wikipedia_sample() -> List[str]:
    """Attempt to get Wikipedia sample sentences."""
    if not HAS_REQUESTS:
        logger.warning("requests library not available")
        return []
    
    try:
        # Try to get a few Wikipedia articles via API
        logger.info("Attempting to load Wikipedia samples...")
        
        # Sample of article titles to fetch
        titles = [
            "Machine_learning", "Climate_change", "Renewable_energy",
            "Artificial_intelligence", "Data_science", "Computer_vision"
        ]
        
        sentences = []
        for title in titles:
            if len(sentences) >= 1000:  # Limit Wikipedia contribution
                break
                
            try:
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    extract = data.get('extract', '')
                    if extract:
                        # Split into sentences (simple approach)
                        wiki_sentences = [s.strip() + '.' for s in extract.split('.') if len(s.strip()) > 20]
                        sentences.extend(wiki_sentences[:10])  # Take first 10 sentences per article
            except Exception as e:
                logger.debug(f"Failed to fetch {title}: {e}")
                continue
        
        logger.info(f"Loaded {len(sentences)} sentences from Wikipedia")
        return sentences
    except Exception as e:
        logger.warning(f"Failed to load Wikipedia data: {e}")
        return []

def get_default_corpus(target_size: int = 5000) -> List[str]:
    """Get default corpus combining real datasets and fallbacks."""
    logger.info(f"Building corpus with target size: {target_size}")
    
    all_sentences = []
    
    # Try to load real datasets first
    paws_sentences = try_load_paws_dataset(target_size // 2)
    if paws_sentences:
        all_sentences.extend(paws_sentences)
        logger.info(f"Added {len(paws_sentences)} PAWS sentences")
    
    wiki_sentences = try_load_wikipedia_sample()
    if wiki_sentences:
        all_sentences.extend(wiki_sentences)
        logger.info(f"Added {len(wiki_sentences)} Wikipedia sentences")
    
    # Fill remaining with fallback corpus
    remaining = target_size - len(all_sentences)
    if remaining > 0:
        fallback_sentences = get_fallback_corpus(remaining)
        all_sentences.extend(fallback_sentences)
        logger.info(f"Added {len(fallback_sentences)} fallback sentences")
    
    # Remove duplicates and clean
    seen = set()
    clean_sentences = []
    for sentence in all_sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10 and sentence not in seen:
            seen.add(sentence)
            clean_sentences.append(sentence)
    
    # Shuffle and trim to target size
    random.shuffle(clean_sentences)
    final_corpus = clean_sentences[:target_size]
    
    logger.info(f"Built final corpus with {len(final_corpus)} unique sentences")
    return final_corpus