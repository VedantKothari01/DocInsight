"""
Dataset loaders for DocInsight - Download and process real datasets for plagiarism detection
"""
import os
import json
import urllib.request
import zipfile
import tarfile
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import random
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """Handles downloading and processing of datasets for corpus building."""
    
    def __init__(self, cache_dir: str = "./dataset_cache"):
        """Initialize dataset loader with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_paws_dataset(self, subset_size: int = 10000) -> List[str]:
        """Download and process PAWS (Paraphrase Adversaries from Word Scrambling) dataset."""
        logger.info("Loading PAWS dataset...")
        
        try:
            # Load PAWS dataset from Hugging Face
            dataset = load_dataset("paws", "labeled_final", split="train")
            
            sentences = []
            count = 0
            
            for item in dataset:
                if count >= subset_size:
                    break
                    
                # Extract sentences from both sentence1 and sentence2
                if item['sentence1'] and len(item['sentence1'].strip()) > 10:
                    sentences.append(item['sentence1'].strip())
                    count += 1
                    
                if count < subset_size and item['sentence2'] and len(item['sentence2'].strip()) > 10:
                    sentences.append(item['sentence2'].strip())
                    count += 1
            
            logger.info(f"Loaded {len(sentences)} sentences from PAWS dataset")
            return sentences
            
        except Exception as e:
            logger.warning(f"Error loading PAWS dataset: {e}")
            # Return fallback PAWS-like sentences
            return self._get_fallback_paws_sentences(min(subset_size, 2000))
    
    def download_wikipedia_articles(self, num_articles: int = 5000) -> List[str]:
        """Download Wikipedia articles and extract sentences."""
        logger.info("Loading Wikipedia dataset...")
        
        try:
            # Load Wikipedia dataset from Hugging Face
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            
            sentences = []
            article_count = 0
            
            for article in dataset:
                if article_count >= num_articles:
                    break
                    
                text = article['text']
                if text and len(text) > 100:
                    # Split into sentences
                    import nltk
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt')
                    
                    from nltk.tokenize import sent_tokenize
                    article_sentences = sent_tokenize(text)
                    
                    # Filter and clean sentences
                    for sent in article_sentences:
                        sent = sent.strip()
                        if (len(sent) > 20 and len(sent) < 200 and 
                            not sent.startswith('==') and 
                            not sent.startswith('{{') and
                            '.' in sent):
                            sentences.append(sent)
                            
                            if len(sentences) >= num_articles * 3:  # ~3 sentences per article
                                break
                    
                article_count += 1
                
                if len(sentences) >= num_articles * 3:
                    break
            
            logger.info(f"Loaded {len(sentences)} sentences from Wikipedia")
            return sentences
            
        except Exception as e:
            logger.warning(f"Error loading Wikipedia dataset: {e}")
            # Return fallback Wikipedia-like sentences
            return self._get_fallback_wikipedia_sentences(min(num_articles * 3, 10000))
    
    def download_arxiv_abstracts(self, num_abstracts: int = 2000) -> List[str]:
        """Download arXiv paper abstracts."""
        logger.info("Loading arXiv abstracts...")
        
        try:
            # Load scientific papers dataset
            dataset = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
            
            sentences = []
            count = 0
            
            for paper in dataset:
                if count >= num_abstracts:
                    break
                    
                abstract = paper['abstract']
                if abstract and len(abstract) > 50:
                    # Split abstract into sentences
                    import nltk
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt')
                    
                    from nltk.tokenize import sent_tokenize
                    abstract_sentences = sent_tokenize(abstract)
                    
                    for sent in abstract_sentences:
                        sent = sent.strip()
                        if len(sent) > 20 and len(sent) < 200:
                            sentences.append(sent)
                
                count += 1
            
            logger.info(f"Loaded {len(sentences)} sentences from arXiv abstracts")
            return sentences
            
        except Exception as e:
            logger.warning(f"Error loading arXiv dataset: {e}")
            # Return fallback arXiv-like sentences
            return self._get_fallback_arxiv_sentences(min(num_abstracts * 2, 4000))
    
    def download_common_crawl_academic(self, subset_size: int = 3000) -> List[str]:
        """Download academic content from Common Crawl or similar sources."""
        logger.info("Loading academic content...")
        
        # For now, we'll use a curated set of academic sentences
        # In a full implementation, this could pull from Common Crawl academic domains
        academic_sentences = [
            "The methodology employed in this study follows a quantitative research design.",
            "Results indicate a statistically significant correlation between the variables.",
            "The literature review reveals gaps in current understanding of the phenomenon.",
            "Data collection was conducted through structured interviews and surveys.",
            "The findings contribute to the existing body of knowledge in this field.",
            "Further research is needed to validate these preliminary conclusions.",
            "The theoretical framework is based on established academic theories.",
            "Statistical analysis was performed using appropriate software packages.",
            "The sample size was determined through power analysis calculations.",
            "Ethical considerations were addressed throughout the research process.",
            "The research questions were formulated based on identified knowledge gaps.",
            "Peer review ensures the quality and validity of academic publications.",
            "Citation practices vary across different academic disciplines.",
            "Academic integrity is fundamental to scholarly research.",
            "The peer review process helps maintain standards in academic publishing.",
        ]
        
        # Replicate and vary the sentences to create a larger set
        sentences = []
        for _ in range(subset_size // len(academic_sentences) + 1):
            sentences.extend(academic_sentences)
        
        return sentences[:subset_size]
    
    def generate_synthetic_paraphrases(self, base_sentences: List[str], num_paraphrases: int = 1000) -> List[str]:
        """Generate synthetic paraphrases using simple techniques."""
        logger.info("Generating synthetic paraphrases...")
        
        paraphrases = []
        
        # Simple paraphrasing techniques
        synonyms = {
            'important': 'significant', 'critical': 'crucial', 'large': 'substantial',
            'small': 'minor', 'good': 'effective', 'bad': 'poor', 'new': 'recent',
            'old': 'traditional', 'fast': 'rapid', 'slow': 'gradual', 'high': 'elevated',
            'low': 'reduced', 'increase': 'enhance', 'decrease': 'reduce'
        }
        
        for _ in range(min(num_paraphrases, len(base_sentences))):
            sentence = random.choice(base_sentences)
            words = sentence.split()
            
            # Simple synonym replacement
            new_words = []
            for word in words:
                clean_word = word.lower().strip('.,!?;:')
                if clean_word in synonyms:
                    # 50% chance to replace
                    if random.random() < 0.5:
                        new_words.append(synonyms[clean_word])
                    else:
                        new_words.append(word)
                else:
                    new_words.append(word)
            
            paraphrase = ' '.join(new_words)
            if paraphrase != sentence and len(paraphrase) > 20:
                paraphrases.append(paraphrase)
        
        logger.info(f"Generated {len(paraphrases)} synthetic paraphrases")
        return paraphrases
    
    def _get_fallback_paws_sentences(self, target_size: int) -> List[str]:
        """Get fallback PAWS-like sentences when online dataset is unavailable."""
        base_sentences = [
            "The movie was excellent and I really enjoyed watching it.",
            "I really enjoyed watching the movie as it was excellent.",
            "The film was outstanding and provided great entertainment.",
            "Climate change is one of the most pressing issues of our time.",
            "Global warming represents one of the most urgent challenges today.",
            "Environmental changes pose significant risks to our planet.",
            "Machine learning algorithms can process vast amounts of data efficiently.",
            "Artificial intelligence systems are capable of handling large datasets.",
            "Advanced computational methods enable analysis of complex information.",
            "The research methodology was comprehensive and well-designed.",
            "The study employed a thorough and systematic approach.",
            "The investigation used rigorous scientific methods.",
            "Education plays a crucial role in societal development.",
            "Learning is essential for the progress of human civilization.",
            "Knowledge acquisition drives social advancement.",
            "Technology has transformed the way we communicate with each other.",
            "Digital innovations have revolutionized interpersonal communication.",
            "Modern tools have changed how people interact and share information.",
        ]
        
        # Expand the base set by creating variations
        sentences = []
        for _ in range(target_size // len(base_sentences) + 1):
            sentences.extend(base_sentences)
        
        return sentences[:target_size]
    
    def _get_fallback_wikipedia_sentences(self, target_size: int) -> List[str]:
        """Get fallback Wikipedia-like sentences when online dataset is unavailable."""
        base_sentences = [
            "The Earth is the third planet from the Sun and the only known planet to harbor life.",
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "The human brain contains approximately 86 billion neurons.",
            "Water covers approximately 71 percent of the Earth's surface.",
            "The speed of light in vacuum is exactly 299,792,458 meters per second.",
            "DNA carries genetic information in all living organisms.",
            "The Great Wall of China is one of the most famous architectural achievements in history.",
            "Shakespeare wrote approximately 39 plays and 154 sonnets during his career.",
            "The periodic table organizes chemical elements by their atomic structure.",
            "Gravity is the force that attracts objects with mass toward each other.",
            "The Amazon rainforest is often referred to as the lungs of the Earth.",
            "Ancient civilizations developed writing systems to record information.",
            "The Industrial Revolution began in Britain during the 18th century.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "The theory of evolution explains the diversity of life on Earth.",
            "Mathematics is the language used to describe patterns in nature.",
            "The Internet has revolutionized global communication and information sharing.",
            "Biodiversity refers to the variety of life forms in different ecosystems.",
            "Democracy is a form of government where power is held by the people.",
            "Scientific research follows systematic methods to understand natural phenomena.",
        ]
        
        # Expand and vary the sentences
        sentences = []
        variations = [
            lambda s: s,  # Original
            lambda s: s.replace("The ", "This "),
            lambda s: s.replace(" is ", " was "),
            lambda s: s.replace("approximately", "about"),
        ]
        
        for _ in range(target_size // (len(base_sentences) * len(variations)) + 1):
            for sentence in base_sentences:
                for variation in variations:
                    if len(sentences) < target_size:
                        sentences.append(variation(sentence))
        
        return sentences[:target_size]
    
    def _get_fallback_arxiv_sentences(self, target_size: int) -> List[str]:
        """Get fallback arXiv-like sentences when online dataset is unavailable."""
        base_sentences = [
            "We propose a novel approach for solving complex optimization problems.",
            "The experimental results demonstrate significant improvements over baseline methods.",
            "Our method achieves state-of-the-art performance on multiple benchmark datasets.",
            "The theoretical analysis provides insights into the algorithm's convergence properties.",
            "We evaluate our approach using standard metrics and comparison with existing work.",
            "The proposed framework can be applied to various domains and applications.",
            "Future work will explore extensions to handle more complex scenarios.",
            "The computational complexity of the algorithm is analyzed in detail.",
            "Extensive experiments validate the effectiveness of the proposed method.",
            "The results show promising potential for real-world applications.",
            "We introduce a new mathematical formulation for this challenging problem.",
            "The empirical evaluation confirms the theoretical predictions.",
            "Our contributions include both algorithmic innovations and practical implementations.",
            "The paper presents a comprehensive study of the problem domain.",
            "Statistical significance tests confirm the reliability of our findings.",
            "The methodology follows established best practices in the field.",
            "We compare our results with previous work using identical experimental setups.",
            "The proposed solution addresses key limitations of existing approaches.",
            "Cross-validation experiments ensure the robustness of our conclusions.",
            "The research has implications for both theoretical understanding and practical applications.",
        ]
        
        # Create variations with different phrasings
        sentences = []
        for _ in range(target_size // len(base_sentences) + 1):
            sentences.extend(base_sentences)
        
        return sentences[:target_size]
    
    def load_all_datasets(self, total_target: int = 50000) -> List[str]:
        """Load and combine all datasets to create a comprehensive corpus."""
        logger.info(f"Loading all datasets with target size: {total_target}")
        
        all_sentences = []
        
        # Distribution of sentences across datasets
        paws_size = min(10000, total_target // 4)
        wiki_size = min(20000, total_target // 2)
        arxiv_size = min(5000, total_target // 8)
        academic_size = min(3000, total_target // 10)
        synthetic_size = min(2000, total_target // 20)
        
        # Load each dataset
        try:
            paws_sentences = self.download_paws_dataset(paws_size)
            all_sentences.extend(paws_sentences)
            
            wiki_sentences = self.download_wikipedia_articles(wiki_size // 3)
            all_sentences.extend(wiki_sentences)
            
            arxiv_sentences = self.download_arxiv_abstracts(arxiv_size // 2)
            all_sentences.extend(arxiv_sentences)
            
            academic_sentences = self.download_common_crawl_academic(academic_size)
            all_sentences.extend(academic_sentences)
            
            # Generate synthetic paraphrases from existing sentences
            if all_sentences:
                synthetic_sentences = self.generate_synthetic_paraphrases(all_sentences[:100], synthetic_size)
                all_sentences.extend(synthetic_sentences)
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
        
        # Remove duplicates and filter
        unique_sentences = list(set(all_sentences))
        
        # Filter sentences by length and quality
        filtered_sentences = []
        for sent in unique_sentences:
            sent = sent.strip()
            if (10 <= len(sent) <= 300 and 
                len(sent.split()) >= 3 and
                any(c.isalpha() for c in sent)):
                filtered_sentences.append(sent)
        
        # Shuffle and trim to target size
        random.shuffle(filtered_sentences)
        final_sentences = filtered_sentences[:total_target]
        
        logger.info(f"Final corpus size: {len(final_sentences)} sentences")
        return final_sentences
    
    def save_corpus(self, sentences: List[str], filename: str = "corpus.json") -> str:
        """Save corpus to file."""
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Corpus saved to {filepath}")
        return str(filepath)
    
    def load_corpus(self, filename: str = "corpus.json") -> List[str]:
        """Load corpus from file."""
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                sentences = json.load(f)
            logger.info(f"Corpus loaded from {filepath}: {len(sentences)} sentences")
            return sentences
        else:
            logger.warning(f"Corpus file not found: {filepath}")
            return []

# Convenience function for easy import
def get_default_corpus(force_download: bool = False, target_size: int = 50000) -> List[str]:
    """Get the default corpus, downloading if necessary."""
    loader = DatasetLoader()
    
    if not force_download:
        # Try to load existing corpus
        corpus = loader.load_corpus()
        if corpus and len(corpus) >= target_size * 0.8:  # Accept if we have 80% of target
            return corpus
    
    # Download and create new corpus
    logger.info("Downloading and creating new corpus...")
    corpus = loader.load_all_datasets(target_size)
    
    if corpus:
        loader.save_corpus(corpus)
    
    return corpus