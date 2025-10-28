"""
Stylometry features module for DocInsight

Provides comprehensive stylometric feature extraction for writing style analysis
and AI-likeness detection. Includes both document-level and sentence-level features.
"""

import re
import string
import math
import logging
from typing import Dict, List, Any, Optional
from collections import Counter

from config import FUNCTION_WORDS

logger = logging.getLogger(__name__)


class StylemetryFeatureExtractor:
    """Extracts stylometric features from text for writing style analysis"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.function_words = set(word.lower() for word in FUNCTION_WORDS)
        self.punctuation = set(string.punctuation)
    
    def compute_baseline_features(self, text: str) -> Dict[str, float]:
        """Compute document-level stylometric features
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of stylometric features
        """
        if not text.strip():
            return self._empty_features()
        
        # Basic tokenization
        sentences = self._split_sentences(text)
        tokens = self._tokenize(text)
        words = [token for token in tokens if token.isalpha()]
        
        if not words:
            return self._empty_features()
        
        features = {}
        
        # Basic counts
        features['token_count'] = len(tokens)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['unique_token_count'] = len(set(token.lower() for token in tokens))
        
        # Lexical diversity
        features['type_token_ratio'] = features['unique_token_count'] / features['token_count']
        features['word_diversity'] = len(set(word.lower() for word in words)) / len(words)
        
        # Length features
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        features['char_count'] = len(text)
        features['avg_chars_per_sentence'] = len(text) / len(sentences) if sentences else 0
        
        # Punctuation analysis
        punct_chars = [char for char in text if char in self.punctuation]
        features['punctuation_count'] = len(punct_chars)
        features['punctuation_density'] = len(punct_chars) / len(text)
        
        # Function word analysis
        function_word_count = sum(1 for word in words if word.lower() in self.function_words)
        features['function_word_count'] = function_word_count
        features['stopword_ratio'] = function_word_count / len(words)
        
        # Function word vector (frequency of each function word)
        function_word_vector = self._compute_function_word_vector(words)
        features.update(function_word_vector)
        
        # Character n-gram entropy
        features['char_trigram_entropy'] = self._compute_char_ngram_entropy(text, n=3)
        features['char_bigram_entropy'] = self._compute_char_ngram_entropy(text, n=2)
        
        # Sentence length variance
        sentence_lengths = [len(self._tokenize(sent)) for sent in sentences]
        if len(sentence_lengths) > 1:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((length - mean_len) ** 2 for length in sentence_lengths) / len(sentence_lengths)
            features['sentence_length_variance'] = variance
        else:
            features['sentence_length_variance'] = 0.0
        
        # Complexity features
        features['complexity_score'] = self._compute_complexity_score(text, features)
        
        return features
    
    def compute_sentence_features(self, sentence: str) -> Dict[str, float]:
        """Compute sentence-level stylometric features (subset for potential expansion)
        
        Args:
            sentence: Input sentence to analyze
            
        Returns:
            Dictionary of sentence-level features
        """
        if not sentence.strip():
            return {'sentence_length': 0, 'word_count': 0, 'complexity': 0}
        
        tokens = self._tokenize(sentence)
        words = [token for token in tokens if token.isalpha()]
        
        features = {}
        features['sentence_length'] = len(tokens)
        features['word_count'] = len(words)
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
        features['function_word_ratio'] = sum(1 for word in words if word.lower() in self.function_words) / len(words) if words else 0
        features['punctuation_count'] = sum(1 for char in sentence if char in self.punctuation)
        features['complexity'] = len(set(word.lower() for word in words)) / len(words) if words else 0
        
        return features
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Basic sentence splitting on periods, exclamation marks, question marks
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Split on whitespace and punctuation, keep non-empty tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return [token for token in tokens if token.strip()]
    
    def _compute_function_word_vector(self, words: List[str]) -> Dict[str, float]:
        """Compute frequency vector for function words"""
        word_counts = Counter(word.lower() for word in words)
        total_words = len(words)
        
        vector = {}
        for func_word in self.function_words:
            key = f'fw_{func_word}'
            vector[key] = word_counts.get(func_word, 0) / total_words
        
        return vector
    
    def _compute_char_ngram_entropy(self, text: str, n: int = 3) -> float:
        """Compute entropy of character n-grams"""
        if len(text) < n:
            return 0.0
        
        # Generate n-grams
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        
        # Count frequencies
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        # Calculate entropy
        entropy = 0.0
        for count in ngram_counts.values():
            if count > 0:
                prob = count / total_ngrams
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _compute_complexity_score(self, text: str, features: Dict[str, float]) -> float:
        """Compute overall text complexity score"""
        # Combine multiple complexity indicators
        complexity = 0.0
        
        # Lexical complexity
        complexity += features.get('type_token_ratio', 0) * 0.3
        
        # Syntactic complexity (approximated by sentence length variance)
        complexity += min(features.get('sentence_length_variance', 0) / 100, 1.0) * 0.2
        
        # Semantic complexity (approximated by average word length)
        avg_word_len = features.get('avg_word_length', 0)
        complexity += min(avg_word_len / 10, 1.0) * 0.3
        
        # Structural complexity (punctuation usage)
        complexity += min(features.get('punctuation_density', 0) * 10, 1.0) * 0.2
        
        return min(complexity, 1.0)
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature vector for invalid input"""
        features = {
            'token_count': 0, 'word_count': 0, 'sentence_count': 0,
            'unique_token_count': 0, 'type_token_ratio': 0, 'word_diversity': 0,
            'avg_sentence_length': 0, 'avg_word_length': 0, 'char_count': 0,
            'avg_chars_per_sentence': 0, 'punctuation_count': 0, 'punctuation_density': 0,
            'function_word_count': 0, 'stopword_ratio': 0, 'char_trigram_entropy': 0,
            'char_bigram_entropy': 0, 'sentence_length_variance': 0, 'complexity_score': 0
        }
        
        # Add function word features
        for func_word in self.function_words:
            features[f'fw_{func_word}'] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names for consistency"""
        base_features = [
            'token_count', 'word_count', 'sentence_count', 'unique_token_count',
            'type_token_ratio', 'word_diversity', 'avg_sentence_length', 'avg_word_length',
            'char_count', 'avg_chars_per_sentence', 'punctuation_count', 'punctuation_density',
            'function_word_count', 'stopword_ratio', 'char_trigram_entropy',
            'char_bigram_entropy', 'sentence_length_variance', 'complexity_score'
        ]
        
        # Add function word features
        function_word_features = [f'fw_{word}' for word in self.function_words]
        
        return base_features + function_word_features


def extract_stylometric_features(text: str) -> Dict[str, float]:
    """Convenience function to extract stylometric features
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary of stylometric features
    """
    extractor = StylemetryFeatureExtractor()
    return extractor.compute_baseline_features(text)


def compare_stylometric_profiles(features1: Dict[str, float], features2: Dict[str, float]) -> Dict[str, float]:
    """Compare two stylometric feature profiles
    
    Args:
        features1: First feature profile
        features2: Second feature profile
        
    Returns:
        Dictionary with comparison metrics
    """
    if not features1 or not features2:
        return {'similarity': 0.0, 'deviation': 1.0}
    
    # Calculate feature differences
    differences = {}
    common_features = set(features1.keys()) & set(features2.keys())
    
    total_diff = 0.0
    count = 0
    
    for feature in common_features:
        val1 = features1[feature]
        val2 = features2[feature]
        
        # Normalize difference by the larger value to avoid division by zero
        max_val = max(abs(val1), abs(val2), 1e-6)
        diff = abs(val1 - val2) / max_val
        differences[feature] = diff
        
        total_diff += diff
        count += 1
    
    avg_deviation = total_diff / count if count > 0 else 1.0
    similarity = max(0.0, 1.0 - avg_deviation)
    
    return {
        'similarity': similarity,
        'deviation': avg_deviation,
        'feature_differences': differences
    }


# Example usage for testing
if __name__ == "__main__":
    sample_text = """
    This is a sample document for testing stylometric features. The document contains
    multiple sentences with varying complexity. Some sentences are simple, while others
    demonstrate more sophisticated syntactic structures and vocabulary choices.
    """
    
    extractor = StylemetryFeatureExtractor()
    features = extractor.compute_baseline_features(sample_text)
    
    print("Stylometric Features:")
    for feature, value in features.items():
        if isinstance(value, float):
            print(f"{feature}: {value:.4f}")
        else:
            print(f"{feature}: {value}")
    
    print(f"\nTotal features extracted: {len(features)}")
    print(f"Feature names: {extractor.get_feature_names()[:10]}...")  # Show first 10