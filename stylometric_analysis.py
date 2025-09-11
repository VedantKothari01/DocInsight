"""
Enhanced Stylometric Analysis for DocInsight - Research-Focused Implementation
=============================================================================

Implements SRS v0.2 requirements for stylometric evidence ensemble:
- Academic writing style analysis
- Author attribution features
- AI-detection integration
- Ensemble stylometric classifiers

This module supports the research goal of creating measurable improvements
over baseline semantic-only systems through stylometric corroboration.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import math

# Defensive imports for research environment
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StyleFeatures:
    """Container for stylometric features."""
    # Lexical features
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0
    hapax_legomena_ratio: float = 0.0  # Words appearing exactly once
    
    # Syntactic features
    pos_distribution: Dict[str, float] = None
    function_word_ratio: float = 0.0
    punctuation_frequency: Dict[str, float] = None
    
    # Readability features
    flesch_kincaid_grade: float = 0.0
    gunning_fog_index: float = 0.0
    dale_chall_score: float = 0.0
    
    # Academic writing features
    passive_voice_ratio: float = 0.0
    modal_verb_ratio: float = 0.0
    academic_word_ratio: float = 0.0
    citation_density: float = 0.0
    
    # AI detection features
    perplexity_estimate: float = 0.0
    repetition_score: float = 0.0
    coherence_score: float = 0.0
    
    def __post_init__(self):
        if self.pos_distribution is None:
            self.pos_distribution = {}
        if self.punctuation_frequency is None:
            self.punctuation_frequency = {}


class AcademicStylometricAnalyzer:
    """
    Enhanced stylometric analyzer for academic writing.
    
    Implements SRS v0.2 requirements for stylometric evidence ensemble
    with focus on academic writing patterns and AI detection.
    """
    
    def __init__(self):
        self.nlp = None
        self.academic_words = set()
        self.function_words = set()
        self._init_resources()
    
    def _init_resources(self):
        """Initialize NLP resources and word lists."""
        # Load spaCy model if available
        if HAS_SPACY:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded for stylometric analysis")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
        
        # Academic word list (Academic Word List - AWL subset)
        self.academic_words = {
            'analysis', 'approach', 'assessment', 'concept', 'consistent', 'constitute',
            'data', 'definition', 'derived', 'distribution', 'economic', 'environment',
            'established', 'estimate', 'evidence', 'factor', 'function', 'identified',
            'income', 'individual', 'interpretation', 'involved', 'issues', 'labour',
            'legal', 'legislation', 'major', 'method', 'occurred', 'percent',
            'period', 'policy', 'principle', 'procedure', 'process', 'required',
            'research', 'response', 'role', 'section', 'significant', 'similar',
            'source', 'specific', 'structure', 'theory', 'variables', 'analysis',
            'hypothesis', 'methodology', 'evaluation', 'investigation', 'conclusion',
            'furthermore', 'therefore', 'however', 'moreover', 'consequently'
        }
        
        # Function words for stylometric analysis
        self.function_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'she', 'or', 'which',
            'an', 'we', 'say', 'her', 'she', 'or', 'all', 'would', 'there',
            'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
            'just', 'him', 'know', 'take', 'people', 'into', 'year'
        }
    
    def extract_lexical_features(self, text: str) -> Dict[str, float]:
        """Extract lexical features from text."""
        if not text.strip():
            return {}
        
        # Basic tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {}
        
        # Word length statistics
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Sentence length statistics
        sentence_lengths = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        # Vocabulary richness (Type-Token Ratio)
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0
        
        # Hapax legomena (words appearing exactly once)
        word_counts = Counter(words)
        hapax_legomena = sum(1 for count in word_counts.values() if count == 1)
        hapax_legomena_ratio = hapax_legomena / len(unique_words) if unique_words else 0
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'vocabulary_richness': vocabulary_richness,
            'hapax_legomena_ratio': hapax_legomena_ratio
        }
    
    def extract_syntactic_features(self, text: str) -> Dict[str, Any]:
        """Extract syntactic features using spaCy if available."""
        features = {
            'pos_distribution': {},
            'function_word_ratio': 0.0,
            'punctuation_frequency': {}
        }
        
        if not text.strip():
            return features
        
        # Punctuation analysis
        punctuation_marks = ['.', ',', ';', ':', '!', '?', '-', '"', "'"]
        total_chars = len(text)
        punctuation_frequency = {}
        
        for punct in punctuation_marks:
            count = text.count(punct)
            punctuation_frequency[punct] = count / total_chars if total_chars > 0 else 0
        
        features['punctuation_frequency'] = punctuation_frequency
        
        # Basic function word analysis
        words = re.findall(r'\b\w+\b', text.lower())
        if words:
            function_word_count = sum(1 for word in words if word in self.function_words)
            features['function_word_ratio'] = function_word_count / len(words)
        
        # Advanced syntactic analysis with spaCy
        if self.nlp and HAS_SPACY:
            try:
                doc = self.nlp(text)
                
                # POS tag distribution
                pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)
                total_pos = sum(pos_counts.values())
                
                if total_pos > 0:
                    pos_distribution = {pos: count / total_pos for pos, count in pos_counts.items()}
                    features['pos_distribution'] = pos_distribution
                
            except Exception as e:
                logger.warning(f"spaCy analysis failed: {e}")
        
        return features
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability features."""
        features = {
            'flesch_kincaid_grade': 0.0,
            'gunning_fog_index': 0.0,
            'dale_chall_score': 0.0
        }
        
        if not text.strip() or not HAS_TEXTSTAT:
            return features
        
        try:
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid().flesch_kincaid(text)
            features['gunning_fog_index'] = textstat.gunning_fog(text)
            features['dale_chall_score'] = textstat.dale_chall_readability_score(text)
        except Exception as e:
            logger.warning(f"Readability analysis failed: {e}")
        
        return features
    
    def extract_academic_features(self, text: str) -> Dict[str, float]:
        """Extract academic writing specific features."""
        features = {
            'passive_voice_ratio': 0.0,
            'modal_verb_ratio': 0.0,
            'academic_word_ratio': 0.0,
            'citation_density': 0.0
        }
        
        if not text.strip():
            return features
        
        # Academic word ratio
        words = re.findall(r'\b\w+\b', text.lower())
        if words:
            academic_word_count = sum(1 for word in words if word in self.academic_words)
            features['academic_word_ratio'] = academic_word_count / len(words)
        
        # Citation density (simple pattern matching)
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[\d+\]',              # [1]
            r'\[[\d\s,]+\]',         # [1, 2, 3]
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            total_citations += len(re.findall(pattern, text))
        
        word_count = len(words) if words else 1
        features['citation_density'] = total_citations / (word_count / 100)  # Citations per 100 words
        
        # Passive voice detection (simplified)
        passive_indicators = ['was', 'were', 'been', 'being']
        past_participle_endings = ['ed', 'en', 'ne']
        
        sentences = re.split(r'[.!?]+', text.lower())
        passive_count = 0
        
        for sentence in sentences:
            if any(indicator in sentence for indicator in passive_indicators):
                # Check for past participles nearby
                sentence_words = re.findall(r'\b\w+\b', sentence)
                for word in sentence_words:
                    if any(word.endswith(ending) for ending in past_participle_endings):
                        passive_count += 1
                        break
        
        if sentences:
            features['passive_voice_ratio'] = passive_count / len([s for s in sentences if s.strip()])
        
        # Modal verb ratio
        modal_verbs = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
        if words:
            modal_count = sum(1 for word in words if word in modal_verbs)
            features['modal_verb_ratio'] = modal_count / len(words)
        
        return features
    
    def extract_ai_detection_features(self, text: str) -> Dict[str, float]:
        """Extract features for AI-generated text detection."""
        features = {
            'perplexity_estimate': 0.0,
            'repetition_score': 0.0,
            'coherence_score': 0.0
        }
        
        if not text.strip():
            return features
        
        # Repetition analysis
        sentences = re.split(r'[.!?]+', text.lower())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            # Calculate sentence similarity (simplified)
            repetition_score = 0.0
            comparisons = 0
            
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    words_i = set(re.findall(r'\b\w+\b', sentences[i]))
                    words_j = set(re.findall(r'\b\w+\b', sentences[j]))
                    
                    if words_i and words_j:
                        overlap = len(words_i & words_j) / len(words_i | words_j)
                        repetition_score += overlap
                        comparisons += 1
            
            if comparisons > 0:
                features['repetition_score'] = repetition_score / comparisons
        
        # Perplexity estimate (simplified - based on word frequency distribution)
        words = re.findall(r'\b\w+\b', text.lower())
        if words:
            word_counts = Counter(words)
            total_words = len(words)
            
            # Calculate entropy as perplexity proxy
            entropy = 0.0
            for count in word_counts.values():
                prob = count / total_words
                entropy -= prob * math.log2(prob)
            
            features['perplexity_estimate'] = 2 ** entropy
        
        # Coherence score (simplified - based on function word consistency)
        coherence_indicators = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 'thus']
        coherence_count = sum(1 for word in words if word in coherence_indicators)
        features['coherence_score'] = coherence_count / len(sentences) if sentences else 0
        
        return features
    
    def analyze_text(self, text: str) -> StyleFeatures:
        """
        Comprehensive stylometric analysis of text.
        
        Implements SRS v0.2 stylometric evidence ensemble.
        """
        if not text or not text.strip():
            return StyleFeatures()
        
        logger.debug(f"Analyzing stylometric features for {len(text)} characters of text")
        
        # Extract all feature categories
        lexical_features = self.extract_lexical_features(text)
        syntactic_features = self.extract_syntactic_features(text)
        readability_features = self.extract_readability_features(text)
        academic_features = self.extract_academic_features(text)
        ai_features = self.extract_ai_detection_features(text)
        
        # Combine into StyleFeatures object
        features = StyleFeatures()
        
        # Lexical features
        features.avg_word_length = lexical_features.get('avg_word_length', 0.0)
        features.avg_sentence_length = lexical_features.get('avg_sentence_length', 0.0)
        features.vocabulary_richness = lexical_features.get('vocabulary_richness', 0.0)
        features.hapax_legomena_ratio = lexical_features.get('hapax_legomena_ratio', 0.0)
        
        # Syntactic features
        features.pos_distribution = syntactic_features.get('pos_distribution', {})
        features.function_word_ratio = syntactic_features.get('function_word_ratio', 0.0)
        features.punctuation_frequency = syntactic_features.get('punctuation_frequency', {})
        
        # Readability features
        features.flesch_kincaid_grade = readability_features.get('flesch_kincaid_grade', 0.0)
        features.gunning_fog_index = readability_features.get('gunning_fog_index', 0.0)
        features.dale_chall_score = readability_features.get('dale_chall_score', 0.0)
        
        # Academic features
        features.passive_voice_ratio = academic_features.get('passive_voice_ratio', 0.0)
        features.modal_verb_ratio = academic_features.get('modal_verb_ratio', 0.0)
        features.academic_word_ratio = academic_features.get('academic_word_ratio', 0.0)
        features.citation_density = academic_features.get('citation_density', 0.0)
        
        # AI detection features
        features.perplexity_estimate = ai_features.get('perplexity_estimate', 0.0)
        features.repetition_score = ai_features.get('repetition_score', 0.0)
        features.coherence_score = ai_features.get('coherence_score', 0.0)
        
        return features
    
    def compare_styles(self, features1: StyleFeatures, features2: StyleFeatures) -> float:
        """
        Compare stylometric features between two texts.
        
        Returns similarity score (0-1, higher = more similar).
        """
        if not HAS_NUMPY:
            # Simplified comparison without numpy
            similarity_scores = []
            
            # Compare key numerical features
            numerical_features = [
                ('avg_word_length', 1.0),
                ('avg_sentence_length', 5.0),
                ('vocabulary_richness', 0.2),
                ('function_word_ratio', 0.1),
                ('flesch_kincaid_grade', 3.0),
                ('academic_word_ratio', 0.1),
                ('passive_voice_ratio', 0.1)
            ]
            
            for feature_name, scale in numerical_features:
                val1 = getattr(features1, feature_name, 0.0)
                val2 = getattr(features2, feature_name, 0.0)
                
                if scale > 0:
                    diff = abs(val1 - val2) / scale
                    similarity = max(0, 1 - diff)
                    similarity_scores.append(similarity)
            
            return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        # Advanced comparison with numpy
        try:
            import numpy as np  # Import here to avoid global dependency
            
            # Create feature vectors
            vector1 = self._features_to_vector(features1)
            vector2 = self._features_to_vector(features2)
            
            # Compute cosine similarity
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_sim = dot_product / (norm1 * norm2)
                return max(0, cosine_sim)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Advanced style comparison failed: {e}")
        
        return 0.0
    
    def _features_to_vector(self, features: StyleFeatures):
        """Convert StyleFeatures to numerical vector for comparison."""
        if not HAS_NUMPY:
            return []
        
        import numpy as np  # Import here to avoid global dependency
        
        # Create normalized feature vector
        vector_components = [
            features.avg_word_length / 10.0,  # Normalize by typical range
            features.avg_sentence_length / 20.0,
            features.vocabulary_richness,
            features.hapax_legomena_ratio,
            features.function_word_ratio,
            features.flesch_kincaid_grade / 20.0,
            features.gunning_fog_index / 20.0,
            features.passive_voice_ratio,
            features.modal_verb_ratio,
            features.academic_word_ratio,
            features.citation_density / 10.0,
            features.perplexity_estimate / 1000.0,  # Normalize perplexity
            features.repetition_score,
            features.coherence_score
        ]
        
        return np.array(vector_components)


def create_academic_stylometric_analyzer() -> AcademicStylometricAnalyzer:
    """Create academic stylometric analyzer for DocInsight research."""
    return AcademicStylometricAnalyzer()