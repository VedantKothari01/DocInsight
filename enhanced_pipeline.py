"""
Enhanced plagiarism detection pipeline for DocInsight - Clean implementation with fallbacks
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Defensive imports
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

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

from corpus_builder import CorpusIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimilarityMatch:
    """Represents a similarity match result."""
    text: str
    similarity: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"

@dataclass
class AnalysisResult:
    """Results from sentence analysis."""
    sentence: str
    matches: List[SimilarityMatch]
    stylometry_score: float
    fused_score: float
    confidence: str

class PlagiarismDetector:
    """Clean plagiarism detector with fallback capabilities."""
    
    def __init__(self, corpus_index: CorpusIndex):
        """
        Initialize detector with corpus index.
        
        Args:
            corpus_index: Pre-built corpus index
        """
        self.corpus_index = corpus_index
        self._cross_encoder = None
        self._nlp = None
        
    def _load_cross_encoder(self):
        """Lazy load cross-encoder model."""
        if self._cross_encoder is None and HAS_CROSS_ENCODER:
            try:
                logger.info("Loading cross-encoder model...")
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                self._cross_encoder = None
        return self._cross_encoder
    
    def _load_spacy_model(self):
        """Lazy load spaCy model."""
        if self._nlp is None and HAS_SPACY:
            try:
                logger.info("Loading spaCy model...")
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self._nlp = None
        return self._nlp
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Try spaCy first
        nlp = self._load_spacy_model()
        if nlp:
            try:
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            except Exception as e:
                logger.warning(f"spaCy sentence extraction failed: {e}")
        
        # Fallback to simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def calculate_stylometry(self, sentence: str) -> float:
        """Calculate stylometric features."""
        try:
            features = {}
            
            # Basic features
            features['length'] = len(sentence)
            features['word_count'] = len(sentence.split())
            features['avg_word_length'] = sum(len(word) for word in sentence.split()) / max(len(sentence.split()), 1)
            
            # Character-based features
            features['uppercase_ratio'] = sum(1 for c in sentence if c.isupper()) / max(len(sentence), 1)
            features['digit_ratio'] = sum(1 for c in sentence if c.isdigit()) / max(len(sentence), 1)
            features['punct_ratio'] = sum(1 for c in sentence if not c.isalnum() and not c.isspace()) / max(len(sentence), 1)
            
            # Readability (if textstat available)
            if HAS_TEXTSTAT:
                try:
                    features['flesch_score'] = textstat.flesch_reading_ease(sentence)
                    features['flesch_grade'] = textstat.flesch_kincaid_grade(sentence)
                except:
                    features['flesch_score'] = 50.0  # neutral score
                    features['flesch_grade'] = 10.0
            else:
                features['flesch_score'] = 50.0
                features['flesch_grade'] = 10.0
            
            # Linguistic features (if spaCy available)
            nlp = self._load_spacy_model()
            if nlp:
                try:
                    doc = nlp(sentence)
                    pos_counts = {}
                    for token in doc:
                        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                    
                    total_tokens = len(doc)
                    if total_tokens > 0:
                        features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tokens
                        features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tokens
                        features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tokens
                    else:
                        features['noun_ratio'] = features['verb_ratio'] = features['adj_ratio'] = 0.0
                except:
                    features['noun_ratio'] = features['verb_ratio'] = features['adj_ratio'] = 0.2
            else:
                features['noun_ratio'] = features['verb_ratio'] = features['adj_ratio'] = 0.2
            
            # Normalize and combine features (simple weighted sum)
            stylometry_score = (
                min(features['length'] / 100.0, 1.0) * 0.1 +
                min(features['word_count'] / 20.0, 1.0) * 0.1 +
                min(features['avg_word_length'] / 10.0, 1.0) * 0.1 +
                (1.0 - features['uppercase_ratio']) * 0.1 +  # Prefer normal case
                (1.0 - features['digit_ratio']) * 0.1 +     # Prefer text over numbers
                min(features['flesch_score'] / 100.0, 1.0) * 0.2 +
                features['noun_ratio'] * 0.1 +
                features['verb_ratio'] * 0.1 +
                features['adj_ratio'] * 0.1
            )
            
            return max(0.0, min(1.0, stylometry_score))
            
        except Exception as e:
            logger.warning(f"Stylometry calculation failed: {e}")
            return 0.5  # Neutral score on failure
    
    def analyze_sentence(self, sentence: str) -> AnalysisResult:
        """Analyze a single sentence for plagiarism."""
        # Get initial matches from corpus
        raw_matches = self.corpus_index.search(sentence, k=10)
        
        # Process matches
        matches = []
        for match_text, similarity in raw_matches:
            # Apply cross-encoder reranking if available
            final_score = similarity
            cross_encoder = self._load_cross_encoder()
            if cross_encoder and similarity > 0.3:  # Only rerank promising matches
                try:
                    cross_score = cross_encoder.predict([(sentence, match_text)])[0]
                    # Combine scores (weighted average)
                    final_score = 0.7 * similarity + 0.3 * cross_score
                except Exception as e:
                    logger.debug(f"Cross-encoder failed: {e}")
                    # Keep original score
            
            # Determine confidence level
            if final_score >= 0.8:
                confidence = "HIGH"
            elif final_score >= 0.6:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            matches.append(SimilarityMatch(
                text=match_text,
                similarity=final_score,
                confidence=confidence
            ))
        
        # Calculate stylometry
        stylometry_score = self.calculate_stylometry(sentence)
        
        # Fuse scores (weighted combination)
        if matches:
            max_similarity = max(match.similarity for match in matches)
            fused_score = 0.8 * max_similarity + 0.2 * stylometry_score
        else:
            fused_score = stylometry_score
        
        # Overall confidence
        if fused_score >= 0.8:
            overall_confidence = "HIGH"
        elif fused_score >= 0.6:
            overall_confidence = "MEDIUM"
        else:
            overall_confidence = "LOW"
        
        return AnalysisResult(
            sentence=sentence,
            matches=matches[:5],  # Keep top 5 matches
            stylometry_score=stylometry_score,
            fused_score=fused_score,
            confidence=overall_confidence
        )
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze entire document."""
        sentences = self.extract_sentences(text)
        
        if not sentences:
            return {
                'error': 'No sentences found in document',
                'sentence_analyses': [],
                'overall_stats': {}
            }
        
        # Analyze each sentence
        analyses = []
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short sentences
                analysis = self.analyze_sentence(sentence)
                analyses.append(analysis)
        
        # Calculate overall statistics
        if analyses:
            fused_scores = [a.fused_score for a in analyses]
            high_confidence_count = sum(1 for a in analyses if a.confidence == "HIGH")
            medium_confidence_count = sum(1 for a in analyses if a.confidence == "MEDIUM")
            low_confidence_count = sum(1 for a in analyses if a.confidence == "LOW")
            
            overall_stats = {
                'total_sentences': len(analyses),
                'avg_fused_score': sum(fused_scores) / len(fused_scores),
                'max_fused_score': max(fused_scores),
                'high_confidence_count': high_confidence_count,
                'medium_confidence_count': medium_confidence_count,
                'low_confidence_count': low_confidence_count,
                'high_risk_ratio': high_confidence_count / len(analyses),
            }
        else:
            overall_stats = {
                'total_sentences': 0,
                'avg_fused_score': 0.0,
                'max_fused_score': 0.0,
                'high_confidence_count': 0,
                'medium_confidence_count': 0,
                'low_confidence_count': 0,
                'high_risk_ratio': 0.0,
            }
        
        return {
            'sentence_analyses': [
                {
                    'sentence': a.sentence,
                    'fused_score': a.fused_score,
                    'stylometry_score': a.stylometry_score,
                    'confidence': a.confidence,
                    'matches': [
                        {
                            'text': m.text,
                            'similarity': m.similarity,
                            'confidence': m.confidence
                        } for m in a.matches
                    ]
                } for a in analyses
            ],
            'overall_stats': overall_stats
        }