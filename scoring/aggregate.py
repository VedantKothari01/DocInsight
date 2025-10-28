"""
Unified scoring and aggregation module for DocInsight

Provides document-level scoring combining semantic similarity, stylometric analysis,
and AI-likeness detection with configurable weighting and suspicious section detection.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from config import (
    WEIGHT_SEMANTIC, WEIGHT_STYLO, WEIGHT_AI, SUSPICIOUS_SECTION_COUNT,
    AI_LIKENESS_THRESHOLD, MODEL_FINE_TUNED_PATH, AI_LIKENESS_MODEL_PATH
)

logger = logging.getLogger(__name__)


@dataclass
class SectionScore:
    """Represents scoring for a document section"""
    section_name: str
    semantic_risk: float
    stylometric_deviation: float
    ai_likeness_prob: float
    overall_score: float
    token_count: int
    reason: str


class UnifiedDocumentScorer:
    """Unified scorer for document-level originality analysis"""
    
    def __init__(self):
        """Initialize unified scorer with configured weights"""
        self.weights = {
            'semantic': WEIGHT_SEMANTIC,
            'stylometry': WEIGHT_STYLO,
            'ai_likeness': WEIGHT_AI
        }
        
        # Verify weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights do not sum to 1.0 (sum={total_weight}), normalizing...")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"Initialized scorer with weights: {self.weights}")
    
    def compute_document_summary(self, 
                                semantic_results: List[Dict[str, Any]], 
                                stylometric_features: Dict[str, float],
                                sections: Optional[List[Any]] = None,
                                ai_likeness_prob: Optional[float] = None) -> Dict[str, Any]:
        """Compute unified document-level scoring summary
        
        Args:
            semantic_results: Sentence-level semantic analysis results
            stylometric_features: Document-level stylometric features
            sections: Optional document sections for section-level analysis
            ai_likeness_prob: Optional AI-likeness probability (0-1)
            
        Returns:
            Document summary with unified scoring
        """
        # Compute semantic risk score
        semantic_risk = self._compute_semantic_risk(semantic_results)
        
        # Compute stylometric deviation score
        stylometric_deviation = self._compute_stylometric_deviation(stylometric_features)
        
        # Handle missing AI-likeness score
        if ai_likeness_prob is None:
            logger.warning("AI-likeness model not available, adjusting weights")
            ai_likeness_prob = 0.0
            # Renormalize weights without AI component
            total_weight = self.weights['semantic'] + self.weights['stylometry']
            adjusted_weights = {
                'semantic': self.weights['semantic'] / total_weight,
                'stylometry': self.weights['stylometry'] / total_weight,
                'ai_likeness': 0.0
            }
        else:
            adjusted_weights = self.weights.copy()
        
        # Compute overall weighted score
        overall_score = (
            adjusted_weights['semantic'] * semantic_risk +
            adjusted_weights['stylometry'] * stylometric_deviation +
            adjusted_weights['ai_likeness'] * ai_likeness_prob
        )
        
        # Detect suspicious sections
        suspicious_sections = []
        if sections:
            suspicious_sections = self._detect_suspicious_sections(
                sections, semantic_results, stylometric_features, ai_likeness_prob
            )
        
        return {
            'overall_score': float(overall_score),
            'weights': adjusted_weights,
            'semantic_risk': float(semantic_risk),
            'stylometric_deviation': float(stylometric_deviation),
            'ai_likeness_prob': float(ai_likeness_prob) if ai_likeness_prob is not None else None,
            'suspicious_sections': suspicious_sections,
            'interpretation': self._interpret_score(overall_score),
            'confidence': self._compute_confidence(semantic_results, stylometric_features, ai_likeness_prob)
        }
    
    def _compute_semantic_risk(self, semantic_results: List[Dict[str, Any]]) -> float:
        """Compute semantic similarity risk score (0-1, higher = more risk)"""
        if not semantic_results:
            return 0.0
        
        # Extract similarity scores and risk classifications
        high_risk_count = 0
        medium_risk_count = 0
        total_score = 0.0
        
        for result in semantic_results:
            score = result.get('similarity_score', 0.0)
            risk_level = result.get('risk_level', 'LOW')
            
            total_score += score
            
            if risk_level == 'HIGH':
                high_risk_count += 1
            elif risk_level == 'MEDIUM':
                medium_risk_count += 1
        
        # Weighted risk calculation
        total_sentences = len(semantic_results)
        avg_similarity = total_score / total_sentences
        high_risk_ratio = high_risk_count / total_sentences
        medium_risk_ratio = medium_risk_count / total_sentences
        
        # Combined risk score (0-1)
        semantic_risk = (
            0.5 * avg_similarity +  # Average similarity contributes 50%
            0.3 * high_risk_ratio +  # High risk ratio contributes 30%
            0.2 * medium_risk_ratio  # Medium risk ratio contributes 20%
        )
        
        return min(semantic_risk, 1.0)
    
    def _compute_stylometric_deviation(self, features: Dict[str, float]) -> float:
        """Compute stylometric deviation score (0-1, higher = more deviation)"""
        if not features:
            return 0.0
        
        # Baseline expectations for typical academic writing
        baselines = {
            'type_token_ratio': 0.5,  # Expected lexical diversity
            'avg_sentence_length': 20.0,  # Expected sentence length
            'avg_word_length': 5.0,  # Expected word length
            'punctuation_density': 0.1,  # Expected punctuation usage
            'stopword_ratio': 0.4,  # Expected function word usage
            'complexity_score': 0.6  # Expected complexity
        }
        
        deviations = []
        
        for feature, baseline in baselines.items():
            if feature in features:
                actual = features[feature]
                # Normalize deviation by baseline to handle different scales
                if baseline > 0:
                    deviation = abs(actual - baseline) / baseline
                else:
                    deviation = abs(actual)
                deviations.append(min(deviation, 2.0))  # Cap at 200% deviation
        
        if not deviations:
            return 0.0
        
        # Average deviation with sigmoid normalization
        avg_deviation = np.mean(deviations)
        normalized_deviation = 2 / (1 + np.exp(-avg_deviation)) - 1  
        
        return float(normalized_deviation)
    
    def _detect_suspicious_sections(self, 
                                   sections: List[Any], 
                                   semantic_results: List[Dict[str, Any]],
                                   stylometric_features: Dict[str, float],
                                   ai_likeness_prob: Optional[float]) -> List[Dict[str, Any]]:
        """Detect top suspicious sections based on combined scoring"""
        if not sections:
            return []
        
        section_scores = []
        
        for section in sections:
            section_name = getattr(section, 'title', getattr(section, 'section_type', 'Unknown'))
            section_content = getattr(section, 'content', '')
            token_count = getattr(section, 'token_count', len(section_content.split()))
            
            # Compute section-level scores
            section_semantic = self._compute_section_semantic_risk(section_content, semantic_results)
            section_stylometric = self._compute_section_stylometric_deviation(section_content)
            section_ai_prob = ai_likeness_prob or 0.0  # Use document-level for now
            
            # Combined section score
            section_overall = (
                self.weights['semantic'] * section_semantic +
                self.weights['stylometry'] * section_stylometric +
                self.weights['ai_likeness'] * section_ai_prob
            )
            
            # Determine reason for suspicion
            reason = self._determine_suspicion_reason(
                section_semantic, section_stylometric, section_ai_prob
            )
            
            section_scores.append(SectionScore(
                section_name=section_name,
                semantic_risk=section_semantic,
                stylometric_deviation=section_stylometric,
                ai_likeness_prob=section_ai_prob,
                overall_score=section_overall,
                token_count=token_count,
                reason=reason
            ))
        
        # Sort by overall score (descending) and return top N
        section_scores.sort(key=lambda x: x.overall_score, reverse=True)
        top_sections = section_scores[:SUSPICIOUS_SECTION_COUNT]
        
        # Convert to dictionaries for JSON serialization
        return [
            {
                'section': section.section_name,
                'score_breakdown': {
                    'semantic_risk': section.semantic_risk,
                    'stylometric_deviation': section.stylometric_deviation,
                    'ai_likeness_prob': section.ai_likeness_prob,
                    'overall_score': section.overall_score
                },
                'reason': section.reason,
                'token_count': section.token_count
            }
            for section in top_sections
            if section.overall_score > 0.3  # Only include moderately suspicious sections
        ]
    
    def _compute_section_semantic_risk(self, section_content: str, semantic_results: List[Dict[str, Any]]) -> float:
        """Compute semantic risk for a specific section"""
        # This is a simplified approach - in practice, you'd want to match
        # semantic results to specific sections based on text positions
        if not semantic_results:
            return 0.0
        
        # For now, use document-level semantic risk as approximation
        return self._compute_semantic_risk(semantic_results)
    
    def _compute_section_stylometric_deviation(self, section_content: str) -> float:
        """Compute stylometric deviation for a specific section"""
        from stylometry.features import extract_stylometric_features
        
        if not section_content.strip():
            return 0.0
        
        section_features = extract_stylometric_features(section_content)
        return self._compute_stylometric_deviation(section_features)
    
    def _determine_suspicion_reason(self, semantic_risk: float, stylometric_dev: float, ai_prob: float) -> str:
        """Determine primary reason for section suspicion"""
        scores = [
            (semantic_risk, "High semantic similarity"),
            (stylometric_dev, "Stylometric deviation"),
            (ai_prob, "AI-likeness indicators")
        ]
        
        # Find the highest contributing factor
        max_score, reason = max(scores, key=lambda x: x[0])
        
        if max_score < 0.3:
            return "Low overall risk"
        elif semantic_risk > 0.6 and ai_prob > 0.6:
            return "High AI-likeness & paraphrase density"
        elif semantic_risk > 0.6:
            return "High semantic similarity to known sources"
        elif stylometric_dev > 0.6:
            return "Significant writing style deviation"
        elif ai_prob > 0.6:
            return "High probability of AI-generated content"
        else:
            return reason
    
    def _interpret_score(self, overall_score: float) -> str:
        """Provide human-readable interpretation of overall score"""
        if overall_score >= 0.8:
            return "Very High Risk - Strong indicators of non-original content"
        elif overall_score >= 0.6:
            return "High Risk - Multiple originality concerns detected"
        elif overall_score >= 0.4:
            return "Medium Risk - Some originality concerns identified"
        elif overall_score >= 0.2:
            return "Low Risk - Minor originality concerns"
        else:
            return "Very Low Risk - Content appears original"
    
    def _compute_confidence(self, semantic_results: List[Dict[str, Any]], 
                           stylometric_features: Dict[str, float],
                           ai_likeness_prob: Optional[float]) -> float:
        """Compute confidence in the scoring (0-1)"""
        confidence_factors = []
        
        # Semantic analysis confidence
        if semantic_results:
            # Higher confidence with more sentences analyzed
            semantic_confidence = min(len(semantic_results) / 50.0, 1.0)
            confidence_factors.append(semantic_confidence)
        
        # Stylometric confidence
        if stylometric_features:
            # Higher confidence with longer text (more tokens)
            token_count = stylometric_features.get('token_count', 0)
            stylometric_confidence = min(token_count / 500.0, 1.0)
            confidence_factors.append(stylometric_confidence)
        
        # AI-likeness confidence
        if ai_likeness_prob is not None:
            confidence_factors.append(0.8)  # Moderate confidence in AI detection
        
        if not confidence_factors:
            return 0.0
        
        return float(np.mean(confidence_factors))


def compute_unified_score(semantic_results: List[Dict[str, Any]], 
                         stylometric_features: Dict[str, float],
                         sections: Optional[List[Any]] = None,
                         ai_likeness_prob: Optional[float] = None) -> Dict[str, Any]:
    """Convenience function for unified document scoring
    
    Args:
        semantic_results: Sentence-level semantic analysis results
        stylometric_features: Document-level stylometric features  
        sections: Optional document sections
        ai_likeness_prob: Optional AI-likeness probability
        
    Returns:
        Document summary with unified scoring
    """
    scorer = UnifiedDocumentScorer()
    return scorer.compute_document_summary(
        semantic_results, stylometric_features, sections, ai_likeness_prob
    )


# Example usage for testing
if __name__ == "__main__":
    # Sample data for testing
    sample_semantic_results = [
        {'similarity_score': 0.8, 'risk_level': 'HIGH'},
        {'similarity_score': 0.6, 'risk_level': 'MEDIUM'},
        {'similarity_score': 0.3, 'risk_level': 'LOW'},
    ]
    
    sample_stylometric_features = {
        'token_count': 100,
        'type_token_ratio': 0.7,
        'avg_sentence_length': 25.0,
        'avg_word_length': 6.0,
        'punctuation_density': 0.15,
        'stopword_ratio': 0.35,
        'complexity_score': 0.8
    }
    
    summary = compute_unified_score(
        sample_semantic_results, 
        sample_stylometric_features,
        ai_likeness_prob=0.4
    )
    
    print("Document Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")