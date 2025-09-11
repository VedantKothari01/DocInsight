"""
Scoring module for DocInsight

Provides sentence-level classification, span clustering, and document-level 
originality scoring with aggregated metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import logging

from config import (
    HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, AGGREGATION_WEIGHTS,
    FUSION_WEIGHTS, FLESCH_SCORE_NORMALIZATION, RISK_LEVELS
)

logger = logging.getLogger(__name__)


class SentenceClassifier:
    """Classifies sentences based on fused similarity scores"""
    
    def __init__(self):
        self.alpha = FUSION_WEIGHTS['semantic']
        self.beta = FUSION_WEIGHTS['cross_encoder']
        self.gamma = FUSION_WEIGHTS['stylometry']
    
    def compute_fused_score(self, query_sent: str, semantic_results: List[Dict], 
                           rerank_results: List[Dict], stylometry_features_query: Dict,
                           stylometry_features_candidates: List[Dict]) -> List[Dict]:
        """
        Compute fused scores from semantic, cross-encoder, and stylometry signals
        
        Args:
            query_sent: The query sentence
            semantic_results: List of {'sentence': str, 'score': float}
            rerank_results: List of {'sentence': str, 'rerank_score': float}
            stylometry_features_query: Stylometry features for query sentence
            stylometry_features_candidates: List of stylometry features for candidates
            
        Returns:
            List of scored candidates with fused scores
        """
        candidates = [r['sentence'] for r in semantic_results]
        semantic_scores = [r['score'] for r in semantic_results]
        
        # Create rerank score mapping
        rerank_map = {r['sentence']: r['rerank_score'] for r in rerank_results}
        
        # Get min rerank score for normalization
        rerank_scores_list = [r['rerank_score'] for r in rerank_results] if rerank_results else [0.0]
        rer_min = min(rerank_scores_list) if rerank_scores_list else 0.0
        
        fused_results = []
        
        for i, candidate in enumerate(candidates):
            sem_score = semantic_scores[i]
            rer_score = rerank_map.get(candidate, 0.0)
            
            # Stylometry similarity based on Flesch reading ease
            styl_q = stylometry_features_query.get('flesch_reading_ease', 0.0)
            styl_c = stylometry_features_candidates[i].get('flesch_reading_ease', 0.0) if i < len(stylometry_features_candidates) else 0.0
            styl_score = 1.0 - min(abs(styl_q - styl_c) / FLESCH_SCORE_NORMALIZATION, 1.0)
            
            # Compute fused score
            fused_score = (self.alpha * sem_score + 
                          self.beta * (rer_score - rer_min) + 
                          self.gamma * styl_score)
            
            fused_results.append({
                'candidate': candidate,
                'semantic_score': float(sem_score),
                'rerank_score': float(rer_score),
                'stylometry_score': float(styl_score),
                'fused_score': float(fused_score)
            })
        
        return sorted(fused_results, key=lambda x: x['fused_score'], reverse=True)
    
    def classify_sentence(self, fused_results: List[Dict]) -> Tuple[str, float]:
        """
        Classify sentence risk level based on top fused score
        
        Returns:
            Tuple of (risk_level, confidence_score)
        """
        if not fused_results:
            return RISK_LEVELS['LOW'], 0.0
            
        best_score = fused_results[0]['fused_score']
        
        if best_score >= HIGH_RISK_THRESHOLD:
            return RISK_LEVELS['HIGH'], best_score
        elif best_score >= MEDIUM_RISK_THRESHOLD:
            return RISK_LEVELS['MEDIUM'], best_score
        else:
            return RISK_LEVELS['LOW'], best_score


class SpanClusterer:
    """Clusters consecutive high/medium risk sentences into spans"""
    
    def cluster_risk_spans(self, sentence_results: List[Dict]) -> List[Dict]:
        """
        Cluster consecutive sentences with similar risk levels into spans
        
        Args:
            sentence_results: List of sentence analysis results
            
        Returns:
            List of risk spans with metadata
        """
        spans = []
        current_span = None
        
        for i, result in enumerate(sentence_results):
            risk_level = result.get('risk_level', RISK_LEVELS['LOW'])
            
            # Only cluster HIGH and MEDIUM risk sentences
            if risk_level in [RISK_LEVELS['HIGH'], RISK_LEVELS['MEDIUM']]:
                if (current_span is None or 
                    current_span['risk_level'] != risk_level or 
                    i - current_span['end_index'] > 1):
                    
                    # Start new span
                    if current_span is not None:
                        spans.append(current_span)
                    
                    current_span = {
                        'start_index': i,
                        'end_index': i,
                        'risk_level': risk_level,
                        'sentences': [result],
                        'avg_score': result.get('confidence_score', 0.0),
                        'token_count': len(result.get('sentence', '').split())
                    }
                else:
                    # Extend current span
                    current_span['end_index'] = i
                    current_span['sentences'].append(result)
                    
                    # Update average score
                    scores = [s.get('confidence_score', 0.0) for s in current_span['sentences']]
                    current_span['avg_score'] = np.mean(scores)
                    current_span['token_count'] += len(result.get('sentence', '').split())
            else:
                # End current span if it exists
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
        
        # Don't forget the last span
        if current_span is not None:
            spans.append(current_span)
        
        return sorted(spans, key=lambda x: x['avg_score'], reverse=True)


class DocumentScorer:
    """Computes document-level aggregated originality metrics"""
    
    def __init__(self):
        self.alpha = AGGREGATION_WEIGHTS['alpha']  # Coverage weight
        self.beta = AGGREGATION_WEIGHTS['beta']    # Severity weight  
        self.gamma = AGGREGATION_WEIGHTS['gamma']  # Span ratio weight
    
    def compute_originality_score(self, sentence_results: List[Dict], 
                                 risk_spans: List[Dict]) -> Dict[str, Any]:
        """
        Compute document-level originality metrics
        
        Args:
            sentence_results: List of sentence analysis results
            risk_spans: List of clustered risk spans
            
        Returns:
            Dictionary with originality metrics
        """
        total_sentences = len(sentence_results)
        total_tokens = sum(len(r.get('sentence', '').split()) for r in sentence_results)
        
        if total_sentences == 0:
            return {
                'originality_score': 1.0,
                'plagiarized_coverage': 0.0,
                'severity_index': 0.0,
                'risk_span_ratio': 0.0,
                'sentence_distribution': {
                    RISK_LEVELS['HIGH']: 0,
                    RISK_LEVELS['MEDIUM']: 0,
                    RISK_LEVELS['LOW']: total_sentences
                }
            }
        
        # Calculate coverage (token-weighted)
        plagiarized_tokens = sum(span['token_count'] for span in risk_spans)
        plagiarized_coverage = plagiarized_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Calculate severity index (normalized average of span severities)
        if risk_spans:
            severity_scores = []
            for span in risk_spans:
                # Weight severity by span length
                weight = span['token_count']
                severity_scores.extend([span['avg_score']] * weight)
            severity_index = np.mean(severity_scores) if severity_scores else 0.0
        else:
            severity_index = 0.0
        
        # Calculate span ratio
        risk_span_ratio = len(risk_spans) / total_sentences if total_sentences > 0 else 0.0
        
        # Compute originality score using weighted formula
        # Originality = 1 - f(coverage, severity, span_ratio)
        plagiarism_factor = (self.alpha * plagiarized_coverage + 
                           self.beta * severity_index + 
                           self.gamma * risk_span_ratio)
        
        originality_score = max(0.0, 1.0 - plagiarism_factor)
        
        # Calculate sentence distribution
        distribution = defaultdict(int)
        for result in sentence_results:
            risk_level = result.get('risk_level', RISK_LEVELS['LOW'])
            distribution[risk_level] += 1
        
        return {
            'originality_score': float(originality_score),
            'plagiarized_coverage': float(plagiarized_coverage),
            'severity_index': float(severity_index),
            'risk_span_ratio': float(risk_span_ratio),
            'sentence_distribution': dict(distribution),
            'total_sentences': total_sentences,
            'total_tokens': total_tokens
        }
    
    def get_top_risk_spans(self, risk_spans: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Get top N risk spans for preview
        
        Args:
            risk_spans: List of risk spans
            top_n: Number of top spans to return
            
        Returns:
            List of top risk spans with preview info
        """
        top_spans = sorted(risk_spans, key=lambda x: x['avg_score'], reverse=True)[:top_n]
        
        for span in top_spans:
            # Add preview text (first sentence)
            if span['sentences']:
                first_sentence = span['sentences'][0].get('sentence', '')
                span['preview_text'] = first_sentence[:100] + '...' if len(first_sentence) > 100 else first_sentence
            else:
                span['preview_text'] = ''
        
        return top_spans


def analyze_document_originality(sentence_results: List[Dict]) -> Dict[str, Any]:
    """
    Main function to analyze document originality with all metrics
    
    Args:
        sentence_results: List of sentence analysis results with risk classifications
        
    Returns:
        Complete originality analysis with metrics and spans
    """
    # Cluster risk spans
    clusterer = SpanClusterer()
    risk_spans = clusterer.cluster_risk_spans(sentence_results)
    
    # Compute document-level metrics
    scorer = DocumentScorer()
    originality_metrics = scorer.compute_originality_score(sentence_results, risk_spans)
    
    # Get top risk spans for preview
    top_risk_spans = scorer.get_top_risk_spans(risk_spans)
    
    return {
        'originality_metrics': originality_metrics,
        'risk_spans': risk_spans,
        'top_risk_spans': top_risk_spans,
        'total_risk_spans': len(risk_spans)
    }