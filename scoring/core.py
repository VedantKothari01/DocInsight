"""
Core scoring module for DocInsight

Provides sentence-level classification, span clustering, and document-level 
originality scoring with aggregated metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import logging

from config import (
    HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, AGGREGATION_WEIGHTS,
    FUSION_WEIGHTS, FLESCH_SCORE_NORMALIZATION, RISK_LEVELS,
    SEMANTIC_HIGH_FLOOR, SEMANTIC_MEDIUM_FLOOR, SEMANTIC_MIN_MATCH
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
        """Fuse semantic, rerank and stylometry signals with normalization.

        We normalise each score family to 0-1 across the candidate set so that
        no single raw scale (e.g. negative cross-encoder scores) dominates.
        """
        if not semantic_results:
            return []

        candidates = [r['sentence'] for r in semantic_results]
        semantic_scores = np.array([r.get('score', 0.0) for r in semantic_results], dtype=float)

        # Normalise semantic scores (already cosine similarity in [-1,1] after faiss IP w/ L2 norm ~ [0,1])
        sem_min, sem_max = float(np.min(semantic_scores)), float(np.max(semantic_scores))
        if sem_max - sem_min > 1e-6:
            semantic_norm = (semantic_scores - sem_min) / (sem_max - sem_min)
        else:
            semantic_norm = np.zeros_like(semantic_scores)

        # Map rerank scores
        rerank_map = {r['sentence']: r.get('rerank_score', 0.0) for r in (rerank_results or [])}
        rerank_scores = np.array([rerank_map.get(c, 0.0) for c in candidates], dtype=float)
        rer_min, rer_max = float(np.min(rerank_scores)), float(np.max(rerank_scores))
        if rer_max - rer_min > 1e-6:
            rerank_norm = (rerank_scores - rer_min) / (rer_max - rer_min)
        else:
            rerank_norm = np.zeros_like(rerank_scores)

        fused_results: List[Dict[str, Any]] = []
        styl_q = stylometry_features_query.get('flesch_reading_ease', 0.0)

        for i, candidate in enumerate(candidates):
            styl_c = 0.0
            if i < len(stylometry_features_candidates):
                styl_c = stylometry_features_candidates[i].get('flesch_reading_ease', 0.0)
            # Stylometry similarity -> 1 when close, 0 when far
            styl_score = 1.0 - min(abs(styl_q - styl_c) / FLESCH_SCORE_NORMALIZATION, 1.0)

            fused_score = (
                self.alpha * float(semantic_norm[i]) +
                self.beta * float(rerank_norm[i]) +
                self.gamma * float(styl_score)
            )

            fused_results.append({
                'candidate': candidate,
                'semantic_score': float(semantic_scores[i]),
                'semantic_norm': float(semantic_norm[i]),
                'rerank_score': float(rerank_scores[i]),
                'rerank_norm': float(rerank_norm[i]),
                'stylometry_score': float(styl_score),
                'fused_score': float(fused_score),
                'components': {
                    'semantic': float(semantic_norm[i]),
                    'cross_encoder': float(rerank_norm[i]),
                    'stylometry': float(styl_score)
                }
            })

        # Sort by fused score descending
        fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
        return fused_results
    
    def classify_sentence(self, fused_results: List[Dict]) -> Tuple[str, float, str, str]:
        """Classify risk using fused + semantic floors and return explanation.

        Returns: (risk_level, fused_score, match_strength, reason)
        """
        if not fused_results:
            return RISK_LEVELS['LOW'], 0.0, 'NONE', 'No candidates returned'

        top = fused_results[0]
        fused_score = top['fused_score']
        sem_norm = top.get('semantic_norm', 0.0)
        sem_raw = top.get('semantic_score', 0.0)

        # Match strength label (for UI)
        if sem_norm >= 0.75:
            match_strength = 'STRONG'
        elif sem_norm >= 0.55:
            match_strength = 'MODERATE'
        elif sem_norm >= 0.40:
            match_strength = 'WEAK'
        else:
            match_strength = 'VERY_WEAK'

        # Semantic raw floor guard
        if sem_raw < SEMANTIC_MIN_MATCH:
            return RISK_LEVELS['LOW'], fused_score, match_strength, (
                f"Semantic raw {sem_raw:.3f} < min {SEMANTIC_MIN_MATCH}")

        # High risk gating
        if fused_score >= HIGH_RISK_THRESHOLD and sem_norm >= SEMANTIC_HIGH_FLOOR:
            return RISK_LEVELS['HIGH'], fused_score, match_strength, (
                f"Fused {fused_score:.3f}>=HIGH({HIGH_RISK_THRESHOLD}) & semantic_norm {sem_norm:.3f}>=floor {SEMANTIC_HIGH_FLOOR}" )

        # Medium risk gating
        if fused_score >= MEDIUM_RISK_THRESHOLD and sem_norm >= SEMANTIC_MEDIUM_FLOOR:
            return RISK_LEVELS['MEDIUM'], fused_score, match_strength, (
                f"Fused {fused_score:.3f}>=MED({MEDIUM_RISK_THRESHOLD}) & semantic_norm {sem_norm:.3f}>=floor {SEMANTIC_MEDIUM_FLOOR}")

        return RISK_LEVELS['LOW'], fused_score, match_strength, "Below medium thresholds"


class SpanClusterer:
    """Clusters consecutive high/medium risk sentences into spans"""
    
    def cluster_risk_spans(self, sentence_results: List[Dict]) -> List[Dict]:
        spans = []
        current_span = None
        for i, result in enumerate(sentence_results):
            risk_level = result.get('risk_level', RISK_LEVELS['LOW'])
            if risk_level in [RISK_LEVELS['HIGH'], RISK_LEVELS['MEDIUM']]:
                if (current_span is None or 
                    current_span['risk_level'] != risk_level or 
                    i - current_span['end_index'] > 1):
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
                    current_span['end_index'] = i
                    current_span['sentences'].append(result)
                    scores = [s.get('confidence_score', 0.0) for s in current_span['sentences']]
                    current_span['avg_score'] = np.mean(scores)
                    current_span['token_count'] += len(result.get('sentence', '').split())
            else:
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
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
        plagiarized_tokens = sum(span['token_count'] for span in risk_spans)
        plagiarized_coverage = plagiarized_tokens / total_tokens if total_tokens > 0 else 0.0
        if risk_spans:
            severity_scores = []
            for span in risk_spans:
                weight = span['token_count']
                severity_scores.extend([span['avg_score']] * weight)
            severity_index = np.mean(severity_scores) if severity_scores else 0.0
        else:
            severity_index = 0.0
        risk_span_ratio = len(risk_spans) / total_sentences if total_sentences > 0 else 0.0
        plagiarism_factor = (self.alpha * plagiarized_coverage + 
                           self.beta * severity_index + 
                           self.gamma * risk_span_ratio)
        originality_score = max(0.0, 1.0 - plagiarism_factor)
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
        top_spans = sorted(risk_spans, key=lambda x: x['avg_score'], reverse=True)[:top_n]
        for span in top_spans:
            if span['sentences']:
                first_sentence = span['sentences'][0].get('sentence', '')
                span['preview_text'] = first_sentence[:100] + '...' if len(first_sentence) > 100 else first_sentence
            else:
                span['preview_text'] = ''
        return top_spans

def analyze_document_originality(sentence_results: List[Dict]) -> Dict[str, Any]:
    """Full document originality analysis including span filtering and component breakdown."""
    # Cluster risk spans
    clusterer = SpanClusterer()
    risk_spans = clusterer.cluster_risk_spans(sentence_results)

    # Deduplicate spans that may have identical start/end indices or identical sentence id sets
    # (Can occur after future post-processing adjustments that alter risk levels.)
    unique = []
    seen_keys = set()
    for span in risk_spans:
        key = (span.get('start_index'), span.get('end_index'), span.get('risk_level'))
        # Fallback: hash of concatenated sentence texts if indices missing
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(span)
    if len(unique) != len(risk_spans):
        logger.debug(f"Deduplicated {len(risk_spans)-len(unique)} duplicate spans")
    risk_spans = unique

    # Filter out weak single-sentence spans (reduce noise)
    filtered_spans: List[Dict[str, Any]] = []
    for span in risk_spans:
        if span['start_index'] == span['end_index']:
            sent = span['sentences'][0] if span['sentences'] else {}
            sem_norm = sent.get('semantic_norm', 0.0)
            if sem_norm < 0.55:  # heuristic suppression threshold
                continue
        filtered_spans.append(span)
    if len(filtered_spans) != len(risk_spans):
        logger.debug(f"Filtered {len(risk_spans)-len(filtered_spans)} weak spans")
    risk_spans = filtered_spans

    scorer = DocumentScorer()
    originality_metrics = scorer.compute_originality_score(sentence_results, risk_spans)

    # Add component explanation for originality score
    alpha = scorer.alpha; beta = scorer.beta; gamma = scorer.gamma
    plagiarism_factor = (
        alpha * originality_metrics.get('plagiarized_coverage', 0.0) +
        beta * originality_metrics.get('severity_index', 0.0) +
        gamma * originality_metrics.get('risk_span_ratio', 0.0)
    )
    originality_metrics['plagiarism_factor'] = float(plagiarism_factor)
    originality_metrics['plagiarism_components'] = {
        'coverage_component': float(alpha * originality_metrics.get('plagiarized_coverage', 0.0)),
        'severity_component': float(beta * originality_metrics.get('severity_index', 0.0)),
        'span_ratio_component': float(gamma * originality_metrics.get('risk_span_ratio', 0.0)),
        'weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma}
    }

    top_risk_spans = scorer.get_top_risk_spans(risk_spans)
    return {
        'originality_metrics': originality_metrics,
        'risk_spans': risk_spans,
        'top_risk_spans': top_risk_spans,
        'total_risk_spans': len(risk_spans)
    }
