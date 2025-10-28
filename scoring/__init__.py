"""
Scoring module for DocInsight

Provides unified document-level scoring and aggregation capabilities.
"""

from scoring.aggregate import UnifiedDocumentScorer, compute_unified_score
from .core import SentenceClassifier, SpanClusterer, DocumentScorer, analyze_document_originality