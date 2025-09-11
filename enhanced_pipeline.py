"""
Enhanced pipeline for DocInsight

Cleaned, simplified, and integrated pipeline with scoring aggregation.
Prepared for future DB integration.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import html

# Third-party imports
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
import textstat
import nltk
from nltk.tokenize import sent_tokenize
import docx2txt
import fitz  # PyMuPDF

# Local imports
from config import *
from scoring import SentenceClassifier, analyze_document_originality

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class TextExtractor:
    """Handles text extraction from various file formats"""
    
    @staticmethod
    def extract_text_from_pdf(path: str) -> str:
        """Extract text from PDF file"""
        try:
            text_parts = []
            doc = fitz.open(path)
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {path}: {e}")
            raise
    
    @staticmethod
    def extract_text(path: str) -> str:
        """Extract text from supported file formats"""
        path = str(path)
        ext = Path(path).suffix.lower()
        
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f'Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}')
        
        try:
            if ext == '.pdf':
                return TextExtractor.extract_text_from_pdf(path)
            elif ext == '.docx':
                return docx2txt.process(path)
            elif ext == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error extracting text from {path}: {e}")
            raise


class SentenceProcessor:
    """Handles sentence tokenization and preprocessing"""
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            sentences = sent_tokenize(text)
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > MIN_SENTENCE_LENGTH]
            return sentences
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            return []


class StylemetryAnalyzer:
    """Handles stylometry feature extraction"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
            logger.info(f"Loaded spaCy model: {SPACY_MODEL_NAME}")
        except OSError:
            logger.warning(f"spaCy model {SPACY_MODEL_NAME} not found. Stylometry features will be limited.")
            self.nlp = None
    
    def extract_features(self, sentence: str) -> Dict[str, float]:
        """Extract stylometry features from sentence"""
        if self.nlp is None:
            return {'flesch_reading_ease': 0.0}
        
        try:
            doc = self.nlp(sentence)
            features = {}
            
            # Token-based features
            alpha_tokens = [t for t in doc if t.is_alpha]
            features['num_tokens'] = len(alpha_tokens)
            features['avg_word_len'] = (sum(len(t.text) for t in alpha_tokens) / 
                                      max(1, len(alpha_tokens)))
            
            # Readability
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(sentence)
            
            # Lexical diversity
            words = [t.text.lower() for t in alpha_tokens]
            features['ttr'] = len(set(words)) / max(1, len(words))
            
            # Punctuation density
            features['punct_density'] = len([t for t in doc if t.is_punct]) / max(1, len(doc))
            
            # POS ratios
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            
            features['noun_ratio'] = pos_counts.get('NOUN', 0) / max(1, len(doc))
            features['verb_ratio'] = pos_counts.get('VERB', 0) / max(1, len(doc))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting stylometry features: {e}")
            return {'flesch_reading_ease': 0.0}


class SemanticSearchEngine:
    """Handles semantic search using SBERT and FAISS"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.corpus_sentences = []
        self._load_models()
    
    def _load_models(self):
        """Load SBERT model"""
        try:
            logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}")
            self.model = SentenceTransformer(SBERT_MODEL_NAME)
            logger.info("SBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SBERT model: {e}")
            self.model = None
    
    def build_index(self, corpus_sentences: List[str]):
        """Build FAISS index from corpus sentences"""
        if self.model is None:
            logger.error("SBERT model not loaded. Cannot build index.")
            return False
        
        try:
            logger.info(f"Building FAISS index for {len(corpus_sentences)} sentences")
            self.corpus_sentences = corpus_sentences
            
            # Encode corpus
            corpus_embeddings = self.model.encode(corpus_sentences, convert_to_numpy=True)
            faiss.normalize_L2(corpus_embeddings)
            
            # Build index
            d = corpus_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(corpus_embeddings)
            
            logger.info(f"FAISS index built with {self.index.ntotal} sentences")
            return True
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False
    
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Perform semantic search"""
        if self.model is None or self.index is None:
            logger.warning("Search engine not properly initialized")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.corpus_sentences):
                    results.append({
                        'sentence': self.corpus_sentences[idx],
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []


class CrossEncoderReranker:
    """Handles cross-encoder reranking for higher precision"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model with resilient failure handling"""
        try:
            logger.info(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL_NAME}")
            self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder model: {e}. Reranking will be disabled.")
            self.model = None
    
    def rerank(self, query: str, candidates: List[str]) -> List[Dict]:
        """Rerank candidates using cross-encoder"""
        if self.model is None:
            logger.warning("Cross-encoder not available. Returning candidates with default scores.")
            return [{'sentence': s, 'rerank_score': 0.0} for s in candidates]
        
        if not candidates:
            return []
        
        try:
            pairs = [[query, candidate] for candidate in candidates]
            scores = self.model.predict(pairs)
            
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [{'sentence': s, 'rerank_score': float(score)} for s, score in ranked]
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return [{'sentence': s, 'rerank_score': 0.0} for s in candidates]


class DocumentAnalysisPipeline:
    """Main pipeline for document analysis"""
    
    def __init__(self, corpus_sentences: Optional[List[str]] = None):
        self.text_extractor = TextExtractor()
        self.sentence_processor = SentenceProcessor()
        self.stylometry_analyzer = StylemetryAnalyzer()
        self.semantic_engine = SemanticSearchEngine()
        self.reranker = CrossEncoderReranker()
        self.sentence_classifier = SentenceClassifier()
        
        # Use demo corpus if none provided
        if corpus_sentences is None:
            corpus_sentences = self._get_demo_corpus()
        
        # Build search index
        if corpus_sentences:
            self.semantic_engine.build_index(corpus_sentences)
    
    def _get_demo_corpus(self) -> List[str]:
        """Get demo corpus for testing"""
        return [
            "Climate change is a critical global issue that affects agriculture and health.",
            "The effects of global warming include rising sea levels and more extreme weather.",
            "Machine learning improves many real world tasks such as image recognition and language modeling.",
            "Neural networks can approximate complex functions and are widely used in deep learning.",
            "The French Revolution began in 1789 and led to major political changes in Europe.",
            "Photosynthesis is the process by which green plants convert sunlight into energy.",
            "The mitochondrion is the powerhouse of the cell.",
            "In 1969, Neil Armstrong became the first person to walk on the Moon.",
            "The capital of France is Paris.",
            "SQL stands for Structured Query Language and is used to manage relational databases."
        ]
    
    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """Analyze a single sentence for similarity and risk"""
        try:
            # Semantic search
            semantic_results = self.semantic_engine.search(sentence, top_k=DEFAULT_TOP_K)
            
            if not semantic_results:
                return self._empty_sentence_result(sentence)
            
            # Extract candidates for reranking
            candidates = [r['sentence'] for r in semantic_results]
            
            # Cross-encoder reranking
            rerank_results = self.reranker.rerank(sentence, candidates)
            
            # Stylometry analysis
            query_features = self.stylometry_analyzer.extract_features(sentence)
            candidate_features = [self.stylometry_analyzer.extract_features(c) for c in candidates]
            
            # Compute fused scores
            fused_results = self.sentence_classifier.compute_fused_score(
                sentence, semantic_results, rerank_results, 
                query_features, candidate_features
            )
            
            # Classify risk level
            risk_level, confidence_score = self.sentence_classifier.classify_sentence(fused_results)
            
            # Get best match
            best_match = fused_results[0] if fused_results else {}
            
            return {
                'sentence': sentence,
                'risk_level': risk_level,
                'confidence_score': confidence_score,
                'best_match': best_match.get('candidate', ''),
                'semantic_score': best_match.get('semantic_score', 0.0),
                'rerank_score': best_match.get('rerank_score', 0.0),
                'stylometry_score': best_match.get('stylometry_score', 0.0),
                'fused_score': best_match.get('fused_score', 0.0),
                'stylometry_features': query_features,
                'all_candidates': fused_results[:MAX_CANDIDATES]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentence: {e}")
            return self._empty_sentence_result(sentence)
    
    def _empty_sentence_result(self, sentence: str) -> Dict[str, Any]:
        """Return empty result for failed sentence analysis"""
        return {
            'sentence': sentence,
            'risk_level': RISK_LEVELS['LOW'],
            'confidence_score': 0.0,
            'best_match': '',
            'semantic_score': 0.0,
            'rerank_score': 0.0,
            'stylometry_score': 0.0,
            'fused_score': 0.0,
            'stylometry_features': {},
            'all_candidates': []
        }
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Analyze complete document for originality"""
        try:
            logger.info(f"Analyzing document: {file_path}")
            
            # Extract text
            text = self.text_extractor.extract_text(file_path)
            if not text.strip():
                raise ValueError("Empty document")
            
            # Split into sentences
            sentences = self.sentence_processor.split_sentences(text)
            if not sentences:
                raise ValueError("No sentences found in document")
            
            logger.info(f"Found {len(sentences)} sentences to analyze")
            
            # Analyze each sentence
            sentence_results = []
            for i, sentence in enumerate(sentences):
                if i % 10 == 0:  # Log progress
                    logger.info(f"Analyzing sentence {i+1}/{len(sentences)}")
                
                result = self.analyze_sentence(sentence)
                sentence_results.append(result)
            
            # Perform document-level analysis
            originality_analysis = analyze_document_originality(sentence_results)
            
            # Compile final report
            report = {
                'document_path': str(file_path),
                'total_sentences': len(sentences),
                'sentence_results': sentence_results,
                'originality_analysis': originality_analysis,
                'processing_info': {
                    'semantic_engine_available': self.semantic_engine.model is not None,
                    'cross_encoder_available': self.reranker.model is not None,
                    'stylometry_available': self.stylometry_analyzer.nlp is not None
                }
            }
            
            logger.info(f"Document analysis completed. Originality score: {originality_analysis['originality_metrics']['originality_score']:.2%}")
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {e}")
            raise
    
    def generate_report_files(self, analysis_result: Dict[str, Any], 
                            output_dir: str = TEMP_DIR) -> Dict[str, str]:
        """Generate JSON and HTML report files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filenames
            json_path = os.path.join(output_dir, 'docinsight_report.json')
            html_path = os.path.join(output_dir, 'docinsight_report.html')
            
            # Save JSON report
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            
            # Generate HTML report
            html_content = self._generate_html_report(analysis_result)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Reports saved: {json_path}, {html_path}")
            return {'json': json_path, 'html': html_path}
            
        except Exception as e:
            logger.error(f"Error generating report files: {e}")
            raise
    
    def _generate_html_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate HTML report from analysis results"""
        doc_path = analysis_result.get('document_path', 'Unknown')
        originality = analysis_result.get('originality_analysis', {})
        metrics = originality.get('originality_metrics', {})
        
        # HTML template
        html_parts = [
            f"<html><head><title>DocInsight Report</title></head><body>",
            f"<h1>DocInsight Originality Report</h1>",
            f"<p><strong>Document:</strong> {html.escape(doc_path)}</p>",
            f"<h2>Summary</h2>",
            f"<p><strong>Originality Score:</strong> {metrics.get('originality_score', 0.0):.1%}</p>",
            f"<p><strong>Plagiarized Coverage:</strong> {metrics.get('plagiarized_coverage', 0.0):.1%}</p>",
            f"<p><strong>Risk Spans Found:</strong> {originality.get('total_risk_spans', 0)}</p>",
        ]
        
        # Sentence distribution
        distribution = metrics.get('sentence_distribution', {})
        html_parts.append("<h2>Sentence Risk Distribution</h2>")
        html_parts.append("<ul>")
        for risk_level, count in distribution.items():
            html_parts.append(f"<li><strong>{risk_level}:</strong> {count} sentences</li>")
        html_parts.append("</ul>")
        
        # Top risk spans
        top_spans = originality.get('top_risk_spans', [])
        if top_spans:
            html_parts.append("<h2>Top Risk Spans</h2>")
            for i, span in enumerate(top_spans, 1):
                html_parts.append(f"<div style='border:1px solid #ddd; margin:10px; padding:10px;'>")
                html_parts.append(f"<h3>Risk Span {i} ({span['risk_level']})</h3>")
                html_parts.append(f"<p><strong>Score:</strong> {span['avg_score']:.3f}</p>")
                html_parts.append(f"<p><strong>Sentences:</strong> {len(span['sentences'])}</p>")
                html_parts.append(f"<p><strong>Preview:</strong> {html.escape(span.get('preview_text', ''))}</p>")
                html_parts.append("</div>")
        
        html_parts.append("</body></html>")
        return "\n".join(html_parts)


# Convenience function for simple document analysis
def analyze_document_file(file_path: str, corpus_sentences: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a document file
    
    Args:
        file_path: Path to document file
        corpus_sentences: Optional custom corpus sentences
        
    Returns:
        Complete analysis results
    """
    pipeline = DocumentAnalysisPipeline(corpus_sentences)
    return pipeline.analyze_document(file_path)