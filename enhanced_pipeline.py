"""
Enhanced pipeline for DocInsight with fine-tuned model support

Updated to use fine-tuned Cross-Encoder and spaCy models
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import html
import math

# Third-party imports
import numpy as np
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None
    FAISS_AVAILABLE = False
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
import textstat
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import docx2txt
import fitz  # PyMuPDF

# Local imports
from config import *
from scoring.core import SentenceClassifier, analyze_document_originality

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


class StylometryAnalyzer:
    """Handles stylometry feature extraction with fine-tuned spaCy support"""
    
    def __init__(self):
        try:
            nltk.data.find('corpora/stopwords')
        except Exception as e:
            nltk.download('stopwords', quiet=True)

        self.nlp = None
        self.model_source = 'none'
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load fine-tuned or base spaCy model"""
        # Try fine-tuned model first
        fine_tuned_path = "models/spacy_finetuned"
        
        if Path(fine_tuned_path).exists():
            try:
                logger.info(f"Loading fine-tuned spaCy model from: {fine_tuned_path}")
                self.nlp = spacy.load(fine_tuned_path)
                self.model_source = 'fine_tuned'
                logger.info("✓ Fine-tuned spaCy model loaded")
                return
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned spaCy model: {e}")
        
        # Fallback to base model
        try:
            logger.info(f"Loading base spaCy model: {SPACY_MODEL_NAME}")
            self.nlp = spacy.load(SPACY_MODEL_NAME)
            self.model_source = 'base'
            logger.info(f"✓ Base spaCy model loaded: {SPACY_MODEL_NAME}")
        except OSError: 
            logger.warning(f"spaCy model {SPACY_MODEL_NAME} not found. Stylometry features will be limited.")
            self.nlp = None
            self.model_source = 'unavailable'
    
    def extract_features(self, sentence: str) -> Dict[str, float]:
        """Extract comprehensive stylometry features from sentence"""
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
            features['adj_ratio'] = pos_counts.get('ADJ', 0) / max(1, len(doc))
            features['adv_ratio'] = pos_counts.get('ADV', 0) / max(1, len(doc))
            
            # Function words using NLTK's stopwords
            english_stopwords = set(stopwords.words('english'))
            func_word_count = sum(1 for w in words if w in english_stopwords)
            features['function_word_ratio'] = func_word_count / max(1, len(words))
            
            # Stopword ratio using spaCy's stopwords
            stopword_count = sum(1 for t in alpha_tokens if t.is_stop)
            features['stopword_ratio'] = stopword_count / max(1, len(alpha_tokens))
            
            # N-gram entropy
            features['bigram_entropy'] = self._calculate_ngram_entropy(words, n=2)
            features['trigram_entropy'] = self._calculate_ngram_entropy(words, n=3)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting stylometry features: {e}")
            return {'flesch_reading_ease': 0.0}
        
    def _calculate_ngram_entropy(self, words: List[str], n: int) -> float:
        """Calculate entropy of n-grams in text"""
        if len(words) < n:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        if not ngrams:
            return 0.0
        
        ngram_counts = {}
        for ng in ngrams:
            ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
        
        total = len(ngrams)
        entropy = 0.0
        for count in ngram_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy


class SemanticSearchEngine:
    """Handles semantic search using fine-tuned SBERT and FAISS"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.corpus_sentences = []
        self.model_source = 'unloaded'
        self.model_path = None
        self._load_models()
        
        # Phase 2: Try to use persistent retrieval system
        self.use_persistent_retrieval = False
        self.retrieval_engine = None
        self._try_load_persistent_retrieval()
    
    def _load_models(self):
        """Load fine-tuned semantic model"""
        # Try fine-tuned model first
        fine_tuned_path = MODEL_FINE_TUNED_PATH
        
        if Path(fine_tuned_path).exists():
            try:
                logger.info(f"Loading fine-tuned SBERT model from: {fine_tuned_path}")
                self.model = SentenceTransformer(fine_tuned_path)
                self.model_source = 'fine_tuned'
                self.model_path = fine_tuned_path
                logger.info("✓ Fine-tuned SBERT model loaded (trained on PAWS, QQP, STSB)")
                return
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model: {e}")
        
        # Fallback to base model
        try:
            logger.info(f"Loading base SBERT model: {SBERT_MODEL_NAME}")
            self.model = SentenceTransformer(SBERT_MODEL_NAME)
            self.model_source = 'base'
            self.model_path = SBERT_MODEL_NAME
            logger.info(f"✓ Base SBERT model loaded: {SBERT_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.model_source = 'error'
            
    def _try_load_persistent_retrieval(self):
        """Try to load Phase 2 persistent retrieval system"""
        try:
            from retrieval import get_retrieval_engine
            self.retrieval_engine = get_retrieval_engine()
            
            if self.retrieval_engine.is_ready():
                self.use_persistent_retrieval = True
                logger.info("Using persistent retrieval system (Phase 2)")
            else:
                logger.info("Persistent retrieval not ready, using in-memory system (Phase 1)")
                
        except ImportError:
            logger.debug("Phase 2 retrieval system not available")
        except Exception as e:
            logger.debug(f"Could not load persistent retrieval: {e}")
    
    def build_index(self, corpus_sentences: List[str]):
        """Build FAISS index from corpus sentences"""
        if self.use_persistent_retrieval:
            logger.info("Using persistent retrieval system - skipping in-memory index build")
            return True

        if self.model is None:
            logger.error("SBERT model not loaded. Cannot build index.")
            return False

        try:
            logger.info(f"Building semantic structures for {len(corpus_sentences)} sentences")
            self.corpus_sentences = corpus_sentences

            corpus_embeddings = self.model.encode(corpus_sentences, convert_to_numpy=True)
            norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-12
            corpus_embeddings = corpus_embeddings / norms
            self._corpus_embeddings = corpus_embeddings

            if FAISS_AVAILABLE:
                d = corpus_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(d)  # type: ignore
                self.index.add(corpus_embeddings)
                logger.info(f"FAISS index built with {self.index.ntotal} sentences")
            else:
                logger.warning("FAISS not available; will use numpy fallback")
            return True

        except Exception as e:
            logger.error(f"Error building semantic index: {e}")
            return False

    def _fallback_numpy_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback cosine similarity search"""
        if self.model is None or not hasattr(self, '_corpus_embeddings'):
            return []
        try:
            q_emb = self.model.encode([query], convert_to_numpy=True)
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
            sims = np.dot(self._corpus_embeddings, q_emb[0])
            top_indices = np.argsort(-sims)[:top_k]
            results = []
            for idx in top_indices:
                results.append({'sentence': self.corpus_sentences[idx], 'score': float(sims[idx])})
            return results
        except Exception as e:
            logger.error(f"Fallback numpy search failed: {e}")
            return []
    
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Perform semantic search"""
        if self.use_persistent_retrieval and self.retrieval_engine:
            try:
                results = self.retrieval_engine.retrieve_similar_chunks(
                    [query], top_k=top_k
                )
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'sentence': result.text,
                        'score': float(result.score)
                    })
                    
                return formatted_results
                
            except Exception as e:
                logger.warning(f"Persistent retrieval failed, falling back: {e}")
                
        if self.model is None:
            logger.warning("Search engine model not loaded")
            return []
        
        try:
            if self.index is not None:
                query_embedding = self.model.encode([query], convert_to_numpy=True)
                query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12)
                scores, indices = self.index.search(query_embedding, top_k)  # type: ignore
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self.corpus_sentences):
                        results.append({'sentence': self.corpus_sentences[idx], 'score': float(score)})
                return results
            else:
                return self._fallback_numpy_search(query, top_k)
        except Exception as e:
            logger.warning(f"FAISS search error: {e}")
            return self._fallback_numpy_search(query, top_k)


class CrossEncoderReranker:
    """Handles cross-encoder reranking with fine-tuned model support"""
    
    def __init__(self):
        self.model = None
        self.model_source = 'none'
        self._load_model()
    
    def _load_model(self):
        """Load fine-tuned or base cross-encoder model"""
        # Try fine-tuned model first
        fine_tuned_path = "models/cross_encoder_finetuned"
        
        if Path(fine_tuned_path).exists():
            try:
                logger.info(f"Loading fine-tuned Cross-Encoder from: {fine_tuned_path}")
                self.model = CrossEncoder(fine_tuned_path)
                self.model_source = 'fine_tuned'
                logger.info("✓ Fine-tuned Cross-Encoder loaded")
                return
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned Cross-Encoder: {e}")
        
        # Fallback to base model
        try:
            logger.info(f"Loading base Cross-Encoder: {CROSS_ENCODER_MODEL_NAME}")
            self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            self.model_source = 'base'
            logger.info("✓ Base Cross-Encoder loaded")
        except Exception as e:
            logger.warning(f"Failed to load Cross-Encoder: {e}. Reranking disabled.")
            self.model = None
            self.model_source = 'unavailable'
    
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
    """Main pipeline for document analysis with fine-tuned models"""
    
    def __init__(self, corpus_sentences: Optional[List[str]] = None):
        self.text_extractor = TextExtractor()
        self.sentence_processor = SentenceProcessor()
        self.stylometry_analyzer = StylometryAnalyzer()
        self.semantic_engine = SemanticSearchEngine()
        self.reranker = CrossEncoderReranker()
        self.sentence_classifier = SentenceClassifier()

        # Retrieval setup
        from retrieval import RetrievalEngine
        from pathlib import Path

        logger.info("Initializing retrieval engine...")
        self.retrieval_engine = RetrievalEngine()

        db_path = Path("docinsight.db")
        index_path = Path("indexes/faiss.index")

        if db_path.exists() and index_path.exists():
            try:
                logger.info(f"Loading local prebuilt corpus: {db_path} + {index_path}")
                self.retrieval_engine.index_manager.load_index()
                corpus_stats = self.retrieval_engine.get_corpus_stats()
                logger.info(f"✅ Local corpus loaded: {corpus_stats.get('documents', 'N/A')} docs, "
                            f"{corpus_stats.get('embedded_chunks', 'N/A')} chunks")
            except Exception as e:
                logger.error(f"⚠️ Failed to load prebuilt corpus: {e}")
        else:
            logger.warning("⚠️ No prebuilt corpus found; using in-memory retrieval only.")

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
            risk_level, confidence_score, match_strength, reason = self.sentence_classifier.classify_sentence(fused_results)
            
            # Get best match
            best_match = fused_results[0] if fused_results else {}
            
            return {
                'sentence': sentence,
                'risk_level': risk_level,
                'confidence_score': confidence_score,
                'match_strength': match_strength,
                'reason': reason,
                'best_match': best_match.get('candidate', '')[:300] + "...",
                'semantic_score': best_match.get('semantic_score', 0.0),
                'semantic_norm': best_match.get('semantic_norm', 0.0),
                'rerank_score': best_match.get('rerank_score', 0.0),
                'rerank_norm': best_match.get('rerank_norm', 0.0),
                'stylometry_score': best_match.get('stylometry_score', 0.0),
                'fused_score': best_match.get('fused_score', 0.0),
                'components': best_match.get('components', {}),
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
            'match_strength': 'NONE',
            'reason': 'Analysis failed',
            'best_match': '',
            'semantic_score': 0.0,
            'semantic_norm': 0.0,
            'rerank_score': 0.0,
            'rerank_norm': 0.0,
            'stylometry_score': 0.0,
            'fused_score': 0.0,
            'components': {},
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

            # Citation masking
            try:
                from ingestion.citation_mask import CitationMasker
                citation_masker = CitationMasker()
            except Exception:
                citation_masker = type('NullMasker',(object,),{'enabled':False})()
            
            masked_text, citations = text, []
            citation_summary = {}
            try:
                if citation_masker.enabled:
                    masked_text, citations = citation_masker.mask_citations(text)
                    citation_summary = citation_masker.get_citation_summary(citations)
                    logger.info(f"Citation masking applied: {citation_summary.get('total',0)} citations masked")
            except Exception as ce:
                logger.warning(f"Citation masking failed: {ce}")
            
            # Split into sentences
            sentences = self.sentence_processor.split_sentences(masked_text)
            if not sentences:
                raise ValueError("No sentences found in document")
            
            logger.info(f"Found {len(sentences)} sentences to analyze")
            
            # Analyze each sentence
            sentence_results = []
            for i, sentence in enumerate(sentences):
                if i % 10 == 0:
                    logger.info(f"Analyzing sentence {i+1}/{len(sentences)}")
                
                result = self.analyze_sentence(sentence)
                sentence_results.append(result)

            # Post-process repeated matches
            sentence_results = self._postprocess_repeated_matches(sentence_results)
            
            # Document-level analysis
            originality_analysis = analyze_document_originality(sentence_results)
            
            # Compile final report
            report = {
                'document_path': str(file_path),
                'total_sentences': len(sentences),
                'sentence_results': sentence_results,
                'originality_analysis': originality_analysis,
                'citations': {
                    'masking_enabled': citation_masker.enabled,
                    'summary': citation_summary
                },
                'processing_info': {
                    'semantic_engine_available': self.semantic_engine.model is not None,
                    'semantic_model_source': self.semantic_engine.model_source,
                    'cross_encoder_available': self.reranker.model is not None,
                    'cross_encoder_source': self.reranker.model_source,
                    'stylometry_available': self.stylometry_analyzer.nlp is not None,
                    'stylometry_source': self.stylometry_analyzer.model_source,
                    'semantic_model_path': self.semantic_engine.model_path,
                    'reuse_decay': {
                        'allowance': REUSE_DECAY_ALLOWANCE,
                        'decay_factor': REUSE_DECAY_FACTOR
                    }
                }
            }
            
            logger.info(f"Document analysis completed. Originality score: {originality_analysis['originality_metrics']['originality_score']:.2%}")
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {e}")
            raise

    def _postprocess_repeated_matches(self, sentence_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dampen confidence/risk when same corpus sentence is reused excessively"""
        allowance = REUSE_DECAY_ALLOWANCE
        decay_factor = REUSE_DECAY_FACTOR
        match_counts: Dict[str, int] = {}

        for res in sentence_results:
            bm = res.get('best_match') or ''
            if not bm:
                continue
            match_counts[bm] = match_counts.get(bm, 0) + 1
            occurrence = match_counts[bm]
            if occurrence <= allowance:
                continue
            
            extra = occurrence - allowance
            multiplier = decay_factor ** extra
            original_conf = res.get('confidence_score', 0.0)
            new_conf = max(0.0, original_conf * multiplier)
            res['confidence_score'] = new_conf
            
            fused = res.get('fused_score', original_conf)
            res['fused_score'] = max(0.0, fused * multiplier)
            
            # Downgrade risk if needed
            risk = res.get('risk_level', RISK_LEVELS['LOW'])
            sem_norm = res.get('semantic_norm', 0.0)
            if risk == RISK_LEVELS['HIGH']:
                if not (res['fused_score'] >= HIGH_RISK_THRESHOLD and sem_norm >= SEMANTIC_HIGH_FLOOR):
                    if res['fused_score'] >= MEDIUM_RISK_THRESHOLD and sem_norm >= SEMANTIC_MEDIUM_FLOOR:
                        res['risk_level'] = RISK_LEVELS['MEDIUM']
                        res['reason'] += f" | downgraded to MEDIUM (repeated match: {occurrence} uses)"
                    else:
                        res['risk_level'] = RISK_LEVELS['LOW']
                        res['reason'] += f" | downgraded to LOW (repeated match: {occurrence} uses)"
            elif risk == RISK_LEVELS['MEDIUM']:
                if not (res['fused_score'] >= MEDIUM_RISK_THRESHOLD and sem_norm >= SEMANTIC_MEDIUM_FLOOR):
                    res['risk_level'] = RISK_LEVELS['LOW']
                    res['reason'] += f" | downgraded to LOW (repeated match: {occurrence} uses)"
            
        return sentence_results
    
    def generate_report_files(self, analysis_result: Dict[str, Any], 
                            output_dir: str = TEMP_DIR) -> Dict[str, str]:
        """Generate JSON and HTML report files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            json_path = os.path.join(output_dir, 'docinsight_report.json')
            html_path = os.path.join(output_dir, 'docinsight_report.html')
            
            # Save JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            
            # Generate HTML
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
        processing = analysis_result.get('processing_info', {})
        
        html_parts = [
            f"<html><head><title>DocInsight Report</title></head><body>",
            f"<h1>DocInsight Originality Report</h1>",
            f"<p><strong>Document:</strong> {html.escape(doc_path)}</p>",
            f"<h2>Summary</h2>",
            f"<p><strong>Originality Score:</strong> {metrics.get('originality_score', 0.0):.1%}</p>",
            f"<p><strong>Plagiarized Coverage:</strong> {metrics.get('plagiarized_coverage', 0.0):.1%}</p>",
            f"<p><strong>Risk Spans:</strong> {originality.get('total_risk_spans', 0)}</p>",
        ]
        
        # Model info
        html_parts.append("<h2>Model Information</h2>")
        html_parts.append("<ul>")
        html_parts.append(f"<li><strong>Semantic Model:</strong> {processing.get('semantic_model_source', 'unknown')} "
                         f"({processing.get('semantic_model_path', 'N/A')})</li>")
        html_parts.append(f"<li><strong>Cross-Encoder:</strong> {processing.get('cross_encoder_source', 'unknown')}</li>")
        html_parts.append(f"<li><strong>spaCy Model:</strong> {processing.get('stylometry_source', 'unknown')}</li>")
        html_parts.append("</ul>")
        
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


# Convenience function
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