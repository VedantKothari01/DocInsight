"""
Enhanced pipeline for DocInsight - Improved plagiarism detection with better features
"""
import os
import json
import html
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
from simple_corpus_builder import get_simple_corpus

# Optional imports for full functionality
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
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    HAS_ML_MODELS = True
except ImportError:
    HAS_ML_MODELS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass

class EnhancedPlagiarismDetector:
    """Enhanced plagiarism detection pipeline with improved features."""
    
    def __init__(self, 
                 corpus_size: int = 1000,  # Reduced for testing
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 spacy_model: str = "en_core_web_sm"):
        """
        Initialize the enhanced plagiarism detector.
        
        Args:
            corpus_size: Size of the reference corpus
            cross_encoder_model: Model for reranking
            spacy_model: SpaCy model for linguistic features
        """
        self.corpus_size = corpus_size
        self.cross_encoder_model_name = cross_encoder_model
        self.spacy_model_name = spacy_model
        
        # Models and data
        self.corpus_sentences = []
        self.faiss_index = None
        self.sbert_model = None
        self.cross_encoder = None
        self.nlp = None
        self.embeddings = None
        
        # Scoring weights
        self.alpha = 0.6  # Semantic similarity weight
        self.beta = 0.3   # Cross-encoder rerank weight
        self.gamma = 0.1  # Stylometry weight
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.4
        
    def initialize(self, force_rebuild_corpus: bool = False):
        """Initialize all models and load corpus."""
        logger.info("Initializing enhanced plagiarism detector...")
        
        try:
            # Load corpus (always works with fallback data)
            logger.info("Loading corpus...")
            self.corpus_sentences = get_simple_corpus(target_size=self.corpus_size)
            logger.info(f"Loaded corpus with {len(self.corpus_sentences)} sentences")
            
            # Try to load ML models if available
            if HAS_ML_MODELS:
                try:
                    logger.info("Loading SentenceTransformer model...")
                    # Try to use a simple model that might be available
                    self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Build embeddings and index
                    self._build_simple_embeddings()
                    
                    logger.info("Loading cross-encoder...")
                    self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
                    
                except Exception as e:
                    logger.warning(f"Could not load ML models: {e}")
                    logger.info("Running in simplified mode without embeddings")
                    self.sbert_model = None
                    self.cross_encoder = None
            else:
                logger.warning("ML libraries not available. Running in basic mode.")
            
            # Load spaCy model if available
            if HAS_SPACY:
                try:
                    logger.info(f"Loading spaCy model: {self.spacy_model_name}")
                    self.nlp = spacy.load(self.spacy_model_name)
                except OSError:
                    logger.warning(f"spaCy model {self.spacy_model_name} not found. Using basic features only.")
                    self.nlp = None
            else:
                logger.warning("spaCy not available. Using basic stylometry features only.")
            
            logger.info("Initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            # Continue with basic functionality
            logger.info("Continuing with basic functionality only")
    
    def _build_simple_embeddings(self):
        """Build simple embeddings and FAISS index if models are available."""
        if not self.sbert_model or not self.corpus_sentences:
            return
        
        try:
            logger.info("Building embeddings...")
            self.embeddings = self.sbert_model.encode(
                self.corpus_sentences, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            # Build FAISS index
            d = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)
            self.faiss_index.add(self.embeddings)
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.warning(f"Error building embeddings: {e}")
            self.embeddings = None
            self.faiss_index = None
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats."""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.suffix.lower() == '.pdf':
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling."""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            
            # Filter and clean sentences
            clean_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10 and len(sent.split()) >= 3:
                    clean_sentences.append(sent)
            
            return clean_sentences
            
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            # Fallback: simple split by periods
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            return sentences
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search in the corpus."""
        if not self.corpus_sentences:
            logger.warning("No corpus available")
            return []
        
        # If we have ML models, use semantic search
        if self.faiss_index is not None and self.sbert_model is not None:
            try:
                # Encode query
                query_embedding = self.sbert_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                
                # Search
                scores, indices = self.faiss_index.search(query_embedding, top_k)
                
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
        
        # Fallback: simple text-based similarity
        return self._simple_text_search(query, top_k)
    
    def _simple_text_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple text-based search when ML models are not available."""
        query_words = set(query.lower().split())
        results = []
        
        for sentence in self.corpus_sentences:
            sentence_words = set(sentence.lower().split())
            # Simple Jaccard similarity
            intersection = len(query_words & sentence_words)
            union = len(query_words | sentence_words)
            similarity = intersection / union if union > 0 else 0
            
            results.append({
                'sentence': sentence,
                'score': similarity
            })
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def rerank_candidates(self, query: str, candidates: List[str]) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        if self.cross_encoder is None or not candidates:
            return [{'sentence': c, 'rerank_score': 0.0} for c in candidates]
        
        try:
            pairs = [[query, candidate] for candidate in candidates]
            scores = self.cross_encoder.predict(pairs)
            
            # Sort by rerank scores
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            
            return [{'sentence': sent, 'rerank_score': float(score)} for sent, score in ranked]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return [{'sentence': c, 'rerank_score': 0.0} for c in candidates]
    
    def extract_stylometry_features(self, text: str) -> Dict[str, float]:
        """Extract enhanced stylometry features."""
        features = {}
        
        try:
            # Basic text statistics
            words = text.split()
            features['num_tokens'] = len(words)
            features['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0
            features['sentence_length'] = len(text)
            
            # Readability scores (if textstat available)
            if HAS_TEXTSTAT:
                features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
                features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
                features['automated_readability_index'] = textstat.automated_readability_index(text)
            else:
                # Simple approximations
                avg_sentence_len = len(words)
                avg_word_len = features['avg_word_len']
                features['flesch_reading_ease'] = max(0, 100 - avg_sentence_len - avg_word_len * 2)
                features['flesch_kincaid_grade'] = avg_sentence_len / 10 + avg_word_len / 2
                features['automated_readability_index'] = features['flesch_kincaid_grade']
            
            # Lexical diversity
            unique_words = set(word.lower() for word in words if word.isalpha())
            features['ttr'] = len(unique_words) / max(1, len(words))  # Type-token ratio
            
            # Punctuation density
            punctuation = sum(1 for char in text if not char.isalnum() and not char.isspace())
            features['punct_density'] = punctuation / max(1, len(text))
            
            # spaCy-based features (if available)
            if HAS_SPACY and self.nlp:
                doc = self.nlp(text)
                
                # POS tag ratios
                pos_counts = {}
                for token in doc:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                
                total_tokens = len(doc)
                features['noun_ratio'] = pos_counts.get('NOUN', 0) / max(1, total_tokens)
                features['verb_ratio'] = pos_counts.get('VERB', 0) / max(1, total_tokens)
                features['adj_ratio'] = pos_counts.get('ADJ', 0) / max(1, total_tokens)
                features['adv_ratio'] = pos_counts.get('ADV', 0) / max(1, total_tokens)
                
                # Dependency relations
                dep_counts = {}
                for token in doc:
                    dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
                
                features['subj_ratio'] = (dep_counts.get('nsubj', 0) + dep_counts.get('nsubjpass', 0)) / max(1, total_tokens)
                features['obj_ratio'] = (dep_counts.get('dobj', 0) + dep_counts.get('pobj', 0)) / max(1, total_tokens)
                
                # Named entities
                features['entity_density'] = len(doc.ents) / max(1, total_tokens)
            else:
                # Simple approximations when spaCy is not available
                # Basic POS estimation based on word endings and positions
                nouns = sum(1 for word in words if word.lower().endswith(('tion', 'ness', 'ment', 'ing', 'er', 'ly')))
                verbs = sum(1 for word in words if word.lower().endswith(('ed', 'ing', 'es', 's')))
                
                features.update({
                    'noun_ratio': nouns / max(1, len(words)),
                    'verb_ratio': verbs / max(1, len(words)),
                    'adj_ratio': 0.1,  # Default approximation
                    'adv_ratio': 0.05,  # Default approximation
                    'subj_ratio': 0.1,  # Default approximation
                    'obj_ratio': 0.1,   # Default approximation
                    'entity_density': 0.05  # Default approximation
                })
            
        except Exception as e:
            logger.error(f"Error extracting stylometry features: {e}")
            # Return default features on error
            features = {
                'num_tokens': 0, 'avg_word_len': 0, 'sentence_length': 0,
                'flesch_reading_ease': 50, 'flesch_kincaid_grade': 8, 'automated_readability_index': 8,
                'ttr': 0.5, 'punct_density': 0.1, 'noun_ratio': 0.2, 'verb_ratio': 0.2,
                'adj_ratio': 0.1, 'adv_ratio': 0.05, 'subj_ratio': 0.1, 'obj_ratio': 0.1, 'entity_density': 0.05
            }
        
        return features
    
    def compute_stylometry_similarity(self, query_features: Dict, candidate_features: Dict) -> float:
        """Compute stylometry similarity between query and candidate."""
        try:
            # Normalize and compute similarity for key features
            similarity_scores = []
            
            # Features to compare with their weights
            feature_weights = {
                'flesch_reading_ease': 0.2,
                'ttr': 0.15,
                'avg_word_len': 0.1,
                'punct_density': 0.1,
                'noun_ratio': 0.15,
                'verb_ratio': 0.15,
                'adj_ratio': 0.05,
                'adv_ratio': 0.05,
                'entity_density': 0.05
            }
            
            total_weight = 0
            weighted_similarity = 0
            
            for feature, weight in feature_weights.items():
                if feature in query_features and feature in candidate_features:
                    q_val = query_features[feature]
                    c_val = candidate_features[feature]
                    
                    # Compute normalized similarity (1 - normalized absolute difference)
                    if feature == 'flesch_reading_ease':
                        # Reading ease ranges from 0-100
                        similarity = 1 - abs(q_val - c_val) / 100.0
                    elif feature in ['ttr', 'punct_density', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']:
                        # Ratios range from 0-1
                        similarity = 1 - abs(q_val - c_val)
                    elif feature == 'avg_word_len':
                        # Word length similarity (normalized by max expected length)
                        similarity = 1 - abs(q_val - c_val) / 20.0
                    elif feature == 'entity_density':
                        # Entity density similarity
                        similarity = 1 - abs(q_val - c_val)
                    else:
                        similarity = 0.5  # Default
                    
                    similarity = max(0, min(1, similarity))  # Clamp to [0,1]
                    weighted_similarity += similarity * weight
                    total_weight += weight
            
            return weighted_similarity / max(total_weight, 0.01)
            
        except Exception as e:
            logger.error(f"Error computing stylometry similarity: {e}")
            return 0.5  # Default similarity
    
    def compute_fused_score(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Compute fused score combining semantic, rerank, and stylometry signals."""
        if not search_results:
            return []
        
        try:
            # Extract candidates
            candidates = [result['sentence'] for result in search_results]
            semantic_scores = [result['score'] for result in search_results]
            
            # Rerank candidates
            rerank_results = self.rerank_candidates(query, candidates)
            rerank_map = {r['sentence']: r['rerank_score'] for r in rerank_results}
            
            # Get rerank scores and normalize
            rerank_scores = [rerank_map.get(cand, 0.0) for cand in candidates]
            if rerank_scores:
                min_rerank = min(rerank_scores)
                rerank_scores = [(score - min_rerank) for score in rerank_scores]
            
            # Extract stylometry features
            query_features = self.extract_stylometry_features(query)
            
            # Compute fused scores
            fused_results = []
            for i, candidate in enumerate(candidates):
                candidate_features = self.extract_stylometry_features(candidate)
                stylometry_score = self.compute_stylometry_similarity(query_features, candidate_features)
                
                # Compute fused score
                semantic_score = semantic_scores[i]
                rerank_score = rerank_scores[i] if i < len(rerank_scores) else 0.0
                
                fused_score = (self.alpha * semantic_score + 
                              self.beta * rerank_score + 
                              self.gamma * stylometry_score)
                
                # Determine confidence level
                confidence = self._get_confidence_level(fused_score)
                
                fused_results.append({
                    'candidate': candidate,
                    'semantic': float(semantic_score),
                    'rerank': float(rerank_map.get(candidate, 0.0)),
                    'stylometry_score': float(stylometry_score),
                    'stylometry_features': candidate_features,
                    'fused': float(fused_score),
                    'confidence': confidence
                })
            
            # Sort by fused score
            fused_results.sort(key=lambda x: x['fused'], reverse=True)
            return fused_results
            
        except Exception as e:
            logger.error(f"Error computing fused scores: {e}")
            return []
    
    def _get_confidence_level(self, score: float) -> str:
        """Determine confidence level based on score."""
        if score >= self.high_confidence_threshold:
            return "HIGH"
        elif score >= self.medium_confidence_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_enhanced_report(self, 
                               document_path: str,
                               output_json: str = "/tmp/report.json",
                               output_html: str = "/tmp/report.html") -> Dict:
        """Generate comprehensive plagiarism report."""
        try:
            # Extract text and split into sentences
            text = self.extract_text(document_path)
            sentences = self.split_sentences(text)
            
            if not sentences:
                raise EnhancedPipelineError("No sentences found in document")
            
            # Initialize report
            report = {
                'document': str(document_path),
                'total_sentences': len(sentences),
                'corpus_size': len(self.corpus_sentences),
                'sentences': []
            }
            
            # Process each sentence
            for sentence in sentences:
                # Search for similar sentences
                search_results = self.semantic_search(sentence, top_k=5)
                
                if search_results:
                    # Compute fused scores
                    fused_results = self.compute_fused_score(sentence, search_results)
                    best_match = fused_results[0] if fused_results else {}
                    
                    # Add to report
                    sentence_report = {
                        'sentence': sentence,
                        'best_match': best_match.get('candidate', ''),
                        'semantic_score': best_match.get('semantic', 0.0),
                        'rerank_score': best_match.get('rerank', 0.0),
                        'stylometry_score': best_match.get('stylometry_score', 0.0),
                        'fused_score': best_match.get('fused', 0.0),
                        'confidence': best_match.get('confidence', 'LOW'),
                        'stylometry_features': best_match.get('stylometry_features', {}),
                        'all_matches': fused_results[:3]  # Top 3 matches
                    }
                else:
                    # No matches found
                    sentence_report = {
                        'sentence': sentence,
                        'best_match': '',
                        'semantic_score': 0.0,
                        'rerank_score': 0.0,
                        'stylometry_score': 0.0,
                        'fused_score': 0.0,
                        'confidence': 'LOW',
                        'stylometry_features': {},
                        'all_matches': []
                    }
                
                report['sentences'].append(sentence_report)
            
            # Compute overall statistics
            fused_scores = [s['fused_score'] for s in report['sentences']]
            report['overall_stats'] = {
                'avg_fused_score': np.mean(fused_scores),
                'max_fused_score': max(fused_scores) if fused_scores else 0,
                'high_confidence_count': sum(1 for s in report['sentences'] if s['confidence'] == 'HIGH'),
                'medium_confidence_count': sum(1 for s in report['sentences'] if s['confidence'] == 'MEDIUM'),
                'low_confidence_count': sum(1 for s in report['sentences'] if s['confidence'] == 'LOW')
            }
            
            # Save reports
            self._save_json_report(report, output_json)
            self._save_html_report(report, output_html)
            
            logger.info(f"Report generated: {len(sentences)} sentences analyzed")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise EnhancedPipelineError(f"Failed to generate report: {e}")
    
    def _save_json_report(self, report: Dict, filepath: str):
        """Save report as JSON."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON report saved to {filepath}")
        except Exception as e:
            logger.warning(f"Error saving JSON report: {e}")
    
    def _save_html_report(self, report: Dict, filepath: str):
        """Save report as HTML."""
        try:
            html_parts = [
                "<html><head><title>DocInsight Plagiarism Report</title></head><body>",
                f"<h1>DocInsight Plagiarism Report</h1>",
                f"<p><strong>Document:</strong> {html.escape(str(report['document']))}</p>",
                f"<p><strong>Total Sentences:</strong> {report['total_sentences']}</p>",
                f"<p><strong>Corpus Size:</strong> {report['corpus_size']}</p>",
            ]
            
            # Overall statistics
            stats = report['overall_stats']
            html_parts.extend([
                "<h2>Overall Statistics</h2>",
                f"<p><strong>Average Fused Score:</strong> {stats['avg_fused_score']:.3f}</p>",
                f"<p><strong>Max Fused Score:</strong> {stats['max_fused_score']:.3f}</p>",
                f"<p><strong>High Confidence Matches:</strong> {stats['high_confidence_count']}</p>",
                f"<p><strong>Medium Confidence Matches:</strong> {stats['medium_confidence_count']}</p>",
                f"<p><strong>Low Confidence Matches:</strong> {stats['low_confidence_count']}</p>",
            ])
            
            # Sentence-by-sentence analysis
            html_parts.append("<h2>Detailed Analysis</h2>")
            
            for i, sentence_data in enumerate(report['sentences']):
                confidence_color = {
                    'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#44aa44'
                }.get(sentence_data['confidence'], '#888888')
                
                html_parts.extend([
                    f'<div style="border:1px solid #ddd;padding:10px;margin:10px 0;border-left:5px solid {confidence_color};">',
                    f"<h3>Sentence {i+1} ({sentence_data['confidence']} confidence)</h3>",
                    f"<p><strong>Text:</strong> {html.escape(sentence_data['sentence'])}</p>",
                    f"<p><strong>Best Match:</strong> {html.escape(sentence_data['best_match'])}</p>",
                    f"<p><strong>Scores:</strong> Semantic={sentence_data['semantic_score']:.3f}, "
                    f"Rerank={sentence_data['rerank_score']:.3f}, "
                    f"Stylometry={sentence_data['stylometry_score']:.3f}, "
                    f"<strong>Fused={sentence_data['fused_score']:.3f}</strong></p>",
                    "</div>"
                ])
            
            html_parts.append("</body></html>")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))
            
            logger.info(f"HTML report saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Error saving HTML report: {e}")

# Convenience function for easy usage
def get_enhanced_detector(corpus_size: int = 50000, 
                         force_rebuild: bool = False) -> EnhancedPlagiarismDetector:
    """Get an initialized enhanced plagiarism detector."""
    detector = EnhancedPlagiarismDetector(corpus_size=corpus_size)
    detector.initialize(force_rebuild_corpus=force_rebuild)
    return detector