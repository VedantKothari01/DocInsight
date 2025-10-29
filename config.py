"""
config.py
Configuration file for DocInsight
Complete merged configuration with all parameters
"""

import os
from pathlib import Path

# ============================================================================
# MODEL PATHS & CONFIGURATIONS
# ============================================================================

# Base models (used for initial training)
MODEL_BASE_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
SBERT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
SPACY_MODEL_NAME = 'en_core_web_sm'

# Fine-tuned model paths (created by fine_tuner.py)
MODEL_FINE_TUNED_PATH = 'models/sbert_finetuned'
CROSS_ENCODER_FINE_TUNED_PATH = 'models/cross_encoder_finetuned'
SPACY_FINE_TUNED_PATH = 'models/spacy_finetuned'
AI_LIKENESS_MODEL_PATH = 'models/ai_likeness/'

# Use fine-tuned models if available (auto-detected by pipeline)
USE_FINE_TUNED_MODEL = os.getenv('DOCINSIGHT_USE_FINE_TUNED', 'true').lower() == 'true'
FORCE_RETRAIN = os.getenv('DOCINSIGHT_FORCE_RETRAIN', 'false').lower() == 'false'

# ============================================================================
# FINE-TUNING PARAMETERS
# ============================================================================

# Training hyperparameters - Sentence Transformer
FINE_TUNING_EPOCHS = int(os.getenv('DOCINSIGHT_FINE_TUNING_EPOCHS', '3'))
FINE_TUNING_BATCH_SIZE = int(os.getenv('DOCINSIGHT_FINE_TUNING_BATCH_SIZE', '16'))
FINE_TUNING_LEARNING_RATE = float(os.getenv('DOCINSIGHT_FINE_TUNING_LR', '2e-5'))
FINE_TUNING_WARMUP_RATIO = 0.1

# Cross-encoder fine-tuning
CROSS_ENCODER_EPOCHS = 4
CROSS_ENCODER_BATCH_SIZE = 16
CROSS_ENCODER_LEARNING_RATE = 2e-5

# spaCy fine-tuning
SPACY_TRAINING_ITERATIONS = 50
SPACY_DROPOUT = 0.2

# Dataset sizes
PAWS_MAX_EXAMPLES = 5000
QQP_MAX_EXAMPLES = 5000
STSB_MAX_EXAMPLES = 20000

# Extended corpus (includes additional training data)
EXTENDED_CORPUS_ENABLED = os.getenv('DOCINSIGHT_EXTENDED_CORPUS', 'true').lower() == 'true'

# Auto-update config after training
AUTO_UPDATE_CONFIG = True

# ============================================================================
# RISK THRESHOLDS (Auto-updated by fine_tuner.py)
# ============================================================================

# Primary risk thresholds for fused scores
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

# Semantic similarity floors (minimum semantic score required)
# A candidate must satisfy BOTH fused thresholds (above) and these semantic floors
# to be promoted to a higher risk class. Prevents random noisy matches.
SEMANTIC_HIGH_FLOOR = 0.60   # Minimum normalized semantic score to allow HIGH
SEMANTIC_MEDIUM_FLOOR = 0.40  # Minimum normalized semantic score to allow MEDIUM
SEMANTIC_MIN_MATCH = 0.35     # Below this semantic raw score => treat as LOW regardless

# Sentence processing
MIN_SENTENCE_LENGTH = 3  # Minimum words for valid sentence

# ============================================================================
# FUSION WEIGHTS (Auto-updated by fine_tuner.py)
# ============================================================================

# Component weights for score fusion
# Rationale: Semantic similarity provides the strongest signal for plagiarism detection,
# stylometry helps identify writing pattern deviations, and AI-likeness detects
# potential AI-generated content that may indicate sophisticated plagiarism attempts.
WEIGHT_SEMANTIC = float(os.getenv('DOCINSIGHT_W_SEMANTIC', '0.6'))  # Semantic similarity weight
WEIGHT_STYLO = float(os.getenv('DOCINSIGHT_W_STYLO', '0.25'))  # Stylometric deviation weight  
WEIGHT_AI = float(os.getenv('DOCINSIGHT_W_AI', '0.15'))  # AI-likeness probability weight

# Ensure weights sum to 1.0 (normalize if environment overrides don't sum to 1.0)
_total_weight = WEIGHT_SEMANTIC + WEIGHT_STYLO + WEIGHT_AI
if abs(_total_weight - 1.0) > 0.01:  # Allow small floating point errors
    WEIGHT_SEMANTIC = WEIGHT_SEMANTIC / _total_weight
    WEIGHT_STYLO = WEIGHT_STYLO / _total_weight
    WEIGHT_AI = WEIGHT_AI / _total_weight

# Validation: weights should sum to 1.0
assert abs(WEIGHT_SEMANTIC + WEIGHT_STYLO + WEIGHT_AI - 1.0) < 0.01, \
    "Fusion weights must sum to 1.0"

# ============================================================================
# SIMILARITY SCORING
# ============================================================================

# Reranking weights (for combining semantic and cross-encoder scores)
RERANK_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

# Fusion weights for sentence-level scoring (legacy - kept for compatibility)
FUSION_WEIGHTS = {
    'semantic': WEIGHT_SEMANTIC,
    'cross_encoder': 0.3, 
    'stylometry': WEIGHT_STYLO
}

# Stylometry scoring
STYLOMETRY_FEATURE_WEIGHTS = {
    'flesch_reading_ease': 0.15,
    'ttr': 0.10,
    'avg_word_len': 0.08,
    'punct_density': 0.07,
    'noun_ratio': 0.10,
    'verb_ratio': 0.10,
    'adj_ratio': 0.08,
    'function_word_ratio': 0.12,
    'stopword_ratio': 0.10,
    'bigram_entropy': 0.05,
    'trigram_entropy': 0.05
}

# Stylometry normalization constants
FLESCH_SCORE_NORMALIZATION = 50.0

# Stylometry feature configuration
function_words = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what'
}  # Most common function words for stylometric analysis

# ============================================================================
# DOCUMENT ANALYSIS
# ============================================================================

# Risk levels
RISK_LEVELS = {
    'HIGH': 'HIGH',
    'MEDIUM': 'MEDIUM',
    'LOW': 'LOW'
}

# Span detection
MIN_SPAN_LENGTH = 3  # Minimum sentences for a risk span
SPAN_MERGE_DISTANCE = 2  # Merge spans within N sentences

# Repeated match handling (reuse/decay configuration)
# These control dampening of confidence for the same corpus best_match appearing
# many times across a document.
REUSE_DECAY_ALLOWANCE = int(os.getenv('DOCINSIGHT_REUSE_ALLOWANCE', '2'))  # First N occurrences unpenalized
REUSE_DECAY_FACTOR = float(os.getenv('DOCINSIGHT_REUSE_DECAY', '0.85'))  # Decay multiplier per extra occurrence

# Validation for decay parameters
if REUSE_DECAY_ALLOWANCE < 0:
    REUSE_DECAY_ALLOWANCE = 0
if not (0.0 < REUSE_DECAY_FACTOR <= 1.0):
    REUSE_DECAY_FACTOR = 0.85  # Clamp to sensible range

# Aggregation weights for document-level scoring
# Formula: Originality = 1 - f(coverage, severity, span_ratio)
AGGREGATION_WEIGHTS = {
    'alpha': 0.55,  # Coverage weight - how much of document is covered by plagiarized spans
    'beta': 0.30,   # Severity weight - average severity of plagiarized spans  
    'gamma': 0.15   # Span ratio weight - ratio of plagiarized spans to total spans
}

# Plagiarism factor components (for originality score)
ALPHA_COVERAGE = 0.4  # Weight for coverage component
BETA_SEVERITY = 0.4  # Weight for severity component
GAMMA_SPAN_RATIO = 0.2  # Weight for span ratio component

# Validation
assert abs(ALPHA_COVERAGE + BETA_SEVERITY + GAMMA_SPAN_RATIO - 1.0) < 0.01, \
    "Plagiarism factor weights must sum to 1.0"

# Match strength thresholds
MATCH_STRENGTH_THRESHOLDS = {
    'EXACT': 0.95,
    'STRONG': 0.85,
    'MODERATE': 0.70,
    'WEAK': 0.50
}

# AI-likeness detection
AI_LIKENESS_THRESHOLD = float(os.getenv('DOCINSIGHT_AI_THRESHOLD', '0.7'))
SUSPICIOUS_SECTION_COUNT = int(os.getenv('DOCINSIGHT_SUSPICIOUS_SECTIONS', '3'))  # Top N suspicious sections to show

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

# Search parameters
DEFAULT_TOP_K = 10  # Number of candidates to retrieve
MAX_CANDIDATES = 10  # Max candidates to include in report
RETRIEVAL_TOP_K = 10

# Minimum similarity threshold for retrieval
MIN_SIM_THRESHOLD = 0.3

# ============================================================================
# FILE HANDLING
# ============================================================================

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt']

# Temporary directory
TEMP_DIR = '/tmp'

# Output directories
OUTPUT_DIR = 'output'
REPORT_DIR = 'training_reports'

# UI configuration
MAX_SENTENCE_DISPLAY = 100  # Limit sentence detail output to prevent UI overload
TOP_RISK_SPANS_PREVIEW = 3  # Number of top risk spans to show in preview

# File processing limits
PDF_MAX_PAGES = 100
DOCX_MAX_PARAGRAPHS = 500

# ============================================================================
# DATABASE & INDEXING
# ============================================================================

# Database configuration
DB_PATH = 'docinsight.db'
INDEX_DIR = 'indexes'
INDEX_PATH = 'indexes/'
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'faiss.index')

# Cache and index configuration
INDEX_DIMENSION = 384  # SBERT all-MiniLM-L6-v2 embedding dimension
FAISS_INDEX_TYPE = 'IndexFlatIP'  # Inner product for cosine similarity
INDEX_TYPE = 'IndexFlatIP'  # FAISS index type

# Document processing configuration
CHUNK_SIZE = int(os.getenv('DOCINSIGHT_CHUNK_SIZE', '512'))  # Tokens per chunk
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP = int(os.getenv('DOCINSIGHT_OVERLAP', '50'))  # Overlap between chunks
OVERLAP = int(os.getenv('DOCINSIGHT_OVERLAP', '50'))
SECTION_MIN_TOKENS = int(os.getenv('DOCINSIGHT_SECTION_MIN_TOKENS', '100'))  # Minimum tokens per section

# Chunking strategy
CHUNKING_STRATEGY = 'sentence'  # 'sentence' or 'sliding_window'
MIN_CHUNK_LENGTH = 50  # minimum characters per chunk
MAX_CHUNK_LENGTH = 2000  # maximum characters per chunk
MAX_CHUNKS_PER_DOC = 100

# Embedding batch size
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MODEL = MODEL_FINE_TUNED_PATH  # Reuse existing SBERT model
INGEST_BATCH_SIZE = 32

# ============================================================================
# CITATION HANDLING
# ============================================================================

# Citation masking (reduces false positives from references)
CITATION_MASKING_ENABLED = os.getenv('DOCINSIGHT_CITATION_MASKING_ENABLED', 'true').lower() == 'true'
ENABLE_CITATION_MASKING = True

# Citation patterns for masking
# Replace CITATION_PATTERNS in config.py with this:

CITATION_PATTERNS = {
    # Numeric citations: [1], [1,2,3], [1-5], (1), (1,2), etc.
    'numeric': [
        r'\[\d+(?:\s*[-,]\s*\d+)*\]',  # [1], [1,2,3], [1-5], [1, 2, 3]
        r'\(\d+(?:\s*[-,]\s*\d+)*\)',  # (1), (1,2), (1-3)
        r'\[\d+\]\s*[-–—]\s*\[\d+\]',  # [1]-[3], [1]–[5]
        r'\b\d{1,3}\s*\)',  # 1), 12), 123) - footnote style
        r'(?<!\w)\d{1,3}(?=\s*[,;.]|\s+[A-Z])',  # Superscript-style: ...findings.¹²
    ],
    
    # Author-year citations: (Smith, 2020), (Smith et al., 2020), (Smith and Jones 2020)
    'author_year': [
        # Single author: (Smith, 2020) or (Smith 2020)
        r'\([A-Z][A-Za-z\'\-]+\s*,?\s*\d{4}[a-z]?\)',
        
        # Multiple authors with 'and': (Smith and Jones, 2020)
        r'\([A-Z][A-Za-z\'\-]+(?:\s+and\s+[A-Z][A-Za-z\'\-]+)+\s*,?\s*\d{4}[a-z]?\)',
        
        # Et al.: (Smith et al., 2020) or (Smith et al. 2020)
        r'\([A-Z][A-Za-z\'\-]+\s+et\s+al\.?\s*,?\s*\d{4}[a-z]?\)',
        
        # Multiple citations: (Smith, 2020; Jones, 2021)
        r'\([A-Z][A-Za-z\'\-]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?(?:\s*[;,]\s*[A-Z][A-Za-z\'\-]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?)*\)',
        
        # Author year without parentheses: Smith (2020), Smith et al. (2020)
        r'[A-Z][A-Za-z\'\-]+(?:\s+et\s+al\.?)?\s+\(\d{4}[a-z]?\)',
    ],
    
    # Footnote and superscript markers
    'footnote': [
        r'\d{1,3}\s*\)',  # 1) at start of line (footnote)
        r'(?<=\w)[¹²³⁴⁵⁶⁷⁸⁹⁰]+',  # Superscript numbers
        r'(?<=\w)\^\d+',  # ^1, ^12 (caret notation)
    ],
    
    # Reference to bibliography sections
    'reference_marker': [
        r'\(see\s+(?:e\.g\.\s*,?\s*)?(?:ref|reference)s?\.?\s*\d+(?:\s*[-,]\s*\d+)*\)',
        r'\((?:ref|reference)s?\.?\s*\[\d+\]\)',
    ]
}

# ============================================================================
# ACADEMIC DOCUMENT PROCESSING
# ============================================================================

# Academic document section patterns
ACADEMIC_SECTIONS = {
    'abstract': ['abstract', 'summary'],
    'introduction': ['introduction', 'intro'],
    'methods': ['methods', 'methodology', 'approach', 'materials and methods'],
    'results': ['results', 'findings'],
    'discussion': ['discussion', 'analysis'],
    'conclusion': ['conclusion', 'conclusions', 'summary'],
    'references': ['references', 'bibliography', 'works cited']
}

# ============================================================================
# INGESTION & WEB SOURCES
# ============================================================================

# Ingestion source limits
ARXIV_RATE_LIMIT = 1.0  # seconds between requests
WIKI_MAX_PAGES = 50
MAX_WEB_CONCURRENCY = 5

# Language filtering
SUPPORTED_LANGUAGES = ['en']  # Language codes to keep during ingestion
LANGUAGE_DETECTION_CONFIDENCE = 0.8

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# PERFORMANCE
# ============================================================================

# Threading
MAX_WORKERS = 4

# Caching
ENABLE_CACHE = True
CACHE_DIR = '.cache'

# ============================================================================
# MODEL DOWNLOAD SETTINGS
# ============================================================================

# HuggingFace cache
HF_HOME = os.environ.get('HF_HOME', '.cache/huggingface')
TRANSFORMERS_CACHE = os.environ.get('TRANSFORMERS_CACHE', '.cache/transformers')

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

# Training report settings
SAVE_TRAINING_REPORTS = True
REPORT_FORMAT = 'json'  # Options: 'json', 'yaml', 'txt'

# Model versioning
MODEL_VERSION = '1.0'
CONFIG_VERSION = '2.0'  # Incremented for fine-tuning support

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [TEMP_DIR, OUTPUT_DIR, REPORT_DIR, INDEX_DIR, CACHE_DIR]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def get_model_info():
    """Return information about available models"""
    info = {
        'sbert': {
            'base': MODEL_BASE_NAME,
            'fine_tuned': MODEL_FINE_TUNED_PATH,
            'available': Path(MODEL_FINE_TUNED_PATH).exists()
        },
        'cross_encoder': {
            'base': CROSS_ENCODER_MODEL_NAME,
            'fine_tuned': CROSS_ENCODER_FINE_TUNED_PATH,
            'available': Path(CROSS_ENCODER_FINE_TUNED_PATH).exists()
        },
        'spacy': {
            'base': SPACY_MODEL_NAME,
            'fine_tuned': SPACY_FINE_TUNED_PATH,
            'available': Path(SPACY_FINE_TUNED_PATH).exists()
        }
    }
    return info

def validate_config():
    """Validate configuration parameters"""
    # Check weight sums
    assert abs(WEIGHT_SEMANTIC + WEIGHT_STYLO + WEIGHT_AI - 1.0) < 0.01, \
        "Fusion weights must sum to 1.0"
    
    assert abs(ALPHA_COVERAGE + BETA_SEVERITY + GAMMA_SPAN_RATIO - 1.0) < 0.01, \
        "Plagiarism factor weights must sum to 1.0"
    
    # Check thresholds
    assert 0 <= HIGH_RISK_THRESHOLD <= 1, "HIGH_RISK_THRESHOLD must be in [0, 1]"
    assert 0 <= MEDIUM_RISK_THRESHOLD <= 1, "MEDIUM_RISK_THRESHOLD must be in [0, 1]"
    assert MEDIUM_RISK_THRESHOLD < HIGH_RISK_THRESHOLD, \
        "MEDIUM_RISK_THRESHOLD must be less than HIGH_RISK_THRESHOLD"
    
    return True

# ============================================================================
# INITIALIZATION
# ============================================================================

# Create directories on import
ensure_directories()

# Validate configuration
validate_config()

# Print model availability on import
if __name__ != '__main__':
    model_info = get_model_info()
    fine_tuned_available = all(m['available'] for m in model_info.values())
    if fine_tuned_available:
        print("✓ All fine-tuned models available")
    else:
        print("⚠ Some fine-tuned models not found - will use base models")