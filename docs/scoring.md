# Unified Scoring System for DocInsight

## Overview

DocInsight Phase 2+ introduces a unified scoring system that combines semantic similarity, stylometric analysis, and AI-likeness detection into a single document-level originality score.

## Scoring Components

### 1. Semantic Risk Score (0-1)

Measures similarity to known sources in the corpus:

```python
semantic_risk = (
    0.5 * average_similarity +      # Mean sentence similarity
    0.3 * high_risk_ratio +         # Proportion of high-risk sentences  
    0.2 * medium_risk_ratio         # Proportion of medium-risk sentences
)
```

**Interpretation:**
- `0.0-0.3`: Low semantic similarity
- `0.3-0.6`: Moderate similarity concerns
- `0.6-1.0`: High similarity to existing sources

### 2. Stylometric Deviation Score (0-1)

Measures writing style deviation from academic norms:

```python
# Baseline expectations for academic writing
baselines = {
    'type_token_ratio': 0.5,        # Lexical diversity
    'avg_sentence_length': 20.0,    # Sentence complexity
    'avg_word_length': 5.0,         # Vocabulary sophistication
    'punctuation_density': 0.1,     # Punctuation usage
    'stopword_ratio': 0.4,          # Function word frequency
    'complexity_score': 0.6         # Overall text complexity
}
```

**Features analyzed:**
- **Lexical diversity:** Type-token ratio, unique words
- **Syntactic complexity:** Sentence length variance
- **Vocabulary:** Average word length, function words
- **Style markers:** Punctuation patterns, complexity metrics

### 3. AI-likeness Probability (0-1)

Estimates probability that text is AI-generated:

```python
# Combined feature vector
features = stylometric_features + embedding_features

# Trained classifier (logistic regression or MLP)
ai_probability = classifier.predict_proba(features)[1]
```

**Detection signals:**
- **Stylometric patterns:** Uniform sentence structure, hedging language
- **Semantic consistency:** Coherent but generic content
- **Linguistic markers:** Formal vocabulary, balanced arguments

## Unified Overall Score

### Weighted Combination

```python
overall_score = (
    w_semantic * semantic_risk +
    w_stylometry * stylometric_deviation +  
    w_ai * ai_likeness_probability
)
```

### Default Weights

```python
# Rationale for weight selection:
WEIGHT_SEMANTIC = 0.6    # Primary plagiarism signal
WEIGHT_STYLO = 0.25      # Writing pattern analysis  
WEIGHT_AI = 0.15         # AI-generated content detection
```

**Weight rationale:**
- **Semantic (60%):** Direct similarity is strongest plagiarism indicator
- **Stylometric (25%):** Style changes suggest authorship issues
- **AI-likeness (15%):** Emerging concern for sophisticated plagiarism

### Environment Overrides

Configure weights via environment variables:

```bash
export DOCINSIGHT_W_SEMANTIC=0.7
export DOCINSIGHT_W_STYLO=0.2  
export DOCINSIGHT_W_AI=0.1
```

Weights are automatically normalized to sum to 1.0.

## Suspicious Sections Detection

### Section-level Analysis

The system identifies the most suspicious document sections:

```python
suspicious_sections = [
    {
        "section": "Introduction",
        "score_breakdown": {
            "semantic_risk": 0.8,
            "stylometric_deviation": 0.6, 
            "ai_likeness_prob": 0.7,
            "overall_score": 0.74
        },
        "reason": "High AI-likeness & paraphrase density",
        "token_count": 245
    }
]
```

### Suspicion Criteria

Sections are flagged based on:
- **Overall score > 0.3:** Moderate concern threshold
- **Top N sections:** Configurable via `SUSPICIOUS_SECTION_COUNT`
- **Reason classification:** Primary contributing factor

### Reason Categories

- `"High semantic similarity to known sources"`
- `"Significant writing style deviation"`  
- `"High probability of AI-generated content"`
- `"High AI-likeness & paraphrase density"`
- `"Low overall risk"`

## Document Summary JSON Structure

### Complete Output Format

```json
{
  "document_summary": {
    "overall_score": 0.42,
    "weights": {
      "semantic": 0.6,
      "stylometry": 0.25, 
      "ai_likeness": 0.15
    },
    "semantic_risk": 0.35,
    "stylometric_deviation": 0.62,
    "ai_likeness_prob": 0.28,
    "suspicious_sections": [
      {
        "section": "Methods",
        "score_breakdown": {
          "semantic_risk": 0.45,
          "stylometric_deviation": 0.78,
          "ai_likeness_prob": 0.31,
          "overall_score": 0.52
        },
        "reason": "Significant writing style deviation",
        "token_count": 189
      }
    ],
    "interpretation": "Medium Risk - Some originality concerns identified",
    "confidence": 0.78
  },
  "sentence_analyses": [
    // Existing sentence-level analysis preserved
  ]
}
```

### Backward Compatibility

- **Existing reports:** Load without `document_summary` field
- **Sentence analyses:** Preserved unchanged for detailed review
- **API compatibility:** All existing endpoints work unchanged

## Score Interpretation

### Risk Levels

| Score Range | Risk Level | Interpretation |
|-------------|------------|----------------|
| 0.0 - 0.2 | Very Low | Content appears original |
| 0.2 - 0.4 | Low Risk | Minor originality concerns |
| 0.4 - 0.6 | Medium Risk | Some originality concerns identified |
| 0.6 - 0.8 | High Risk | Multiple originality concerns detected |
| 0.8 - 1.0 | Very High | Strong indicators of non-original content |

### Confidence Score

```python
confidence = mean([
    min(sentence_count / 50.0, 1.0),    # More sentences = higher confidence
    min(token_count / 500.0, 1.0),      # Longer text = higher confidence  
    0.8 if ai_model_available else 0.6  # Model availability
])
```

## Implementation Example

### Basic Usage

```python
from scoring.aggregate import compute_unified_score

# Required inputs
semantic_results = [...]  # From sentence-level analysis
stylometric_features = extract_stylometric_features(text)

# Optional inputs  
sections = parse_academic_document(text)
ai_probability = predict_ai_likeness(text)

# Compute unified score
summary = compute_unified_score(
    semantic_results,
    stylometric_features, 
    sections=sections,
    ai_likeness_prob=ai_probability
)

print(f"Overall risk: {summary['overall_score']:.2f}")
print(f"Interpretation: {summary['interpretation']}")
```

### Integration with Existing Pipeline

```python
# In enhanced_pipeline.py
def analyze_document_enhanced(text):
    # Phase 1: Existing analysis
    sentence_results = analyze_sentences(text)
    
    # Phase 2+: Enhanced analysis
    stylometric_features = extract_stylometric_features(text)
    sections = parse_academic_document(text)
    ai_prob = predict_ai_likeness(text) if ai_model_available else None
    
    # Unified scoring
    document_summary = compute_unified_score(
        sentence_results, stylometric_features, sections, ai_prob
    )
    
    return {
        'sentence_analyses': sentence_results,
        'document_summary': document_summary
    }
```

## Calibration and Tuning

### Threshold Adjustment

Adjust risk thresholds based on your use case:

```python
# Conservative (flag more content)
WEIGHT_SEMANTIC = 0.7
WEIGHT_STYLO = 0.2
WEIGHT_AI = 0.1

# Balanced (default)
WEIGHT_SEMANTIC = 0.6  
WEIGHT_STYLO = 0.25
WEIGHT_AI = 0.15

# Permissive (flag less content)
WEIGHT_SEMANTIC = 0.5
WEIGHT_STYLO = 0.3
WEIGHT_AI = 0.2
```

### Domain Adaptation

For specific academic fields:

```python
# STEM fields (more technical vocabulary)
stylometric_baselines['avg_word_length'] = 6.0
stylometric_baselines['complexity_score'] = 0.8

# Humanities (more varied writing styles)  
stylometric_baselines['type_token_ratio'] = 0.6
stylometric_baselines['sentence_length_variance'] = 25.0
```

## Performance Considerations

### Computational Complexity

- **Stylometric features:** O(n) where n = text length
- **Embedding computation:** O(sentences) with model call overhead
- **Unified scoring:** O(1) lightweight aggregation
- **Section analysis:** O(sections) typically < 10

### Memory Usage

- **Feature extraction:** ~1KB per document
- **Model storage:** ~400MB (sentence-transformers + classifier)
- **Runtime memory:** ~50MB during analysis

### Optimization Tips

1. **Cache embeddings:** Reuse for multiple analyses
2. **Batch processing:** Process multiple documents together  
3. **Lazy evaluation:** Skip sentence details for large documents
4. **Model quantization:** Reduce memory footprint for deployment

## Validation and Testing

### Unit Tests

```python
def test_unified_scoring():
    semantic_results = [{'similarity_score': 0.8, 'risk_level': 'HIGH'}]
    stylometric_features = {'type_token_ratio': 0.7, 'avg_sentence_length': 25}
    
    summary = compute_unified_score(semantic_results, stylometric_features)
    
    assert 0 <= summary['overall_score'] <= 1
    assert summary['semantic_risk'] > 0
    assert 'interpretation' in summary
```

### Integration Tests

```python
def test_end_to_end_scoring():
    text = "Sample academic document for testing..."
    
    # Full pipeline
    result = analyze_document_enhanced(text)
    
    assert 'document_summary' in result
    assert 'sentence_analyses' in result
    assert result['document_summary']['confidence'] > 0
```

## Error Handling

### Missing Models

```python
# Graceful degradation when AI model unavailable
if ai_likeness_prob is None:
    logger.warning("AI-likeness model not available, adjusting weights")
    # Automatically renormalize weights
    adjusted_weights = normalize_weights(semantic_weight, stylometric_weight)
```

### Invalid Input

```python
# Handle edge cases
if not semantic_results:
    semantic_risk = 0.0
    
if not stylometric_features:
    stylometric_deviation = 0.0
    
if token_count < minimum_threshold:
    confidence *= 0.5  # Reduce confidence for short texts
```

## Files and Dependencies

### Core Files

- `scoring/aggregate.py` - Main unified scoring implementation
- `stylometry/features.py` - Stylometric feature extraction
- `config.py` - Weight configuration and environment overrides

### Dependencies

- `numpy` - Numerical computations
- `scikit-learn` - AI-likeness classifier
- `sentence-transformers` - Semantic embeddings (optional)

### Configuration

```python
# In config.py
WEIGHT_SEMANTIC = float(os.getenv('DOCINSIGHT_W_SEMANTIC', '0.6'))
WEIGHT_STYLO = float(os.getenv('DOCINSIGHT_W_STYLO', '0.25'))  
WEIGHT_AI = float(os.getenv('DOCINSIGHT_W_AI', '0.15'))
SUSPICIOUS_SECTION_COUNT = int(os.getenv('DOCINSIGHT_SUSPICIOUS_SECTIONS', '3'))
```