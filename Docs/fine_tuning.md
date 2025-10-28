# Fine-tuning Guide for DocInsight

## Overview

DocInsight Phase 2+ includes optional fine-tuning capabilities to improve semantic similarity detection for your specific domain. This guide explains how to build datasets, train models, and evaluate results.

## Quick Start

1. **Generate synthetic training data:**
   ```bash
   python scripts/generate_synthetic_pairs.py
   ```

2. **Run the complete fine-tuning pipeline:**
   ```bash
   bash scripts/run_fine_tuning.sh
   ```

3. **Train AI-likeness classifier:**
   ```bash
   bash scripts/run_ai_likeness_training.sh
   ```

## Dataset Preparation

### Synthetic Data Generation

The system includes a synthetic data generator that creates training pairs from your existing documents:

```python
from scripts.generate_synthetic_pairs import SyntheticPairGenerator

generator = SyntheticPairGenerator()
generator.generate_dataset(positive_count=200, negative_count=200)
```

**Generated transformations include:**
- Synonym replacement
- Sentence reordering  
- Passive/active voice changes
- Minor paraphrasing
- Word order variations

### Data Requirements

- **Minimum pairs:** 50 (100+ recommended)
- **Class balance:** 30-70% positive pairs
- **Text length:** At least 3 words per text
- **Format:** CSV with columns: `text_a`, `text_b`, `label`

### Custom Dataset

You can provide your own training data in CSV format:

```csv
text_a,text_b,label
"Machine learning is powerful","ML techniques are effective",1
"The sky is blue","Databases store information",0
```

## Model Fine-tuning

### Semantic Similarity Model

Fine-tunes sentence-transformers models using contrastive learning:

```python
from fine_tuning.fine_tune_semantic import fine_tune_semantic_model

results = fine_tune_semantic_model(
    data_path="fine_tuning/data/",
    epochs=3
)
```

**Configuration options:**
- `DOCINSIGHT_FINE_TUNING_EPOCHS` (default: 3)
- `DOCINSIGHT_FINE_TUNING_BATCH_SIZE` (default: 16)  
- `DOCINSIGHT_FINE_TUNING_LR` (default: 2e-5)

### AI-likeness Classifier

Trains a classifier to detect AI-generated content:

```python
from fine_tuning.train_ai_likeness import train_ai_likeness_classifier

metrics = train_ai_likeness_classifier(
    pairs_path="fine_tuning/data/pairs.csv",
    model_type="logistic"  # or "mlp"
)
```

## Model Architecture

### Semantic Model
- **Base:** sentence-transformers (all-MiniLM-L6-v2)
- **Fine-tuning:** Cosine similarity loss
- **Output:** 384-dimensional embeddings
- **Storage:** `models/semantic_local/`

### AI-likeness Classifier
- **Features:** Stylometric (40+) + Embeddings (384)
- **Model:** Logistic Regression or MLP
- **Training:** Synthetic AI-like transformations
- **Storage:** `models/ai_likeness/`

## Evaluation Metrics

### Semantic Model
- **Similarity Score:** Spearman correlation on validation set
- **Evaluation:** Embedding similarity evaluator

### AI-likeness Classifier
- **Accuracy:** Overall classification accuracy
- **Precision/Recall:** For AI-generated content detection
- **F1 Score:** Balanced measure

## Production Deployment

### Fallback Mechanism

If fine-tuned models aren't available, the system automatically falls back to pre-trained models:

```python
# Automatic fallback in enhanced_pipeline.py
if os.path.exists('models/semantic_local/config.json'):
    model = SentenceTransformer('models/semantic_local/')
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Base model
```

### Model Validation

Before deployment, validate your models:

```bash
python scripts/run_self_check.py
```

## Limitations & Considerations

### Synthetic Data Quality
- **NOT production grade:** Simple rule-based transformations
- **Domain specific:** Best results with domain-relevant source texts
- **Size limitations:** Small datasets may not capture full complexity

### Training Scale
- **Lightweight:** Designed for college-student-level resources
- **Quick training:** 1-10 minutes on CPU for small datasets
- **Memory efficient:** Works with limited computational resources

### Model Performance
- **Baseline improvement:** 5-15% improvement over base models typical
- **Domain adaptation:** Most effective for specialized domains
- **Evaluation needed:** Always validate on held-out test data

## Troubleshooting

### Common Issues

1. **Import errors:** Install dependencies with `pip install sentence-transformers scikit-learn`
2. **Small dataset:** Increase synthetic pair generation count
3. **Poor performance:** Check data quality and class balance
4. **Memory issues:** Reduce batch size or use CPU-only training

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Transformations

Extend the synthetic data generator:

```python
class CustomPairGenerator(SyntheticPairGenerator):
    def _custom_transformation(self, text):
        # Your domain-specific transformations
        return modified_text
```

### Hyperparameter Tuning

Modify training parameters:

```bash
export DOCINSIGHT_FINE_TUNING_EPOCHS=5
export DOCINSIGHT_FINE_TUNING_BATCH_SIZE=32
export DOCINSIGHT_FINE_TUNING_LR=1e-5
```

### Multi-GPU Training

For larger datasets (future enhancement):

```python
# Not currently implemented - single GPU/CPU only
# Future: Add DataParallel support
```

## Future Enhancements

- **Back-translation:** Use translation services for paraphrase generation
- **Cross-encoder integration:** Add reranking capabilities
- **Active learning:** Iterative dataset improvement
- **Domain adaptation:** Fine-tune on specific academic fields
- **Evaluation benchmarks:** Standard plagiarism detection datasets

## Files Created

```
fine_tuning/
├── __init__.py
├── dataset_prep.py          # Dataset preparation utilities
├── fine_tune_semantic.py    # Semantic model fine-tuning
├── train_ai_likeness.py     # AI-likeness classifier training
└── data/
    ├── .gitkeep
    ├── pairs.csv            # Generated training pairs
    ├── train.csv            # Training split
    ├── val.csv              # Validation split
    ├── test.csv             # Test split
    └── metadata.json        # Dataset statistics

models/
├── semantic_local/          # Fine-tuned semantic model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── training_info.json
└── ai_likeness/            # AI-likeness classifier
    ├── ai_likeness_model.pkl
    ├── feature_schema.json
    └── metrics.json
```