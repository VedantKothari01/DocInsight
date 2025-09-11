"""
Domain Adaptation Module for DocInsight - Research-Focused Implementation
========================================================================

Implements SRS v0.2 requirements for domain-adapted semantic embeddings:
- Fine-tune SBERT on academic paraphrase curriculum
- Domain adaptation for academic writing patterns
- Research-quality evaluation and benchmarking

This module supports the research goal of creating conference-submission quality
improvements over baseline semantic-only systems.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Defensive imports for research environment
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # Create dummy classes for type hints when imports fail
    class SentenceTransformer:
        pass
    class InputExample:
        pass
    class DataLoader:
        pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from dataset_loaders import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DomainAdaptationConfig:
    """Configuration for domain adaptation training."""
    base_model: str = "all-MiniLM-L6-v2"
    output_model_path: str = "models/docinsight_academic_sbert"
    train_batch_size: int = 16
    eval_batch_size: int = 64
    num_epochs: int = 4
    warmup_steps: int = 1000
    evaluation_steps: int = 1000
    max_seq_length: int = 256
    use_amp: bool = True  # Automatic Mixed Precision for efficiency


class AcademicDomainAdapter:
    """
    Domain adaptation for academic paraphrase detection.
    
    Implements SRS v0.2 requirements for domain-adapted semantic embeddings
    with fine-tuning on academic paraphrase curriculum (PAWS + Quora + synthetic).
    """
    
    def __init__(self, config: DomainAdaptationConfig = None, cache_dir: str = "models"):
        self.config = config or DomainAdaptationConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model: Optional[object] = None  # SentenceTransformer when available
        self.dataset_loader = DatasetLoader()
        
    def load_base_model(self):
        """Load base sentence transformer model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers not available - domain adaptation disabled")
            return None
        
        from sentence_transformers import SentenceTransformer  # Import here for safety
        logger.info(f"Loading base model: {self.config.base_model}")
        model = SentenceTransformer(self.config.base_model)
        model.max_seq_length = self.config.max_seq_length
        return model
    
    def prepare_academic_training_data(self, target_size: int = 20000) -> List[InputExample]:
        """
        Prepare academic paraphrase training data.
        
        Creates training examples from academic paraphrase curriculum:
        - PAWS paraphrase pairs (positive examples)
        - Quora question pairs (positive examples) 
        - Cross-dataset negative examples
        - Synthetic academic paraphrases
        """
        logger.info("Preparing academic paraphrase training data...")
        
        # Load academic paraphrase curriculum
        paws_sentences = self.dataset_loader.load_paws_dataset(max_samples=target_size // 2)
        quora_sentences = self.dataset_loader.load_quora_question_pairs(max_samples=target_size // 4)
        
        # Load additional academic content for negative sampling
        wiki_sentences = self.dataset_loader.load_wikipedia_articles(
            topics=self.dataset_loader.academic_topics[:10], 
            sentences_per_topic=50
        )
        arxiv_sentences = self.dataset_loader.load_arxiv_abstracts(max_papers=200)
        
        training_examples = []
        
        # Create positive examples from PAWS (already paraphrase pairs)
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                from datasets import load_dataset
                logger.info("Creating positive paraphrase examples from PAWS...")
                
                dataset = load_dataset("paws", "labeled_final", split="train")
                positive_count = 0
                negative_count = 0
                
                for i, example in enumerate(dataset):
                    if i >= target_size // 2:
                        break
                    
                    if example['label'] == 1:  # Paraphrase pair
                        training_examples.append(InputExample(
                            texts=[example['sentence1'], example['sentence2']], 
                            label=1.0
                        ))
                        positive_count += 1
                    elif example['label'] == 0 and negative_count < positive_count:  # Non-paraphrase
                        training_examples.append(InputExample(
                            texts=[example['sentence1'], example['sentence2']], 
                            label=0.0
                        ))
                        negative_count += 1
                
                logger.info(f"Created {positive_count} positive and {negative_count} negative examples from PAWS")
                
            except Exception as e:
                logger.warning(f"Failed to load PAWS training pairs: {e}")
        
        # Create additional negative examples by cross-sampling different domains
        logger.info("Creating negative examples from cross-domain sampling...")
        import random
        
        all_academic_sentences = wiki_sentences + arxiv_sentences
        negative_pairs = 0
        target_negatives = len(training_examples)  # Balance positive/negative
        
        for i in range(min(target_negatives, len(all_academic_sentences) - 1)):
            # Sample two unrelated sentences
            idx1 = random.randint(0, len(all_academic_sentences) - 1)
            idx2 = random.randint(0, len(all_academic_sentences) - 1)
            
            if idx1 != idx2:
                training_examples.append(InputExample(
                    texts=[all_academic_sentences[idx1], all_academic_sentences[idx2]], 
                    label=0.0
                ))
                negative_pairs += 1
        
        logger.info(f"Created {negative_pairs} negative examples from cross-domain sampling")
        
        # Shuffle training examples
        random.shuffle(training_examples)
        
        logger.info(f"Total training examples: {len(training_examples)}")
        return training_examples
    
    def prepare_evaluation_data(self) -> List[InputExample]:
        """Prepare evaluation data for academic domain."""
        logger.info("Preparing academic evaluation data...")
        
        eval_examples = []
        
        # Use a subset of PAWS validation set for evaluation
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                from datasets import load_dataset
                dataset = load_dataset("paws", "labeled_final", split="validation")
                
                for i, example in enumerate(dataset):
                    if i >= 1000:  # Limit evaluation set size
                        break
                    
                    eval_examples.append(InputExample(
                        texts=[example['sentence1'], example['sentence2']], 
                        label=float(example['label'])
                    ))
                
                logger.info(f"Created {len(eval_examples)} evaluation examples")
                
            except Exception as e:
                logger.warning(f"Failed to create evaluation data: {e}")
        
        return eval_examples
    
    def fine_tune_academic_model(self, target_size: int = 20000) -> str:
        """
        Fine-tune SBERT model on academic paraphrase curriculum.
        
        Returns path to the fine-tuned model.
        """
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_TORCH:
            logger.error("sentence-transformers and torch required for fine-tuning")
            return self.config.base_model
        
        logger.info("Starting academic domain adaptation fine-tuning...")
        
        # Check if model already exists
        model_path = Path(self.config.output_model_path)
        if model_path.exists():
            logger.info(f"Fine-tuned model already exists at {model_path}")
            return str(model_path)
        
        try:
            # Load base model
            model = self.load_base_model()
            
            # Prepare training data
            train_examples = self.prepare_academic_training_data(target_size)
            if not train_examples:
                logger.warning("No training data available - using base model")
                return self.config.base_model
            
            # Create data loader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.config.train_batch_size)
            
            # Setup loss function for similarity learning
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Prepare evaluation
            eval_examples = self.prepare_evaluation_data()
            evaluator = None
            if eval_examples:
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                    eval_examples, batch_size=self.config.eval_batch_size, name='academic_eval'
                )
            
            # Configure training
            total_steps = len(train_dataloader) * self.config.num_epochs
            warmup_steps = min(self.config.warmup_steps, total_steps // 10)
            
            logger.info(f"Training configuration:")
            logger.info(f"  - Training examples: {len(train_examples)}")
            logger.info(f"  - Batch size: {self.config.train_batch_size}")
            logger.info(f"  - Epochs: {self.config.num_epochs}")
            logger.info(f"  - Total steps: {total_steps}")
            logger.info(f"  - Warmup steps: {warmup_steps}")
            
            # Create output directory
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Fine-tune model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=self.config.num_epochs,
                evaluation_steps=self.config.evaluation_steps,
                warmup_steps=warmup_steps,
                output_path=str(model_path),
                use_amp=self.config.use_amp
            )
            
            logger.info(f"Fine-tuning completed! Model saved to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            logger.info("Falling back to base model")
            return self.config.base_model
    
    def load_or_train_academic_model(self, target_size: int = 20000) -> SentenceTransformer:
        """
        Load existing fine-tuned model or train new one if needed.
        
        Returns the best available model for academic domain.
        """
        model_path = Path(self.config.output_model_path)
        
        if model_path.exists():
            logger.info(f"Loading existing fine-tuned academic model from {model_path}")
            try:
                if HAS_SENTENCE_TRANSFORMERS:
                    self.model = SentenceTransformer(str(model_path))
                    return self.model
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model: {e}")
        
        # Fine-tune new model
        logger.info("Training new academic domain-adapted model...")
        model_path_str = self.fine_tune_academic_model(target_size)
        
        # Load the fine-tuned model
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_path_str)
                logger.info("✅ Academic domain-adapted model ready")
                return self.model
            except Exception as e:
                logger.warning(f"Failed to load newly trained model: {e}")
        
        # Fallback to base model
        logger.info("Using base model as fallback")
        if HAS_SENTENCE_TRANSFORMERS:
            self.model = self.load_base_model()
        return self.model
    
    def evaluate_academic_performance(self, test_pairs: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        Evaluate model performance on academic paraphrase detection.
        
        Args:
            test_pairs: List of (sentence1, sentence2, similarity_score) tuples
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model or not HAS_NUMPY:
            logger.warning("Model or numpy not available for evaluation")
            return {}
        
        logger.info(f"Evaluating academic performance on {len(test_pairs)} test pairs")
        
        try:
            # Compute embeddings for all sentences
            sentences1 = [pair[0] for pair in test_pairs]
            sentences2 = [pair[1] for pair in test_pairs]
            true_scores = [pair[2] for pair in test_pairs]
            
            embeddings1 = self.model.encode(sentences1, convert_to_numpy=True)
            embeddings2 = self.model.encode(sentences2, convert_to_numpy=True)
            
            # Compute cosine similarities
            cosine_scores = []
            for i in range(len(embeddings1)):
                cos_sim = np.dot(embeddings1[i], embeddings2[i]) / (
                    np.linalg.norm(embeddings1[i]) * np.linalg.norm(embeddings2[i])
                )
                cosine_scores.append(cos_sim)
            
            # Compute evaluation metrics
            try:
                from scipy.stats import pearsonr, spearmanr
                pearson_r, _ = pearsonr(true_scores, cosine_scores)
                spearman_r, _ = spearmanr(true_scores, cosine_scores)
            except ImportError:
                # Fallback correlation calculation
                pearson_r = np.corrcoef(true_scores, cosine_scores)[0, 1]
                spearman_r = pearson_r  # Simplified fallback
            
            # Compute classification metrics (assuming threshold of 0.5)
            threshold = 0.5
            predicted_labels = [1 if score > threshold else 0 for score in cosine_scores]
            true_labels = [1 if score > threshold else 0 for score in true_scores]
            
            tp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 1)
            fp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 1)
            fn = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'pearson_correlation': pearson_r,
                'spearman_correlation': spearman_r,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'num_test_pairs': len(test_pairs)
            }
            
            logger.info("Academic performance evaluation results:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  - {metric}: {value:.4f}")
                else:
                    logger.info(f"  - {metric}: {value}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}


def create_academic_domain_adapter(target_size: int = 20000) -> AcademicDomainAdapter:
    """Create and configure academic domain adapter for DocInsight research."""
    config = DomainAdaptationConfig(
        output_model_path="models/docinsight_academic_sbert_v2",
        num_epochs=3,  # Reduced for faster training
        train_batch_size=16,
        warmup_steps=500
    )
    
    adapter = AcademicDomainAdapter(config)
    return adapter