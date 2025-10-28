"""
Fine-tuning module for semantic similarity models

Provides lightweight fine-tuning of sentence-transformers models for improved
semantic similarity detection with fallback to pre-trained models.
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_BASE_NAME, MODEL_FINE_TUNED_PATH,
    FINE_TUNING_EPOCHS, FINE_TUNING_BATCH_SIZE, FINE_TUNING_LEARNING_RATE,
    FORCE_RETRAIN
)

logger = logging.getLogger(__name__)

# Try to import sentence-transformers with fallback
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available, fine-tuning will be skipped")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticModelFineTuner:
    """Fine-tunes sentence transformer models for semantic similarity"""
    
    def __init__(self, base_model_name: str = None, output_path: str = None):
        """Initialize fine-tuner
        
        Args:
            base_model_name: Name of base sentence transformer model
            output_path: Path to save fine-tuned model
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package required for fine-tuning")
        
        self.base_model_name = base_model_name or MODEL_BASE_NAME
        self.output_path = output_path or MODEL_FINE_TUNED_PATH
        self.model = None
        self.training_stats = {}
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
    
    def load_base_model(self) -> None:
        """Load the base sentence transformer model"""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            self.model = SentenceTransformer(self.base_model_name)
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def prepare_training_data(self, data_files: Dict[str, str]) -> Tuple[List, Optional[EmbeddingSimilarityEvaluator]]:
        """Prepare training data from CSV files
        
        Args:
            data_files: Dictionary with 'train', 'val', 'test' file paths
            
        Returns:
            Tuple of (training_examples, evaluator)
        """
        # Load training data
        train_df = pd.read_csv(data_files['train'])
        logger.info(f"Loaded {len(train_df)} training examples")
        
        # Create training examples
        train_examples = []
        for _, row in train_df.iterrows():
            # For sentence-transformers, we need to convert to InputExample
            # Label should be float (similarity score 0.0-1.0)
            score = float(row['label'])  # 0 or 1 -> 0.0 or 1.0
            example = InputExample(texts=[row['text_a'], row['text_b']], label=score)
            train_examples.append(example)
        
        # Create evaluator if validation data exists
        evaluator = None
        if 'val' in data_files and os.path.exists(data_files['val']):
            val_df = pd.read_csv(data_files['val'])
            logger.info(f"Loaded {len(val_df)} validation examples")
            
            # Create evaluation data
            sentences1 = val_df['text_a'].tolist()
            sentences2 = val_df['text_b'].tolist()
            scores = val_df['label'].astype(float).tolist()
            
            evaluator = EmbeddingSimilarityEvaluator(
                sentences1, sentences2, scores, 
                name='validation',
                show_progress_bar=True
            )
        
        return train_examples, evaluator
    
    def fine_tune(self, 
                  data_files: Dict[str, str], 
                  epochs: int = None, 
                  batch_size: int = None,
                  learning_rate: float = None,
                  warmup_steps: int = None) -> Dict[str, any]:
        """Fine-tune the model on training data
        
        Args:
            data_files: Dictionary with dataset file paths
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            warmup_steps: Number of warmup steps
            
        Returns:
            Dictionary with training statistics
        """
        if self.model is None:
            self.load_base_model()
        
        # Set training parameters
        epochs = epochs or FINE_TUNING_EPOCHS
        batch_size = batch_size or FINE_TUNING_BATCH_SIZE
        learning_rate = learning_rate or FINE_TUNING_LEARNING_RATE
        
        # Prepare data
        train_examples, evaluator = self.prepare_training_data(data_files)
        
        if not train_examples:
            raise ValueError("No training examples loaded")
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Calculate warmup steps (10% of training steps)
        if warmup_steps is None:
            warmup_steps = int(len(train_dataloader) * epochs * 0.1)
        
        # Define loss function (cosine similarity loss for binary classification)
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        
        logger.info(f"Starting fine-tuning with {len(train_examples)} examples")
        logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # Training configuration
        evaluation_steps = max(1, len(train_dataloader) // 4)  # Evaluate 4 times per epoch
        save_best_model = True
        
        # Store training configuration
        self.training_stats = {
            'base_model': self.base_model_name,
            'training_examples': len(train_examples),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'warmup_steps': warmup_steps,
            'evaluation_steps': evaluation_steps
        }
        
        try:
            # Fine-tune the model
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=epochs,
                evaluation_steps=evaluation_steps,
                warmup_steps=warmup_steps,
                output_path=self.output_path,
                save_best_model=save_best_model,
                show_progress_bar=True
            )
            
            logger.info("Fine-tuning completed successfully")
            self.training_stats['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            self.training_stats['status'] = 'failed'
            self.training_stats['error'] = str(e)
            raise
        
        return self.training_stats
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, float]:
        """Evaluate fine-tuned model on test data
        
        Args:
            test_data_path: Path to test CSV file
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not os.path.exists(test_data_path):
            logger.warning(f"Test data not found: {test_data_path}")
            return {}
        
        # Load fine-tuned model
        if os.path.exists(os.path.join(self.output_path, 'config.json')):
            model = SentenceTransformer(self.output_path)
        else:
            logger.warning("Fine-tuned model not found, using base model")
            model = self.model or SentenceTransformer(self.base_model_name)
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Evaluating on {len(test_df)} test examples")
        
        # Create evaluator
        sentences1 = test_df['text_a'].tolist()
        sentences2 = test_df['text_b'].tolist()
        scores = test_df['label'].astype(float).tolist()
        
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1, sentences2, scores,
            name='test_evaluation'
        )
        
        # Run evaluation (catch correlation edge cases -> NaN)
        raw_score = evaluator(model, output_path=self.output_path)
        try:
            display_score = float(raw_score) if raw_score == raw_score else 0.0  # NaN check
        except Exception:
            display_score = 0.0
        metrics = {
            'test_score': display_score,
            'test_examples': len(test_df)
        }
        logger.info(f"Test evaluation score: {display_score:.4f}")
        return metrics
    
    def save_training_info(self, additional_metrics: Dict = None) -> None:
        """Save training information and metrics
        
        Args:
            additional_metrics: Additional metrics to save
        """
        info = self.training_stats.copy()
        
        if additional_metrics:
            info.update(additional_metrics)
        
        # Add model info
        info_path = os.path.join(self.output_path, 'training_info.json')
        
        try:
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            logger.info(f"Training info saved to {info_path}")
        except Exception as e:
            logger.warning(f"Could not save training info: {e}")
    
    def check_model_exists(self) -> bool:
        """Check if fine-tuned model already exists and capture timestamp."""
        config_path = os.path.join(self.output_path, 'config.json')
        if os.path.exists(config_path):
            try:
                stat = os.stat(config_path)
                self.training_stats['last_modified'] = stat.st_mtime
            except OSError:
                pass
            return True
        return False


def fine_tune_semantic_model(data_path: str = "fine_tuning/data/train.csv",
                            epochs: int = None) -> Dict[str, any]:
    """Convenience function to fine-tune semantic model
    
    Args:
        data_path: Path to training data CSV or directory containing splits
        epochs: Number of training epochs
        
    Returns:
        Training statistics dictionary
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers package not available")
        return {'status': 'failed', 'error': 'sentence-transformers not installed'}
    
    try:
        # Determine data files
        if os.path.isdir(data_path):
            data_dir = data_path
        else:
            data_dir = os.path.dirname(data_path)
        
        data_files = {
            'train': os.path.join(data_dir, 'train.csv'),
            'val': os.path.join(data_dir, 'val.csv'),
            'test': os.path.join(data_dir, 'test.csv')
        }
        
        # Check if training data exists
        if not os.path.exists(data_files['train']):
            raise FileNotFoundError(f"Training data not found: {data_files['train']}")
        
        # Initialize fine-tuner
        fine_tuner = SemanticModelFineTuner()
        
        # Check if model already exists
        if fine_tuner.check_model_exists() and not FORCE_RETRAIN:
            logger.info("Fine-tuned model already exists and FORCE_RETRAIN is False - skipping")
            return {
                'status': 'already_exists',
                'model_path': fine_tuner.output_path,
                'last_modified': fine_tuner.training_stats.get('last_modified')
            }
        elif fine_tuner.check_model_exists() and FORCE_RETRAIN:
            logger.info("FORCE_RETRAIN=True -> retraining over existing model")
        
        # Fine-tune model
        training_stats = fine_tuner.fine_tune(data_files, epochs=epochs)
        
        # Evaluate if test data exists
        if os.path.exists(data_files['test']):
            eval_metrics = fine_tuner.evaluate_model(data_files['test'])
            training_stats.update(eval_metrics)
        
        # Save training information
        fine_tuner.save_training_info()
        
        return training_stats
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Fine-tune semantic similarity model')
    parser.add_argument('--data', default='fine_tuning/data/train.csv', 
                       help='Path to training data CSV or directory')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers package not available")
        print("💡 Install with: pip install sentence-transformers")
        return
    
    print("🔧 Starting semantic model fine-tuning...")
    
    # Fine-tune model
    results = fine_tune_semantic_model(
        data_path=args.data,
        epochs=args.epochs
    )
    
    if results['status'] == 'completed':
        print("✅ Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {MODEL_FINE_TUNED_PATH}")
        print(f"📊 Training examples: {results.get('training_examples', 'N/A')}")
        ts = results.get('test_score')
        if isinstance(ts, (int, float)):
            print(f"🎯 Test score: {ts:.4f}")
        else:
            print(f"🎯 Test score: {ts}")
    elif results['status'] == 'already_exists':
        print("ℹ️ Fine-tuned model already exists")
        print(f"📁 Model location: {results['model_path']}")
    else:
        print(f"❌ Fine-tuning failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()