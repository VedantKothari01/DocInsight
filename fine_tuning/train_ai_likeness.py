"""
AI-likeness classifier training module

Trains a classifier to detect AI-generated content by combining stylometric features
with semantic embeddings using logistic regression or shallow MLP.
"""

import os
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

from config import AI_LIKENESS_MODEL_PATH, MODEL_BASE_NAME, MODEL_FINE_TUNED_PATH
from stylometry.features import StylemetryFeatureExtractor

logger = logging.getLogger(__name__)

# Try to import sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available for embedding extraction")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class AILikenessTrainer:
    """Trains AI-likeness detection classifier"""
    
    def __init__(self, model_path: str = None):
        """Initialize trainer
        
        Args:
            model_path: Path to save trained model
        """
        self.model_path = model_path or AI_LIKENESS_MODEL_PATH
        self.embedding_model = None
        self.stylometry_extractor = StylemetryFeatureExtractor()
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Ensure output directory exists
        os.makedirs(self.model_path, exist_ok=True)
    
    def load_embedding_model(self) -> None:
        """Load sentence transformer model for embeddings"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, skipping embedding features")
            return
        
        try:
            # Try to load fine-tuned model first
            fine_tuned_path = MODEL_FINE_TUNED_PATH
            if os.path.exists(os.path.join(fine_tuned_path, 'config.json')):
                logger.info(f"Loading fine-tuned model from {fine_tuned_path}")
                self.embedding_model = SentenceTransformer(fine_tuned_path)
            else:
                logger.info(f"Loading base model: {MODEL_BASE_NAME}")
                self.embedding_model = SentenceTransformer(MODEL_BASE_NAME)
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None
    
    def generate_synthetic_ai_data(self, human_texts: List[str], count: int = 200) -> List[Tuple[str, int]]:
        """Generate synthetic AI-like text samples for training
        
        Args:
            human_texts: List of human-written texts
            count: Number of synthetic samples to generate
            
        Returns:
            List of (text, label) tuples where label=1 for AI-like
        """
        synthetic_samples = []
        
        # Simple AI-like transformations (NOT production grade)
        for _ in range(count):
            if not human_texts:
                break
                
            original = np.random.choice(human_texts)
            
            # Apply AI-like transformations
            ai_like = self._apply_ai_transformations(original)
            
            if ai_like != original:
                synthetic_samples.append((ai_like, 1))
        
        logger.info(f"Generated {len(synthetic_samples)} synthetic AI-like samples")
        return synthetic_samples
    
    def _apply_ai_transformations(self, text: str) -> str:
        """Apply transformations that make text appear more AI-like"""
        # Simple heuristic transformations (clearly marked as NOT production grade)
        
        # 1. Make text more formal/verbose
        transformations = [
            ("I think", "It is my belief that"),
            ("I believe", "It is my considered opinion that"),
            ("good", "beneficial and advantageous"),
            ("bad", "detrimental and problematic"),
            ("important", "of paramount significance"),
            ("shows", "demonstrates and illustrates"),
            ("uses", "utilizes and employs"),
            ("helps", "facilitates and assists in"),
        ]
        
        modified_text = text
        for original, replacement in transformations:
            if original in modified_text.lower():
                # Apply transformation with some probability
                if np.random.random() < 0.3:
                    modified_text = modified_text.replace(original, replacement)
        
        # 2. Add hedging language (AI often hedges)
        hedges = ["arguably", "potentially", "presumably", "apparently", "seemingly"]
        if np.random.random() < 0.4:
            hedge = np.random.choice(hedges)
            sentences = modified_text.split('. ')
            if sentences:
                # Add hedge to random sentence
                idx = np.random.randint(len(sentences))
                sentences[idx] = f"{hedge}, {sentences[idx].lower()}"
                modified_text = '. '.join(sentences)
        
        # 3. Make sentence structure more uniform
        if np.random.random() < 0.3:
            # Simple sentence restructuring
            modified_text = modified_text.replace(", and", ". Additionally,")
            modified_text = modified_text.replace(", but", ". However,")
        
        return modified_text
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract combined features (stylometric + embeddings) from texts
        
        Args:
            texts: List of text samples
            
        Returns:
            Feature matrix
        """
        features_list = []
        
        for text in texts:
            feature_vector = []
            
            # 1. Stylometric features
            stylometric_features = self.stylometry_extractor.compute_baseline_features(text)
            
            # Convert to list in consistent order
            for feature_name in self.stylometry_extractor.get_feature_names():
                feature_vector.append(stylometric_features.get(feature_name, 0.0))
            
            # 2. Embedding features (if available)
            if self.embedding_model is not None:
                try:
                    embedding = self.embedding_model.encode([text])[0]
                    feature_vector.extend(embedding.tolist())
                except Exception as e:
                    logger.debug(f"Error extracting embedding: {e}")
                    # Fill with zeros if embedding extraction fails
                    feature_vector.extend([0.0] * 384)  # Default SBERT dimension
            
            features_list.append(feature_vector)
        
        # Store feature names for later use
        if not self.feature_names:
            self.feature_names = self.stylometry_extractor.get_feature_names()
            if self.embedding_model is not None:
                self.feature_names.extend([f'embed_{i}' for i in range(384)])
        
        return np.array(features_list)
    
    def prepare_training_data(self, pairs_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data by extracting features and generating AI samples
        
        Args:
            pairs_path: Path to pairs CSV file
            
        Returns:
            Tuple of (features, labels)
        """
        # Load human text samples from pairs file
        df = pd.read_csv(pairs_path)
        
        # Extract unique texts (assuming they are human-written)
        human_texts = []
        human_texts.extend(df['text_a'].tolist())
        human_texts.extend(df['text_b'].tolist())
        human_texts = list(set(human_texts))  # Remove duplicates
        
        logger.info(f"Loaded {len(human_texts)} unique human text samples")
        
        # Generate synthetic AI samples
        ai_samples = self.generate_synthetic_ai_data(human_texts, count=len(human_texts))
        
        # Combine data
        all_texts = []
        all_labels = []
        
        # Add human samples (label=0)
        for text in human_texts[:len(ai_samples)]:  # Balance the dataset
            all_texts.append(text)
            all_labels.append(0)
        
        # Add AI samples (label=1)
        for text, label in ai_samples:
            all_texts.append(text)
            all_labels.append(label)
        
        logger.info(f"Total samples: {len(all_texts)} (Human: {sum(1 for l in all_labels if l == 0)}, AI: {sum(1 for l in all_labels if l == 1)})")
        
        # Extract features
        features = self.extract_features(all_texts)
        labels = np.array(all_labels)
        
        return features, labels
    
    def train_classifier(self, features: np.ndarray, labels: np.ndarray, 
                        model_type: str = 'logistic') -> Dict[str, Any]:
        """Train AI-likeness classifier
        
        Args:
            features: Feature matrix
            labels: Target labels (0=human, 1=AI)
            model_type: Type of classifier ('logistic' or 'mlp')
            
        Returns:
            Training metrics dictionary
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Choose classifier
        if model_type == 'logistic':
            classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'mlp':
            classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                random_state=42, 
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with scaling
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        # Train model
        logger.info(f"Training {model_type} classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'model_type': model_type,
            'feature_count': features.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Training completed - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        return metrics
    
    def save_model(self, metrics: Dict[str, Any]) -> None:
        """Save trained model and metadata
        
        Args:
            metrics: Training metrics to save
        """
        # Save model
        model_file = os.path.join(self.model_path, 'ai_likeness_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Save feature schema
        feature_schema = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'stylometric_features': len(self.stylometry_extractor.get_feature_names()),
            'embedding_features': len(self.feature_names) - len(self.stylometry_extractor.get_feature_names()),
            'embedding_model': MODEL_FINE_TUNED_PATH if os.path.exists(os.path.join(MODEL_FINE_TUNED_PATH, 'config.json')) else MODEL_BASE_NAME
        }
        
        schema_file = os.path.join(self.model_path, 'feature_schema.json')
        with open(schema_file, 'w') as f:
            json.dump(feature_schema, f, indent=2)
        
        # Save metrics
        metrics_file = os.path.join(self.model_path, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"  Model file: {model_file}")
        logger.info(f"  Feature schema: {schema_file}")
        logger.info(f"  Metrics: {metrics_file}")


def train_ai_likeness_classifier(pairs_path: str = "fine_tuning/data/pairs.csv",
                                model_type: str = 'logistic') -> Dict[str, Any]:
    """Convenience function to train AI-likeness classifier
    
    Args:
        pairs_path: Path to pairs CSV file
        model_type: Type of classifier to train
        
    Returns:
        Training metrics dictionary
    """
    try:
        trainer = AILikenessTrainer()
        
        # Load embedding model
        trainer.load_embedding_model()
        
        # Prepare data
        features, labels = trainer.prepare_training_data(pairs_path)
        
        if len(features) < 50:
            raise ValueError(f"Insufficient training data: {len(features)} samples")
        
        # Train classifier
        metrics = trainer.train_classifier(features, labels, model_type)
        
        # Save model
        trainer.save_model(metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"AI-likeness training failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Train AI-likeness classifier')
    parser.add_argument('--pairs', default='fine_tuning/data/pairs.csv',
                       help='Path to pairs CSV file')
    parser.add_argument('--model-type', choices=['logistic', 'mlp'], default='logistic',
                       help='Type of classifier to train')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🤖 Starting AI-likeness classifier training...")
    
    # Check if pairs file exists
    if not os.path.exists(args.pairs):
        print(f"❌ Pairs file not found: {args.pairs}")
        print("💡 Run 'python scripts/generate_synthetic_pairs.py' first")
        return
    
    # Train classifier
    metrics = train_ai_likeness_classifier(args.pairs, args.model_type)
    
    if 'error' not in metrics:
        print("✅ AI-likeness classifier training completed!")
        print(f"📁 Model saved to: {AI_LIKENESS_MODEL_PATH}")
        print(f"📊 Accuracy: {metrics['accuracy']:.3f}")
        print(f"📊 Precision: {metrics['precision']:.3f}")
        print(f"📊 Recall: {metrics['recall']:.3f}")
        print(f"📊 F1 Score: {metrics['f1_score']:.3f}")
    else:
        print(f"❌ Training failed: {metrics['error']}")


if __name__ == "__main__":
    main()