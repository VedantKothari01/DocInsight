"""
fine_tuner.py
Enhanced version with Cross-Encoder and spaCy fine-tuning
Includes automatic config.py parameter updates based on training performance for ALL models
"""

import os
import re
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import shutil

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, CrossEncoder
from torch.utils.data import DataLoader
import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import torch

from config import (
    MODEL_FINE_TUNED_PATH,
    MODEL_BASE_NAME,
    FINE_TUNING_EPOCHS,
    FINE_TUNING_BATCH_SIZE,
    FINE_TUNING_LEARNING_RATE,
    EXTENDED_CORPUS_ENABLED,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_EPOCHS,
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_LEARNING_RATE,
    SPACY_MODEL_NAME,
    SPACY_TRAINING_ITERATIONS,
    SPACY_DROPOUT,
    RERANK_WEIGHT,
    SEMANTIC_WEIGHT
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConfigUpdater:
    """Updates config.py parameters based on training results."""
    
    @staticmethod
    def update_config_file(updates: Dict[str, any], config_path: str = "config.py"):
        """Update config.py with new parameter values."""
        try:
            # Backup original
            backup_path = f"{config_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(config_path, backup_path)
            logger.info(f"✓ Backup created: {backup_path}")
            
            # Read file
            with open(config_path, 'r') as f:
                lines = f.readlines()
            
            # Update lines
            new_lines = []
            for line in lines:
                new_line = line
                for param, value in updates.items():
                    if line.strip().startswith(f"{param} ="):
                        # Format value
                        if isinstance(value, str):
                            val_str = f"'{value}'"
                        else:
                            val_str = str(value)
                        
                        # Preserve comment if exists
                        if '#' in line:
                            parts = line.split('#', 1)
                            new_line = f"{param} = {val_str}  # {parts[1]}"
                        else:
                            new_line = f"{param} = {val_str}\n"
                        break
                new_lines.append(new_line)
            
            # Write back
            with open(config_path, 'w') as f:
                f.writelines(new_lines)
            
            logger.info(f"✓ Updated config.py:")
            for param, value in updates.items():
                logger.info(f"  {param} = {value}")
                
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
    
    @staticmethod
    def save_report(metrics: Dict, path: str = "training_reports/"):
        """Save training report as JSON."""
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(path, f"report_{timestamp}.json")
        
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Report saved: {report_file}")


class CrossEncoderFineTuner:
    """Fine-tunes Cross-Encoder model for reranking."""
    
    def __init__(self, base_model: str = None, output_path: str = "models/cross_encoder_finetuned"):
        self.base_model = base_model or CROSS_ENCODER_MODEL_NAME
        self.output_path = output_path
        self.model = None
        
    def prepare_data(self, train_examples: List[InputExample]) -> Tuple[List, List]:
        """Convert sentence-transformer examples to cross-encoder format."""
        train_samples = []
        val_samples = []
        
        # Convert to (text1, text2, label) format
        for ex in train_examples:
            sample = {
                'sentence1': ex.texts[0],
                'sentence2': ex.texts[1],
                'label': float(ex.label)
            }
            
            # 90-10 split
            if random.random() < 0.9:
                train_samples.append(sample)
            else:
                val_samples.append(sample)
        
        logger.info(f"Cross-Encoder: {len(train_samples)} train, {len(val_samples)} val")
        return train_samples, val_samples
    
    def fine_tune(self, train_examples: List[InputExample], 
                  epochs: int = None, batch_size: int = None, 
                  learning_rate: float = None):
        """Fine-tune cross-encoder model."""
        # Use config values if not specified
        epochs = epochs or CROSS_ENCODER_EPOCHS
        batch_size = batch_size or CROSS_ENCODER_BATCH_SIZE
        learning_rate = learning_rate or CROSS_ENCODER_LEARNING_RATE
        
        logger.info("=" * 60)
        logger.info("FINE-TUNING CROSS-ENCODER")
        logger.info("=" * 60)
        logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        try:
            # Load model
            logger.info(f"Loading {self.base_model}...")
            self.model = CrossEncoder(self.base_model, num_labels=1)
            
            # Prepare data - returns list of InputSample for cross-encoder
            train_samples, val_samples = self.prepare_data(train_examples)
            
            if not train_samples:
                logger.warning("No training data for cross-encoder")
                return None
            
            # Train - CrossEncoder expects InputSample format
            logger.info(f"Training: {len(train_samples)} samples × {epochs} epochs")
            
            # Convert dict samples to CrossEncoder InputSample format
            from sentence_transformers import InputExample as CEInputExample
            ce_train_samples = [
                CEInputExample(texts=[s['sentence1'], s['sentence2']], label=s['label'])
                for s in train_samples
            ]
            
            self.model.fit(
                train_dataloader=DataLoader(ce_train_samples, shuffle=True, batch_size=batch_size),
                epochs=epochs,
                warmup_steps=100,
                output_path=self.output_path,
                optimizer_params={'lr': learning_rate}
            )
            
            logger.info(f"✓ Cross-Encoder saved to {self.output_path}")
            
            # Evaluate
            if val_samples:
                eval_results = self._evaluate(val_samples)
                # Add hyperparameters to results
                eval_results['hyperparameters'] = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
                return eval_results
            
        except Exception as e:
            logger.error(f"Cross-encoder fine-tuning failed: {e}")
            return None
    
    def _evaluate(self, val_samples: List[Dict]) -> Dict:
        """Evaluate cross-encoder performance and compute optimal weights."""
        logger.info("Evaluating Cross-Encoder...")
        
        pairs = [[s['sentence1'], s['sentence2']] for s in val_samples]
        labels = np.array([s['label'] for s in val_samples])
        
        # Predict
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Compute metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import spearmanr, pearsonr
        
        mse = mean_squared_error(labels, scores)
        mae = mean_absolute_error(labels, scores)
        spearman, _ = spearmanr(labels, scores)
        pearson, _ = pearsonr(labels, scores)
        
        # Compute optimal rerank weight based on correlation strength
        # Higher correlation = more weight to cross-encoder
        if spearman > 0.7:
            optimal_rerank_weight = 0.65
        elif spearman > 0.6:
            optimal_rerank_weight = 0.60
        elif spearman > 0.5:
            optimal_rerank_weight = 0.55
        else:
            optimal_rerank_weight = 0.50
        
        optimal_semantic_weight = 1.0 - optimal_rerank_weight
        
        results = {
            'mse': float(mse),
            'mae': float(mae),
            'spearman': float(spearman),
            'pearson': float(pearson),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'optimal_rerank_weight': round(optimal_rerank_weight, 2),
            'optimal_semantic_weight': round(optimal_semantic_weight, 2)
        }
        
        logger.info(f"Cross-Encoder Metrics:")
        logger.info(f"  MSE: {results['mse']:.4f}")
        logger.info(f"  MAE: {results['mae']:.4f}")
        logger.info(f"  Spearman: {results['spearman']:.4f}")
        logger.info(f"  Pearson: {results['pearson']:.4f}")
        logger.info(f"Recommended RERANK_WEIGHT: {results['optimal_rerank_weight']}")
        logger.info(f"Recommended SEMANTIC_WEIGHT: {results['optimal_semantic_weight']}")
        
        return results


class SpacyFineTuner:
    """Fine-tunes spaCy model for stylometry analysis."""
    
    def __init__(self, base_model: str = None, output_path: str = "models/spacy_finetuned"):
        self.base_model = base_model or SPACY_MODEL_NAME
        self.output_path = output_path
        self.nlp = None
        
    def prepare_training_data(self) -> List[Tuple[str, Dict]]:
        """Prepare training data for spaCy NER and dependency parsing."""
        train_data = []
        
        # Load datasets with entity and linguistic annotations
        try:
            # Use CoNLL for NER training - new format
            logger.info("Loading CoNLL dataset for NER training...")
            try:
                # Try new format first
                conll = load_dataset("wikiann", 'en', split="train[:5000]", trust_remote_code=True)
            except:
                # Fallback to legacy format
                conll = load_dataset("conll2003", split="train[:5000]", trust_remote_code=True)
            
            for item in conll:
                text = " ".join(item['tokens'])
                entities = []
                
                # Convert NER tags to spaCy format
                start = 0
                for token, ner_tag in zip(item['tokens'], item['ner_tags']):
                    end = start + len(token)
                    if ner_tag != 0:  # Not 'O' tag
                        # Map tag to entity type
                        tag_map = {1: 'PER', 2: 'PER', 3: 'ORG', 4: 'ORG', 
                                   5: 'LOC', 6: 'LOC', 7: 'MISC', 8: 'MISC'}
                        entity_type = tag_map.get(ner_tag, 'MISC')
                        entities.append((start, end, entity_type))
                    start = end + 1  # +1 for space
                
                train_data.append((text, {"entities": entities}))
                
        except Exception as e:
            logger.warning(f"Failed to load CoNLL data: {e}")
            logger.info("Generating synthetic NER training data as fallback...")
            
            # Fallback: Generate synthetic training data
            train_data = self._generate_synthetic_ner_data()
        
        logger.info(f"Prepared {len(train_data)} spaCy training examples")
        return train_data
    
    def _generate_synthetic_ner_data(self, n_samples: int = 1000) -> List[Tuple[str, Dict]]:
        """Generate synthetic NER training data for stylometry."""
        synthetic_data = []
        
        # Templates with entities
        templates = [
            ("John Smith works at Microsoft in Seattle.", [(0, 10, "PER"), (20, 29, "ORG"), (33, 40, "LOC")]),
            ("Apple Inc. was founded by Steve Jobs.", [(0, 10, "ORG"), (27, 37, "PER")]),
            ("The conference in Paris starts tomorrow.", [(18, 23, "LOC")]),
            ("Dr. Maria Garcia published research at Stanford University.", [(4, 17, "PER"), (39, 59, "ORG")]),
            ("Amazon opened offices in New York City.", [(0, 6, "ORG"), (25, 38, "LOC")]),
        ]
        
        # Simple name/org/location lists for variation
        names = ["Alice Brown", "Bob Wilson", "Carol Davis", "David Lee", "Emma White"]
        orgs = ["Google", "Facebook", "Tesla", "IBM", "Oracle"]
        locs = ["London", "Tokyo", "Berlin", "Madrid", "Sydney"]
        
        for _ in range(n_samples):
            # Pick a template
            template_text, _ = random.choice(templates)
            
            # Simple text variations
            if random.random() < 0.3:
                text = f"{random.choice(names)} presented at {random.choice(orgs)}."
                entities = [(0, len(names[0]), "PER"), 
                           (len(names[0]) + 14, len(names[0]) + 14 + len(orgs[0]), "ORG")]
            else:
                text, entities = random.choice(templates)
            
            synthetic_data.append((text, {"entities": entities}))
        
        return synthetic_data
    
    def fine_tune(self, n_iter: int = None, drop: float = None):
        """Fine-tune spaCy model."""
        # Use config values if not specified
        n_iter = n_iter or SPACY_TRAINING_ITERATIONS
        drop = drop or SPACY_DROPOUT
        
        logger.info("=" * 60)
        logger.info("FINE-TUNING SPACY")
        logger.info("=" * 60)
        logger.info(f"Hyperparameters: iterations={n_iter}, dropout={drop}")
        
        try:
            # Load base model
            logger.info(f"Loading {self.base_model}...")
            self.nlp = spacy.load(self.base_model)
            
            # Prepare data
            train_data = self.prepare_training_data()
            
            if not train_data:
                logger.warning("No training data for spaCy")
                return None
            
            # Get NER pipeline
            if "ner" not in self.nlp.pipe_names:
                ner = self.nlp.add_pipe("ner")
            else:
                ner = self.nlp.get_pipe("ner")
            
            # Add labels
            for _, annotations in train_data:
                for ent in annotations.get("entities"):
                    ner.add_label(ent[2])
            
            # Disable other pipelines
            other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
            
            # Train with loss tracking
            logger.info(f"Training: {len(train_data)} examples × {n_iter} iterations")
            
            losses_history = []
            
            with self.nlp.disable_pipes(*other_pipes):
                optimizer = self.nlp.resume_training()
                
                for iteration in range(n_iter):
                    random.shuffle(train_data)
                    losses = {}
                    
                    # Batch training
                    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                    
                    for batch in batches:
                        examples = []
                        for text, annotations in batch:
                            doc = self.nlp.make_doc(text)
                            example = Example.from_dict(doc, annotations)
                            examples.append(example)
                        
                        self.nlp.update(examples, drop=drop, losses=losses, sgd=optimizer)
                    
                    current_loss = losses.get('ner', 0)
                    losses_history.append(current_loss)
                    
                    if (iteration + 1) % 10 == 0:
                        logger.info(f"Iteration {iteration + 1}/{n_iter}, Loss: {current_loss:.4f}")
            
            # Save model
            self.nlp.to_disk(self.output_path)
            logger.info(f"✓ spaCy model saved to {self.output_path}")
            
            # Compute metrics and recommendations
            final_loss = losses_history[-1]
            initial_loss = losses_history[0]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            # Recommend hyperparameters based on convergence
            avg_last_5 = np.mean(losses_history[-5:])
            avg_prev_5 = np.mean(losses_history[-10:-5]) if len(losses_history) >= 10 else avg_last_5
            
            # If still improving significantly, recommend more iterations
            if (avg_prev_5 - avg_last_5) / avg_prev_5 > 0.05:
                recommended_iterations = min(n_iter + 10, 50)
            else:
                recommended_iterations = n_iter
            
            # If final loss is high, recommend higher dropout
            if final_loss > 50:
                recommended_dropout = min(drop + 0.05, 0.5)
            elif final_loss < 20:
                recommended_dropout = max(drop - 0.05, 0.1)
            else:
                recommended_dropout = drop
            
            results = {
                'final_loss': float(final_loss),
                'initial_loss': float(initial_loss),
                'improvement_pct': float(improvement),
                'mean_loss': float(np.mean(losses_history)),
                'min_loss': float(np.min(losses_history)),
                'losses_history': [float(l) for l in losses_history],
                'recommended_iterations': int(recommended_iterations),
                'recommended_dropout': round(float(recommended_dropout), 2),
                'hyperparameters': {
                    'iterations': n_iter,
                    'dropout': drop
                }
            }
            
            logger.info(f"spaCy Training Results:")
            logger.info(f"  Initial Loss: {results['initial_loss']:.4f}")
            logger.info(f"  Final Loss: {results['final_loss']:.4f}")
            logger.info(f"  Improvement: {results['improvement_pct']:.2f}%")
            logger.info(f"Recommended SPACY_TRAINING_ITERATIONS: {results['recommended_iterations']}")
            logger.info(f"Recommended SPACY_DROPOUT: {results['recommended_dropout']}")
            
            return results
            
        except Exception as e:
            logger.error(f"spaCy fine-tuning failed: {e}")
            return None


class ModelFineTuner:
    """Fine-tunes all models and updates config with optimized parameters."""

    def __init__(self, base_model: str = None, output_path: str = None, 
                 auto_update: bool = True, tune_all: bool = True):
        self.base_model = base_model or MODEL_BASE_NAME
        self.output_path = output_path or MODEL_FINE_TUNED_PATH
        self.auto_update = auto_update
        self.tune_all = tune_all  # Whether to tune cross-encoder and spacy
        self.model = None
        
        self.epochs = FINE_TUNING_EPOCHS
        self.batch_size = FINE_TUNING_BATCH_SIZE
        self.learning_rate = FINE_TUNING_LEARNING_RATE
        self.extended_corpus = EXTENDED_CORPUS_ENABLED
        
        # Sub-tuners
        self.cross_encoder_tuner = CrossEncoderFineTuner() if tune_all else None
        self.spacy_tuner = SpacyFineTuner() if tune_all else None
        
        self.metrics = {
            'datasets': {},
            'validation': {},
            'recommendations': {},
            'cross_encoder': {},
            'spacy': {}
        }
        
        logger.info(f"Initialized: epochs={self.epochs}, batch={self.batch_size}, "
                   f"lr={self.learning_rate}, tune_all={self.tune_all}")

    def _load_paws(self, max_examples: int = 5000) -> List[InputExample]:
        """Load PAWS dataset."""
        examples = []
        try:
            logger.info("Loading PAWS...")
            paws = load_dataset("paws", "labeled_final", split=f"train[:{max_examples}]")
            for item in paws:
                label = 1.0 if item.get("label") == 1 else 0.0
                examples.append(InputExample(
                    texts=[item["sentence1"], item["sentence2"]], label=label
                ))
            logger.info(f"✓ PAWS: {len(examples)} examples")
            self.metrics['datasets']['paws'] = len(examples)
        except Exception as e:
            logger.warning(f"PAWS failed: {e}")
        return examples

    def _load_qqp(self, max_examples: int = 5000) -> List[InputExample]:
        """Load QQP dataset."""
        examples = []
        try:
            logger.info("Loading QQP...")
            qqp = load_dataset("glue", "qqp", split=f"train[:{max_examples}]")
            for item in qqp:
                q1, q2 = item.get("question1"), item.get("question2")
                if q1 and q2:
                    label = 1.0 if item.get("label") == 1 else 0.0
                    examples.append(InputExample(texts=[q1, q2], label=label))
            logger.info(f"✓ QQP: {len(examples)} examples")
            self.metrics['datasets']['qqp'] = len(examples)
        except Exception as e:
            logger.warning(f"QQP failed: {e}")
        return examples

    def _load_stsb(self, max_examples: int = 20000) -> List[InputExample]:
        """Load STS-B dataset."""
        examples = []
        try:
            logger.info("Loading STS-B...")
            try:
                stsb = load_dataset("sentence-transformers/stsb", split="train")
            except:
                stsb = load_dataset("glue", "stsb", split="train")
            
            for i, item in enumerate(stsb):
                if i >= max_examples:
                    break
                s1, s2 = item.get("sentence1"), item.get("sentence2")
                score = item.get("score") or item.get("label") or 0.0
                
                score = float(score)
                if score > 1.0:
                    score /= 5.0
                score = max(0.0, min(1.0, score))
                
                if s1 and s2:
                    examples.append(InputExample(texts=[s1, s2], label=score))
            
            logger.info(f"✓ STS-B: {len(examples)} examples")
            self.metrics['datasets']['stsb'] = len(examples)
        except Exception as e:
            logger.warning(f"STS-B failed: {e}")
        return examples

    def prepare_data(self) -> Tuple[List[InputExample], List[InputExample]]:
        """Prepare training and validation splits."""
        logger.info("=" * 60)
        logger.info("PREPARING DATA")
        logger.info("=" * 60)
        
        all_examples = []
        all_examples.extend(self._load_paws())
        all_examples.extend(self._load_qqp())
        all_examples.extend(self._load_stsb())
        
        random.shuffle(all_examples)
        
        # 90-10 split
        split = int(0.9 * len(all_examples))
        train = all_examples[:split]
        val = all_examples[split:]
        
        logger.info(f"Total: {len(all_examples)}, Train: {len(train)}, Val: {len(val)}")
        logger.info("=" * 60)
        
        return train, val

    def evaluate(self, val_examples: List[InputExample]) -> Dict:
        """Evaluate model and compute optimal thresholds."""
        logger.info("Evaluating Sentence Transformer model...")
        
        sentences1 = [ex.texts[0] for ex in val_examples]
        sentences2 = [ex.texts[1] for ex in val_examples]
        labels = np.array([ex.label for ex in val_examples])
        
        # Encode
        emb1 = self.model.encode(sentences1, batch_size=32, show_progress_bar=False)
        emb2 = self.model.encode(sentences2, batch_size=32, show_progress_bar=False)
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sims = np.array([
            cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(emb1, emb2)
        ])
        
        # Find optimal threshold
        from sklearn.metrics import f1_score
        best_thresh = 0.5
        best_f1 = 0.0
        
        for thresh in np.arange(0.3, 0.9, 0.05):
            preds = (sims >= thresh).astype(float)
            binary_labels = (labels >= 0.5).astype(float)
            f1 = f1_score(binary_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Compute percentile-based thresholds
        high_sims = sims[labels >= 0.7]
        med_sims = sims[labels >= 0.4]
        
        high_thresh = float(np.percentile(high_sims, 25)) if len(high_sims) > 0 else 0.7
        med_thresh = float(np.percentile(med_sims, 50)) if len(med_sims) > 0 else 0.5
        
        # Clamp to reasonable ranges
        high_thresh = max(0.65, min(0.80, high_thresh))
        med_thresh = max(0.40, min(0.60, med_thresh))
        
        sem_high = max(0.55, min(0.70, float(np.percentile(sims, 75))))
        sem_med = max(0.35, min(0.50, float(np.percentile(sims, 50))))
        
        results = {
            'mean_sim': float(np.mean(sims)),
            'std_sim': float(np.std(sims)),
            'best_f1': float(best_f1),
            'best_thresh': float(best_thresh),
            'high_risk_thresh': high_thresh,
            'med_risk_thresh': med_thresh,
            'sem_high_floor': sem_high,
            'sem_med_floor': sem_med
        }
        
        logger.info(f"Sentence Transformer Metrics:")
        logger.info(f"  Mean similarity: {results['mean_sim']:.3f}")
        logger.info(f"  Best F1: {results['best_f1']:.3f} @ {results['best_thresh']:.3f}")
        logger.info(f"Recommended HIGH_RISK_THRESHOLD: {results['high_risk_thresh']:.3f}")
        logger.info(f"Recommended MEDIUM_RISK_THRESHOLD: {results['med_risk_thresh']:.3f}")
        
        return results

    def compute_weights(self) -> Dict:
        """Compute optimal fusion weights based on dataset composition."""
        logger.info("Computing fusion weights...")
        
        datasets = self.metrics['datasets']
        total = sum(datasets.values())
        
        # Higher semantic weight if more semantic datasets
        semantic_ratio = sum(datasets.get(k, 0) for k in ['paws', 'qqp', 'stsb']) / total
        
        if semantic_ratio > 0.7:
            w_sem = 0.70
        else:
            w_sem = 0.60
        
        w_stylo = 0.25
        w_ai = 1.0 - w_sem - w_stylo
        
        weights = {
            'semantic': round(w_sem, 2),
            'stylo': round(w_stylo, 2),
            'ai': round(w_ai, 2)
        }
        
        logger.info(f"Recommended fusion weights: semantic={weights['semantic']}, "
                   f"stylo={weights['stylo']}, ai={weights['ai']}")
        
        return weights

    def fine_tune(self, warmup_ratio: float = 0.1):
        """Fine-tune all models and update config."""
        logger.info("=" * 60)
        logger.info("FINE-TUNING ALL MODELS")
        logger.info("=" * 60)
        
        # Get data (shared across models)
        train_examples, val_examples = self.prepare_data()
        
        if not train_examples:
            logger.error("No training data!")
            return
        
        # 1. Fine-tune Sentence Transformer
        logger.info("=" * 60)
        logger.info("FINE-TUNING SENTENCE TRANSFORMER")
        logger.info("=" * 60)
        
        logger.info(f"Loading {self.base_model}...")
        self.model = SentenceTransformer(self.base_model)
        
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)
        warmup_steps = int(warmup_ratio * len(train_loader) * self.epochs)
        
        logger.info(f"Training: {len(train_loader)} steps/epoch × {self.epochs} epochs")
        
        self.model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            output_path=str(self.output_path),
            optimizer_params={'lr': self.learning_rate}
        )
        
        logger.info(f"✓ Sentence Transformer saved to {self.output_path}")
        
        # Evaluate Sentence Transformer
        eval_results = self.evaluate(val_examples) if val_examples else {}
        self.metrics['validation'] = eval_results
        
        # 2. Fine-tune Cross-Encoder
        if self.tune_all and self.cross_encoder_tuner:
            cross_results = self.cross_encoder_tuner.fine_tune(train_examples)
            if cross_results:
                self.metrics['cross_encoder'] = cross_results
        
        # 3. Fine-tune spaCy
        if self.tune_all and self.spacy_tuner:
            spacy_results = self.spacy_tuner.fine_tune()
            if spacy_results:
                self.metrics['spacy'] = spacy_results
        
        # 4. Compute weights and update config
        if self.auto_update:
            logger.info("=" * 60)
            logger.info("UPDATING CONFIG")
            logger.info("=" * 60)
            
            weights = self.compute_weights()
            self.metrics['recommendations'] = weights
            
            # Collect all updates
            updates = {}
            
            # SBERT thresholds and weights
            updates.update({
                'HIGH_RISK_THRESHOLD': eval_results.get('high_risk_thresh', 0.7),
                'MEDIUM_RISK_THRESHOLD': eval_results.get('med_risk_thresh', 0.5),
                'SEMANTIC_HIGH_FLOOR': eval_results.get('sem_high_floor', 0.6),
                'SEMANTIC_MEDIUM_FLOOR': eval_results.get('sem_med_floor', 0.4),
                'WEIGHT_SEMANTIC': weights['semantic'],
                'WEIGHT_STYLO': weights['stylo'],
                'WEIGHT_AI': weights['ai']
            })
            
            # Cross-Encoder updates
            if self.metrics.get('cross_encoder'):
                ce_metrics = self.metrics['cross_encoder']
                updates.update({
                    'RERANK_WEIGHT': ce_metrics.get('optimal_rerank_weight', RERANK_WEIGHT),
                    'SEMANTIC_WEIGHT': ce_metrics.get('optimal_semantic_weight', SEMANTIC_WEIGHT),
                    'CROSS_ENCODER_MODEL_NAME': self.cross_encoder_tuner.output_path
                })
                
                # Update hyperparameters if performance suggests changes
                # If Spearman correlation is strong, current hyperparameters are good
                # If weak, suggest adjustments
                if ce_metrics.get('spearman', 0) < 0.5:
                    # Poor performance - increase epochs
                    updates['CROSS_ENCODER_EPOCHS'] = min(CROSS_ENCODER_EPOCHS + 2, 8)
                    logger.info(f"⚠ Low Cross-Encoder correlation - recommending more epochs")
                elif ce_metrics.get('spearman', 0) > 0.75:
                    # Excellent performance - can reduce epochs for faster training
                    updates['CROSS_ENCODER_EPOCHS'] = max(CROSS_ENCODER_EPOCHS - 1, 3)
                    logger.info(f"✓ High Cross-Encoder correlation - can use fewer epochs")
            
            # spaCy updates
            if self.metrics.get('spacy'):
                spacy_metrics = self.metrics['spacy']
                updates.update({
                    'SPACY_MODEL_NAME': self.spacy_tuner.output_path,
                    'SPACY_TRAINING_ITERATIONS': spacy_metrics.get('recommended_iterations', SPACY_TRAINING_ITERATIONS),
                    'SPACY_DROPOUT': spacy_metrics.get('recommended_dropout', SPACY_DROPOUT)
                })
            
            # Update config.py
            ConfigUpdater.update_config_file(updates)
            ConfigUpdater.save_report(self.metrics)
            
            logger.info("=" * 60)
            logger.info("✓ ALL MODELS FINE-TUNED & CONFIG UPDATED")
            logger.info("=" * 60)
            
            # Print summary
            self._print_summary()
    
    def _print_summary(self):
        """Print a summary of all fine-tuning results."""
        logger.info("\n" + "=" * 60)
        logger.info("FINE-TUNING SUMMARY")
        logger.info("=" * 60)
        
        # Dataset info
        logger.info("\n📊 DATASETS:")
        for name, count in self.metrics['datasets'].items():
            logger.info(f"  {name.upper()}: {count} examples")
        
        # SBERT results
        if self.metrics.get('validation'):
            val = self.metrics['validation']
            logger.info("\n🤖 SENTENCE TRANSFORMER:")
            logger.info(f"  Mean Similarity: {val.get('mean_sim', 0):.3f}")
            logger.info(f"  Best F1 Score: {val.get('best_f1', 0):.3f}")
            logger.info(f"  HIGH_RISK_THRESHOLD: {val.get('high_risk_thresh', 0):.3f}")
            logger.info(f"  MEDIUM_RISK_THRESHOLD: {val.get('med_risk_thresh', 0):.3f}")
        
        # Cross-Encoder results
        if self.metrics.get('cross_encoder'):
            ce = self.metrics['cross_encoder']
            logger.info("\n🔄 CROSS-ENCODER:")
            logger.info(f"  Spearman Correlation: {ce.get('spearman', 0):.3f}")
            logger.info(f"  MAE: {ce.get('mae', 0):.4f}")
            logger.info(f"  RERANK_WEIGHT: {ce.get('optimal_rerank_weight', 0):.2f}")
            logger.info(f"  Epochs Used: {ce.get('hyperparameters', {}).get('epochs', 'N/A')}")
        
        # spaCy results
        if self.metrics.get('spacy'):
            sp = self.metrics['spacy']
            logger.info("\n📝 SPACY:")
            logger.info(f"  Final Loss: {sp.get('final_loss', 0):.4f}")
            logger.info(f"  Improvement: {sp.get('improvement_pct', 0):.2f}%")
            logger.info(f"  Recommended Iterations: {sp.get('recommended_iterations', 'N/A')}")
            logger.info(f"  Recommended Dropout: {sp.get('recommended_dropout', 'N/A')}")
        
        # Fusion weights
        if self.metrics.get('recommendations'):
            weights = self.metrics['recommendations']
            logger.info("\n⚖️ FUSION WEIGHTS:")
            logger.info(f"  Semantic: {weights.get('semantic', 0):.2f}")
            logger.info(f"  Stylometry: {weights.get('stylo', 0):.2f}")
            logger.info(f"  AI Detection: {weights.get('ai', 0):.2f}")
        
        logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set tune_all=True to fine-tune all models
    # Set auto_update=True to automatically update config.py
    tuner = ModelFineTuner(auto_update=True, tune_all=True)
    tuner.fine_tune()