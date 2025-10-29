import os
import re
import random
import logging
from pathlib import Path
from typing import List

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from config import MODEL_FINE_TUNED_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelFineTuner:
    """
    Fine-tunes sentence-transformers model on multiple datasets:
    - PAWS (paraphrase detection)
    - QQP (question similarity)
    - STSB (semantic textual similarity)
    - 100 English Novels (stylometric patterns)
    - Source Code Plagiarism dataset (code similarity)
    """

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        from config import MODEL_FINE_TUNED_PATH as CONFIG_PATH
        self.base_model = base_model
        self.output_path = CONFIG_PATH
        self.model = None

    def _load_paws_dataset(self, max_examples: int = 5000) -> List[InputExample]:
        """Load PAWS dataset for paraphrase detection."""
        examples = []
        try:
            logger.info("Loading PAWS dataset...")
            paws = load_dataset("paws", "labeled_final", split=f"train[:{max_examples}]")
            for item in paws:
                label = 1.0 if item.get("label") == 1 else 0.0
                examples.append(InputExample(
                    texts=[item["sentence1"], item["sentence2"]],
                    label=label
                ))
            logger.info(f"✓ Added {len(examples)} PAWS examples")
        except Exception as e:
            logger.warning(f"✗ Failed to load PAWS: {e}")
        return examples

    def _load_qqp_dataset(self, max_examples: int = 5000) -> List[InputExample]:
        """Load QQP dataset for question similarity."""
        examples = []
        try:
            logger.info("Loading QQP dataset...")
            qqp = load_dataset("glue", "qqp", split=f"train[:{max_examples}]")
            for item in qqp:
                q1, q2 = item.get("question1"), item.get("question2")
                if q1 and q2:
                    label = 1.0 if item.get("label") == 1 else 0.0
                    examples.append(InputExample(texts=[q1, q2], label=label))
            logger.info(f"✓ Added {len(examples)} QQP examples")
        except Exception as e:
            logger.warning(f"✗ Failed to load QQP: {e}")
        return examples

    def _load_stsb_dataset(self, max_examples: int = 20000) -> List[InputExample]:
        """Load STS-B dataset for semantic similarity."""
        examples = []
        try:
            logger.info("Loading STS-B dataset...")
            try:
                stsb = load_dataset("sentence-transformers/stsb", split="train")
            except Exception:
                stsb = load_dataset("glue", "stsb", split="train")

            count = 0
            for item in stsb:
                if count >= max_examples:
                    break
                s1, s2 = item.get("sentence1"), item.get("sentence2")
                score = item.get("score") or item.get("label") or 0.0
                
                # Normalize score to [0, 1]
                try:
                    score = float(score)
                    if score > 1.0:
                        score /= 5.0
                    score = max(0.0, min(1.0, score))
                except Exception:
                    score = 0.0
                
                if s1 and s2:
                    examples.append(InputExample(texts=[s1, s2], label=score))
                    count += 1
            
            logger.info(f"✓ Added {len(examples)} STS-B examples")
        except Exception as e:
            logger.warning(f"✗ Failed to load STS-B: {e}")
        return examples

    def _load_novels_dataset(
        self, 
        novels_dir: str = "data/novels",
        max_examples: int = 20000,
        min_sentence_length: int = 40
    ) -> List[InputExample]:
        """
        Load 100 English Novels dataset for stylometric analysis.
        Creates negative pairs from consecutive sentences (different styles).
        """
        examples = []
        novels_path = Path(novels_dir)
        
        if not novels_path.exists():
            logger.warning(f"✗ Novels directory not found at {novels_dir}")
            return examples

        logger.info("Loading 100 English Novels dataset...")
        
        def extract_sentences(file_path: Path) -> List[str]:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                # Split on sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', text)
                return [s.strip() for s in sentences if len(s.strip()) >= min_sentence_length]
            except Exception as e:
                logger.debug(f"Error reading {file_path}: {e}")
                return []

        added = 0
        for txt_file in novels_path.rglob("*.txt"):
            if added >= max_examples:
                break
            
            sentences = extract_sentences(txt_file)
            
            # Create negative pairs from consecutive sentences
            # (same author/style but different content)
            for i in range(len(sentences) - 1):
                if added >= max_examples:
                    break
                
                sent_a, sent_b = sentences[i], sentences[i + 1]
                if len(sent_a) >= min_sentence_length and len(sent_b) >= min_sentence_length:
                    examples.append(InputExample(texts=[sent_a, sent_b], label=0.3))
                    added += 1

        logger.info(f"✓ Added {len(examples)} novel-derived pairs")
        return examples

    def _load_sourcecode_dataset(
        self,
        sourcecode_dir: str = "data/sourcecode",
        min_code_length: int = 40
    ) -> List[InputExample]:
        """
        Load Source Code Plagiarism dataset.
        Creates positive pairs (original vs plagiarized) and negative pairs (original vs non-plagiarized).
        """
        examples = []
        source_path = Path(sourcecode_dir)
        
        if not source_path.exists():
            logger.warning(f"✗ Source code directory not found at {sourcecode_dir}")
            return examples

        logger.info("Loading Source Code Plagiarism dataset...")
        
        added_pos = 0
        added_neg = 0

        for case_dir in source_path.glob("case-*"):
            original_dir = case_dir / "Original"
            plag_dir = case_dir / "plagiarized"
            non_plag_dir = case_dir / "non-plagiarized"

            # Get original file
            orig_files = list(original_dir.rglob("*.java"))
            if not orig_files:
                continue

            try:
                orig_code = orig_files[0].read_text(encoding="utf-8", errors="ignore")
                if len(orig_code) < min_code_length:
                    continue
            except Exception:
                continue

            # Positive pairs: original vs plagiarized
            if plag_dir.exists():
                for lvl_dir in plag_dir.glob("level-*"):
                    for subdir in lvl_dir.iterdir():
                        if not subdir.is_dir():
                            continue
                        for file in subdir.rglob("*.java"):
                            try:
                                plag_code = file.read_text(encoding="utf-8", errors="ignore")
                                if len(plag_code) >= min_code_length:
                                    examples.append(
                                        InputExample(texts=[orig_code, plag_code], label=1.0)
                                    )
                                    added_pos += 1
                            except Exception:
                                continue

            # Negative pairs: original vs non-plagiarized
            if non_plag_dir.exists():
                for subdir in non_plag_dir.iterdir():
                    if not subdir.is_dir():
                        continue
                    for file in subdir.rglob("*.java"):
                        try:
                            non_plag_code = file.read_text(encoding="utf-8", errors="ignore")
                            if len(non_plag_code) >= min_code_length:
                                examples.append(
                                    InputExample(texts=[orig_code, non_plag_code], label=0.0)
                                )
                                added_neg += 1
                        except Exception:
                            continue

        logger.info(f"✓ Added {added_pos} plagiarized (+) and {added_neg} non-plagiarized (−) code pairs")
        return examples

    def prepare_training_data(self) -> List[InputExample]:
        """Prepare all training examples from multiple datasets."""
        logger.info("=" * 60)
        logger.info("PREPARING TRAINING DATA")
        logger.info("=" * 60)
        
        all_examples = []
        
        # Load each dataset
        all_examples.extend(self._load_paws_dataset(max_examples=5000))
        all_examples.extend(self._load_qqp_dataset(max_examples=5000))
        all_examples.extend(self._load_stsb_dataset(max_examples=20000))
        all_examples.extend(self._load_novels_dataset(max_examples=20000))
        all_examples.extend(self._load_sourcecode_dataset())
        
        # Shuffle for better training
        random.shuffle(all_examples)
        
        logger.info("=" * 60)
        logger.info(f"TOTAL TRAINING EXAMPLES: {len(all_examples)}")
        logger.info("=" * 60)
        
        return all_examples

    def fine_tune(
        self,
        epochs: int = 1,
        batch_size: int = 16,
        warmup_ratio: float = 0.1,
        show_progress: bool = True
    ):
        """
        Fine-tune the model on prepared datasets.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_ratio: Ratio of total steps for warmup
            show_progress: Whether to show progress bar
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL FINE-TUNING")
        logger.info("=" * 60)
        
        # Load base model
        logger.info(f"Loading base model: {self.base_model}")
        self.model = SentenceTransformer(self.base_model)
        
        # Prepare training data
        examples = self.prepare_training_data()
        
        if not examples:
            logger.error("✗ No training examples found. Aborting fine-tuning.")
            return
        
        # Create DataLoader
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function (Cosine Similarity Loss)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Calculate warmup steps
        warmup_steps = int(warmup_ratio * len(train_dataloader) * epochs)
        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Total steps: {len(train_dataloader) * epochs}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
        
        # Fine-tune the model
        logger.info("Training in progress...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=show_progress,
            output_path=str(self.output_path)
        )
        
        logger.info("=" * 60)
        logger.info(f"✓ Fine-tuning complete!")
        logger.info(f"✓ Model saved to: {self.output_path}")
        logger.info("=" * 60)

    def evaluate(self, test_examples: List[InputExample] = None):
        """
        Optional: Evaluate the fine-tuned model.
        You can implement custom evaluation logic here.
        """
        if not self.model:
            logger.error("Model not loaded. Train or load a model first.")
            return
        
        # TODO: Implement evaluation logic
        logger.info("Evaluation not yet implemented.")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize fine-tuner
    fine_tuner = ModelFineTuner(
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        output_path="models/fine_tuned_sbert"
    )
    
    # Run fine-tuning
    fine_tuner.fine_tune(
        epochs=1,
        batch_size=16,
        warmup_ratio=0.1,
        show_progress=True
    )