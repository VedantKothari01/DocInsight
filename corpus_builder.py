import os
import re
import json
import random
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from config import CACHE_DIR, MODEL_FINE_TUNED_PATH, USE_FINE_TUNED_MODEL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CorpusIndex:
    """
    Builds, caches, and fine-tunes the corpus index for semantic and stylometric analysis.
    Fine-tunes SBERT using multiple datasets: PAWS, QQP, STSB, 100 English Novels, and
    the Source Code Plagiarism dataset.
    """

    def __init__(self, target_size: int = 20000, use_domain_adaptation: bool = True):
        self.target_size = target_size
        self.use_domain_adaptation = use_domain_adaptation
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load either base or fine-tuned model
        if USE_FINE_TUNED_MODEL and Path(MODEL_FINE_TUNED_PATH).exists():
            logger.info(f"Loading fine-tuned SBERT model from {MODEL_FINE_TUNED_PATH}")
            self.model = SentenceTransformer(MODEL_FINE_TUNED_PATH)
        else:
            logger.info("Loading base SBERT model: sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # --------------------------------------------------------------------------
    #                            DATA PREPARATION
    # --------------------------------------------------------------------------
    def _prepare_training_data(self) -> List[InputExample]:
        """Prepares training examples from multiple datasets for SBERT fine-tuning."""
        examples: List[InputExample] = []

        # --- PAWS ---
        try:
            logger.info("Loading PAWS dataset...")
            paws = load_dataset("paws", "labeled_final", split="train[:5000]")
            for item in paws:
                label = 1.0 if item.get("label") == 1 else 0.0
                examples.append(InputExample(
                    texts=[item["sentence1"], item["sentence2"]],
                    label=label
                ))
            logger.info(f"Added {len(paws)} PAWS examples.")
        except Exception as e:
            logger.warning(f"Failed to load PAWS: {e}")

        # --- QQP ---
        try:
            logger.info("Loading QQP dataset...")
            qqp = load_dataset("glue", "qqp", split="train[:5000]")
            for item in qqp:
                q1, q2 = item.get("question1"), item.get("question2")
                if q1 and q2:
                    label = 1.0 if item.get("label") == 1 else 0.0
                    examples.append(InputExample(texts=[q1, q2], label=label))
            logger.info(f"Added {len(qqp)} QQP examples.")
        except Exception as e:
            logger.warning(f"Failed to load QQP: {e}")

        # --- STSB ---
        try:
            logger.info("Loading STS-B dataset...")
            try:
                stsb = load_dataset("sentence-transformers/stsb", split="train")
            except Exception:
                stsb = load_dataset("glue", "stsb", split="train")

            stsb_count = 0
            for item in stsb:
                s1, s2 = item.get("sentence1"), item.get("sentence2")
                score = item.get("score") or item.get("label") or 0.0
                try:
                    score = float(score)
                    if score > 1.0:
                        score /= 5.0
                except Exception:
                    score = 0.0
                if s1 and s2:
                    examples.append(InputExample(texts=[s1, s2], label=max(0.0, min(1.0, score))))
                    stsb_count += 1
                    if stsb_count >= 20000:
                        break
            logger.info(f"Added {stsb_count} STS-B examples.")
        except Exception as e:
            logger.warning(f"Failed to load STS-B: {e}")

        # --- 100 English Novels (weak negatives) ---
        def _extract_sentences_from_text(path: Path) -> List[str]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 40]
                return sents
            except Exception:
                return []

        NOVELS_DIR = Path("data/novels")
        if NOVELS_DIR.exists():
            logger.info("Parsing 100 English Novels for weak supervision...")
            added = 0
            for txt in NOVELS_DIR.rglob("*.txt"):
                sents = _extract_sentences_from_text(txt)
                for i in range(len(sents) - 1):
                    a, b = sents[i], sents[i + 1]
                    if len(a) < 40 or len(b) < 40:
                        continue
                    examples.append(InputExample(texts=[a, b], label=0.0))  # negatives
                    added += 1
                    if added >= 20000:
                        break
                if added >= 20000:
                    break
            logger.info(f"Added {added} novel-derived pairs.")
        else:
            logger.info("No novels directory found; skipping 100 English Novels.")

        # --- Source Code Plagiarism Dataset ---
        SOURCE_DIR = Path("data/sourcecode")
        if SOURCE_DIR.exists():
            logger.info("Parsing Source Code Plagiarism dataset...")
            added_pos, added_neg = 0, 0

            # Each case folder = one assignment set
            for case_dir in SOURCE_DIR.glob("case-*"):
                original_dir = case_dir / "Original"
                plag_dir = case_dir / "plagiarized"
                non_plag_dir = case_dir / "non-plagiarized"

                orig_files = list(original_dir.rglob("*.java"))
                if not orig_files:
                    continue

                # Assume single original file per case
                orig_code = orig_files[0].read_text(encoding="utf-8", errors="ignore")

                # Positive pairs: original vs plagiarized
                if plag_dir.exists():
                    for lvl_dir in plag_dir.glob("level-*"):
                        for subdir in lvl_dir.iterdir():
                            for file in subdir.rglob("*.java"):
                                try:
                                    plag_code = file.read_text(encoding="utf-8", errors="ignore")
                                    if len(plag_code) < 40:
                                        continue
                                    examples.append(InputExample(texts=[orig_code, plag_code], label=1.0))
                                    added_pos += 1
                                except Exception:
                                    continue

                # Negative pairs: original vs non-plagiarized
                if non_plag_dir.exists():
                    for subdir in non_plag_dir.iterdir():
                        for file in subdir.rglob("*.java"):
                            try:
                                non_plag_code = file.read_text(encoding="utf-8", errors="ignore")
                                if len(non_plag_code) < 40:
                                    continue
                                examples.append(InputExample(texts=[orig_code, non_plag_code], label=0.0))
                                added_neg += 1
                            except Exception:
                                continue

            logger.info(f"Added {added_pos} plagiarized (+) and {added_neg} non-plagiarized (−) code pairs.")
        else:
            logger.info("No source code directory found; skipping Source Code dataset.")

        random.shuffle(examples)
        logger.info(f"Total training examples prepared: {len(examples)}")
        return examples

    # --------------------------------------------------------------------------
    #                             DOMAIN ADAPTATION
    # --------------------------------------------------------------------------
    def _perform_domain_adaptation(self):
        """Fine-tunes SBERT on multiple datasets."""
        if not self.use_domain_adaptation:
            logger.info("Domain adaptation disabled.")
            return

        logger.info("Starting SBERT domain adaptation (multi-dataset fine-tuning).")

        examples = self._prepare_training_data()
        if not examples:
            logger.warning("No examples found — skipping fine-tuning.")
            return

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.model)
        warmup_steps = int(0.1 * len(train_dataloader))

        logger.info(f"Warmup steps: {warmup_steps}")

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            output_path=str(MODEL_FINE_TUNED_PATH)
        )

        logger.info(f"Fine-tuned model saved to: {MODEL_FINE_TUNED_PATH}")

    # --------------------------------------------------------------------------
    #                            CORPUS BUILDING
    # --------------------------------------------------------------------------
    def build_index(self, corpus_texts: List[str]) -> Dict[str, List[float]]:
        """Encodes corpus texts and caches embeddings."""
        logger.info("Encoding corpus and building embeddings...")
        embeddings = self.model.encode(
            corpus_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=16
        )
        logger.info(f"Encoded {len(corpus_texts)} documents.")
        return {"texts": corpus_texts, "embeddings": embeddings.tolist()}

    def save_index(self, index_data: Dict[str, List[float]], name: str = "default_index"):
        path = self.cache_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)
        logger.info(f"Index saved at {path}")

    def load_index(self, name: str = "default_index") -> Dict[str, List[float]]:
        path = self.cache_dir / f"{name}.json"
        if not path.exists():
            logger.error(f"Index {name} not found in cache.")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Index {name} loaded.")
        return data


if __name__ == "__main__":
    corpus = CorpusIndex(target_size=20000, use_domain_adaptation=True)
    corpus._perform_domain_adaptation()
