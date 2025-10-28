#!/usr/bin/env python3
"""Evaluate base vs (optional) fine-tuned semantic model on validation/test splits.

Outputs:
  - JSON metrics summary (scripts/output/model_eval.json)
  - Markdown comparison table (scripts/output/model_eval.md)

Metrics:
  - Spearman correlation (cosine similarity vs labels)
  - Threshold sweep (0.3..0.9) best F1, precision, recall

Usage:
  python scripts/evaluate_models.py --data fine_tuning/data
"""
import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("model_eval")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
    raise SystemExit(1)

# Ensure project root on path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SBERT_MODEL_NAME, MODEL_FINE_TUNED_PATH, USE_FINE_TUNED_MODEL


def load_split_frames(data_dir: str) -> Dict[str, pd.DataFrame]:
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{name}.csv")
        if os.path.exists(path):
            splits[name] = pd.read_csv(path)
    if 'val' not in splits and 'test' not in splits:
        raise FileNotFoundError("Need at least val or test split for evaluation")
    return splits


def load_models() -> Dict[str, SentenceTransformer]:
    models = {}
    # Base model
    models['base'] = SentenceTransformer(SBERT_MODEL_NAME)
    # Fine-tuned (if exists)
    ft_config = os.path.join(MODEL_FINE_TUNED_PATH, 'config.json')
    if os.path.exists(ft_config):
        try:
            models['fine_tuned'] = SentenceTransformer(MODEL_FINE_TUNED_PATH)
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {e}")
    return models


def compute_embeddings(model: SentenceTransformer, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    emb_a = model.encode(df['text_a'].tolist(), convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    emb_b = model.encode(df['text_b'].tolist(), convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    return emb_a, emb_b


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_norm * b_norm).sum(axis=1)


def threshold_metrics(sims: np.ndarray, labels: np.ndarray, thresholds: List[float]) -> Dict:
    best = {'f1': -1, 'threshold': None, 'precision': 0, 'recall': 0}
    rows = []
    for t in thresholds:
        preds = (sims >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
        rows.append({'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1})
        if f1 > best['f1']:
            best = {'f1': f1, 'threshold': t, 'precision': precision, 'recall': recall}
    return {'sweep': rows, 'best': best}


def evaluate_model(model: SentenceTransformer, df: pd.DataFrame) -> Dict:
    emb_a, emb_b = compute_embeddings(model, df)
    sims = cosine_sim(emb_a, emb_b)
    labels = df['label'].astype(float).to_numpy()
    corr, _ = spearmanr(sims, labels)
    thresholds = [round(x,2) for x in np.arange(0.30, 0.91, 0.05)]
    t_metrics = threshold_metrics(sims, labels, thresholds)
    return {
        'spearman': float(corr if corr is not None else 0.0),
        'threshold_sweep': t_metrics['sweep'],
        'best_threshold': t_metrics['best']
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='fine_tuning/data', help='Directory with train/val/test splits')
    parser.add_argument('--output', default='scripts/output', help='Directory for evaluation outputs')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    splits = load_split_frames(args.data)
    eval_split_name = 'val' if 'val' in splits else 'test'
    eval_df = splits[eval_split_name]
    logger.info(f"Using split '{eval_split_name}' with {len(eval_df)} examples for evaluation")

    models = load_models()
    results = {}
    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        results[name] = evaluate_model(model, eval_df)

    # Delta if both present
    if 'base' in results and 'fine_tuned' in results:
        delta = {
            'spearman_gain': results['fine_tuned']['spearman'] - results['base']['spearman'],
            'f1_gain': results['fine_tuned']['best_threshold']['f1'] - results['base']['best_threshold']['f1']
        }
    else:
        delta = {}

    summary = {
        'split_used': eval_split_name,
        'examples': len(eval_df),
        'use_fine_tuned_flag': USE_FINE_TUNED_MODEL,
        'models_evaluated': list(results.keys()),
        'results': results,
        'delta': delta
    }

    json_path = os.path.join(args.output, 'model_eval.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    md_lines = ["# Model Evaluation Summary", f"Split: {eval_split_name} ({len(eval_df)} examples)", ""]
    for name, metrics in results.items():
        best = metrics['best_threshold']
        md_lines.append(f"## {name} model")
        md_lines.append(f"- Spearman: {metrics['spearman']:.4f}")
        md_lines.append(f"- Best Threshold: {best['threshold']:.2f} (F1={best['f1']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f})")
        md_lines.append("")
    if delta:
        md_lines.append("### Improvement (fine_tuned - base)")
        md_lines.append(f"- Spearman gain: {delta['spearman_gain']:.4f}")
        md_lines.append(f"- F1 gain: {delta['f1_gain']:.4f}")
    md_path = os.path.join(args.output, 'model_eval.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    logger.info(f"Evaluation complete. JSON: {json_path}, MD: {md_path}")

if __name__ == '__main__':
    main()
