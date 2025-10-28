#!/usr/bin/env python3
"""One-shot script to generate synthetic pairs (if missing), prepare splits,
optionally fine-tune (skips if model exists and FORCE_RETRAIN false), then evaluate.

Usage:
  python scripts/run_full_finetune_and_eval.py
Environment flags:
  DOCINSIGHT_FORCE_RETRAIN=true  # force retrain
"""
import os
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("full_ft")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'fine_tuning' / 'data'
PAIRS_CSV = DATA_DIR / 'pairs.csv'
OUTPUT_DIR = PROJECT_ROOT / 'scripts' / 'output'

CMDS = []

# 1. Synthetic pairs generation if missing
if not PAIRS_CSV.exists():
    CMDS.append([sys.executable, 'scripts/generate_synthetic_pairs.py'])
else:
    log.info('Synthetic pairs already exist - skipping generation')

# 2. Dataset preparation (always runs to ensure splits)
CMDS.append([sys.executable, 'fine_tuning/dataset_prep.py'])

# 3. Fine-tune semantic model (will skip internally if already present)
CMDS.append([sys.executable, 'fine_tuning/fine_tune_semantic.py', '--data', 'fine_tuning/data'])

# 4. Evaluation (only meaningful if val/test exist)
CMDS.append([sys.executable, 'scripts/evaluate_models.py', '--data', 'fine_tuning/data', '--output', 'scripts/output'])


def run_cmd(cmd):
    log.info('Running: ' + ' '.join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cmd in CMDS:
        run_cmd(cmd)
    log.info('Full fine-tune and evaluation pipeline completed.')

if __name__ == '__main__':
    main()
