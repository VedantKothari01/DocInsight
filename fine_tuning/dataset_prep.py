"""
Dataset preparation module for fine-tuning

Builds supervised datasets for semantic similarity/paraphrase classification
from existing corpus and synthetic transformations.
"""

import os
import csv
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetPreparator:
    """Prepares training datasets for semantic similarity fine-tuning"""
    
    def __init__(self, data_dir: str = "fine_tuning/data"):
        """Initialize dataset preparator
        
        Args:
            data_dir: Directory to store prepared datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_pairs_csv(self, filepath: str) -> pd.DataFrame:
        """Load pairs CSV file into DataFrame
        
        Args:
            filepath: Path to CSV file with text_a, text_b, label columns
            
        Returns:
            DataFrame with loaded pairs
        """
        try:
            df = pd.read_csv(filepath)
            required_columns = {'text_a', 'text_b', 'label'}
            
            if not required_columns.issubset(df.columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            logger.info(f"Loaded {len(df)} pairs from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading pairs CSV: {e}")
            raise
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate dataset quality and return statistics
        
        Args:
            df: DataFrame with pairs
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'total_pairs': len(df),
            'positive_pairs': len(df[df['label'] == 1]),
            'negative_pairs': len(df[df['label'] == 0]),
            'empty_texts': 0,
            'duplicate_pairs': 0,
            'avg_text_length': 0,
            'validation_errors': []
        }
        
        # Check for empty texts
        empty_a = df['text_a'].isna() | (df['text_a'].str.strip() == '')
        empty_b = df['text_b'].isna() | (df['text_b'].str.strip() == '')
        stats['empty_texts'] = (empty_a | empty_b).sum()
        
        if stats['empty_texts'] > 0:
            stats['validation_errors'].append(f"Found {stats['empty_texts']} pairs with empty text")
        
        # Check for duplicates
        duplicate_mask = df.duplicated(subset=['text_a', 'text_b'])
        stats['duplicate_pairs'] = duplicate_mask.sum()
        
        if stats['duplicate_pairs'] > 0:
            stats['validation_errors'].append(f"Found {stats['duplicate_pairs']} duplicate pairs")
        
        # Calculate average text length
        text_lengths = df['text_a'].str.len() + df['text_b'].str.len()
        stats['avg_text_length'] = text_lengths.mean()
        
        # Check class balance
        positive_ratio = stats['positive_pairs'] / stats['total_pairs']
        if positive_ratio < 0.3 or positive_ratio > 0.7:
            stats['validation_errors'].append(f"Class imbalance: {positive_ratio:.2%} positive pairs")
        
        # Check minimum dataset size
        if stats['total_pairs'] < 50:
            stats['validation_errors'].append(f"Dataset too small: {stats['total_pairs']} pairs (minimum 50 recommended)")
        
        logger.info(f"Dataset validation: {len(stats['validation_errors'])} issues found")
        return stats
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by removing invalid entries
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        original_size = len(df)
        
        # Remove rows with empty texts
        df = df.dropna(subset=['text_a', 'text_b'])
        df = df[(df['text_a'].str.strip() != '') & (df['text_b'].str.strip() != '')]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text_a', 'text_b'])
        
        # Ensure labels are binary (0 or 1)
        df = df[df['label'].isin([0, 1])]
        
        # Remove extremely short texts (less than 3 words)
        min_words = 3
        word_count_a = df['text_a'].str.split().str.len()
        word_count_b = df['text_b'].str.split().str.len()
        df = df[(word_count_a >= min_words) & (word_count_b >= min_words)]
        
        cleaned_size = len(df)
        logger.info(f"Dataset cleaned: {original_size} -> {cleaned_size} pairs ({original_size - cleaned_size} removed)")
        
        return df.reset_index(drop=True)
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, 
                     test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets
        
        Args:
            df: Dataset DataFrame
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        logger.info(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, str]:
        """Save dataset splits to separate CSV files
        
        Args:
            train_df: Training set DataFrame
            val_df: Validation set DataFrame  
            test_df: Test set DataFrame
            
        Returns:
            Dictionary with file paths
        """
        files = {}
        
        # Save each split
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            filepath = os.path.join(self.data_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            files[name] = filepath
            logger.info(f"Saved {name} set: {len(df)} pairs to {filepath}")
        
        return files
    
    def prepare_from_synthetic(self, synthetic_pairs_path: str = "fine_tuning/data/pairs.csv") -> Dict[str, str]:
        """Prepare training dataset from synthetic pairs
        
        Args:
            synthetic_pairs_path: Path to synthetic pairs CSV
            
        Returns:
            Dictionary with paths to prepared dataset files
        """
        logger.info("Preparing dataset from synthetic pairs...")
        
        if not os.path.exists(synthetic_pairs_path):
            raise FileNotFoundError(f"Synthetic pairs file not found: {synthetic_pairs_path}")
        
        # Load and validate
        df = self.load_pairs_csv(synthetic_pairs_path)
        stats = self.validate_dataset(df)
        
        if stats['validation_errors']:
            logger.warning("Dataset validation issues:")
            for error in stats['validation_errors']:
                logger.warning(f"  - {error}")
        
        # Clean dataset
        df = self.clean_dataset(df)
        
        if len(df) < 30:
            raise ValueError(f"Dataset too small after cleaning: {len(df)} pairs")
        
        # Split dataset
        train_df, val_df, test_df = self.split_dataset(df)
        
        # Save splits
        files = self.save_splits(train_df, val_df, test_df)
        
        # Save metadata
        metadata = {
            'original_pairs': stats['total_pairs'],
            'cleaned_pairs': len(df),
            'train_pairs': len(train_df),
            'val_pairs': len(val_df),
            'test_pairs': len(test_df),
            'positive_ratio': len(df[df['label'] == 1]) / len(df),
            'avg_text_length': stats['avg_text_length']
        }
        
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        files['metadata'] = metadata_path
        
        logger.info("Dataset preparation complete!")
        logger.info(f"  Training pairs: {len(train_df)}")
        logger.info(f"  Validation pairs: {len(val_df)}")
        logger.info(f"  Test pairs: {len(test_df)}")
        
        return files
    
    def augment_with_corpus(self, corpus_sentences: List[str], target_size: int = 500) -> pd.DataFrame:
        """Augment dataset with additional pairs from corpus (future enhancement)
        
        Args:
            corpus_sentences: List of sentences from existing corpus
            target_size: Target number of pairs to generate
            
        Returns:
            DataFrame with augmented pairs
        """
        # This is a placeholder for future enhancement
        # Could implement more sophisticated paraphrase generation using:
        # - Back-translation
        # - Sentence embedding similarity
        # - Template-based generation
        
        logger.info("Corpus augmentation not implemented yet - using synthetic pairs only")
        return pd.DataFrame(columns=['text_a', 'text_b', 'label'])


def prepare_training_data(synthetic_pairs_path: str = "fine_tuning/data/pairs.csv") -> Dict[str, str]:
    """Convenience function to prepare training data
    
    Args:
        synthetic_pairs_path: Path to synthetic pairs CSV
        
    Returns:
        Dictionary with paths to prepared dataset files
    """
    preparator = DatasetPreparator()
    return preparator.prepare_from_synthetic(synthetic_pairs_path)


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Check if synthetic pairs exist
        pairs_path = "fine_tuning/data/pairs.csv"
        if not os.path.exists(pairs_path):
            print(f"❌ Synthetic pairs not found at {pairs_path}")
            print("💡 Run 'python scripts/generate_synthetic_pairs.py' first")
        else:
            files = prepare_training_data(pairs_path)
            print("✅ Dataset preparation complete!")
            print("📁 Generated files:")
            for name, path in files.items():
                print(f"  {name}: {path}")
    
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        print(f"❌ Error: {e}")