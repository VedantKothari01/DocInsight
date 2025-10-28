#!/usr/bin/env python3
"""
Generate synthetic pairs for fine-tuning

Creates a small supervised dataset for semantic similarity/paraphrase classification.
NOT production grade - simple synthetic transformations for bootstrap training data.
"""

import os
import re
import csv
import random
import logging
from typing import List, Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class SyntheticPairGenerator:
    """Generates synthetic text pairs for training semantic similarity models"""
    
    def __init__(self, output_path: str = "fine_tuning/data/pairs.csv"):
        """Initialize generator
        
        Args:
            output_path: Path to save generated pairs CSV
        """
        self.output_path = output_path
        self.synonyms = {
            'good': ['excellent', 'great', 'fine', 'positive', 'beneficial'],
            'bad': ['poor', 'terrible', 'negative', 'harmful', 'awful'],
            'big': ['large', 'huge', 'massive', 'enormous', 'giant'],
            'small': ['tiny', 'little', 'miniature', 'minor', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'delayed', 'unhurried'],
            'important': ['crucial', 'significant', 'vital', 'essential', 'critical'],
            'show': ['demonstrate', 'exhibit', 'display', 'reveal', 'present'],
            'use': ['utilize', 'employ', 'apply', 'implement', 'adopt'],
            'make': ['create', 'produce', 'generate', 'construct', 'build'],
            'find': ['discover', 'locate', 'identify', 'determine', 'detect'],
            'think': ['believe', 'consider', 'suppose', 'assume', 'conclude']
        }
        
        self.sentence_transformations = [
            self._synonym_replacement,
            self._sentence_reordering,
            self._passive_active_transformation,
            self._word_order_change,
            self._minor_paraphrasing
        ]
    
    def load_source_texts(self, source_dir: str = ".") -> List[str]:
        """Load source texts from available sample files
        
        Args:
            source_dir: Directory to search for source texts
            
        Returns:
            List of sentences extracted from source files
        """
        sentences = []
        
        # Look for sample text files
        text_files = [
            "sample_document.txt",
            "sample_ml_text.txt", 
            "sample_data_science.txt"
        ]
        
        for filename in text_files:
            filepath = os.path.join(source_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"Loading text from {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Simple sentence splitting
                        file_sentences = re.split(r'[.!?]+', content)
                        file_sentences = [s.strip() for s in file_sentences if len(s.strip()) > 20]
                        sentences.extend(file_sentences)
                        logger.info(f"Extracted {len(file_sentences)} sentences from {filename}")
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
        
        # Add some hardcoded academic sentences if no files found
        if not sentences:
            logger.warning("No source files found, using hardcoded examples")
            sentences = [
                "Machine learning algorithms have shown significant improvements in natural language processing tasks",
                "The experimental results demonstrate the effectiveness of our proposed approach",
                "Data preprocessing is a crucial step in any machine learning pipeline",
                "Neural networks require large amounts of training data to achieve optimal performance",
                "Cross-validation techniques help prevent overfitting in predictive models",
                "Feature engineering plays an important role in model accuracy and interpretability",
                "Deep learning models can automatically learn hierarchical representations from raw data",
                "Ensemble methods combine multiple models to improve prediction accuracy",
                "Regularization techniques help reduce model complexity and improve generalization",
                "The training dataset should be representative of the target population"
            ]
        
        logger.info(f"Total sentences loaded: {len(sentences)}")
        return sentences
    
    def generate_positive_pairs(self, sentences: List[str], count: int = 50) -> List[Tuple[str, str, int]]:
        """Generate positive pairs (similar/paraphrased sentences)
        
        Args:
            sentences: Source sentences to transform
            count: Number of positive pairs to generate
            
        Returns:
            List of (text_a, text_b, label=1) tuples
        """
        positive_pairs = []
        
        for _ in range(count):
            # Select random sentence
            original = random.choice(sentences)
            
            # Apply random transformation
            transformation = random.choice(self.sentence_transformations)
            try:
                paraphrased = transformation(original)
                
                # Ensure the paraphrase is different but meaningful
                if paraphrased != original and len(paraphrased) > 10:
                    positive_pairs.append((original, paraphrased, 1))
                else:
                    # Fallback to simple synonym replacement
                    paraphrased = self._synonym_replacement(original)
                    if paraphrased != original:
                        positive_pairs.append((original, paraphrased, 1))
            except Exception as e:
                logger.debug(f"Transformation failed for '{original[:50]}...': {e}")
                continue
        
        logger.info(f"Generated {len(positive_pairs)} positive pairs")
        return positive_pairs
    
    def generate_negative_pairs(self, sentences: List[str], count: int = 50) -> List[Tuple[str, str, int]]:
        """Generate negative pairs (unrelated sentences)
        
        Args:
            sentences: Source sentences to pair
            count: Number of negative pairs to generate
            
        Returns:
            List of (text_a, text_b, label=0) tuples
        """
        negative_pairs = []
        
        for _ in range(count):
            # Select two random, different sentences
            if len(sentences) < 2:
                break
                
            sent1, sent2 = random.sample(sentences, 2)
            
            # Ensure they are sufficiently different (basic check)
            if self._are_sufficiently_different(sent1, sent2):
                negative_pairs.append((sent1, sent2, 0))
        
        logger.info(f"Generated {len(negative_pairs)} negative pairs")
        return negative_pairs
    
    def _synonym_replacement(self, sentence: str) -> str:
        """Replace words with synonyms"""
        words = sentence.split()
        modified_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in self.synonyms and random.random() < 0.3:  # 30% chance
                synonym = random.choice(self.synonyms[word_lower])
                # Preserve original capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                modified_words.append(synonym + word[len(word_lower):])  # Preserve punctuation
            else:
                modified_words.append(word)
        
        return ' '.join(modified_words)
    
    def _sentence_reordering(self, sentence: str) -> str:
        """Reorder clauses in compound sentences"""
        # Simple clause splitting on common conjunctions
        conjunctions = [', and ', ', but ', ', however ', ', therefore ']
        
        for conj in conjunctions:
            if conj in sentence:
                parts = sentence.split(conj)
                if len(parts) == 2:
                    # Reverse the order
                    return f"{parts[1].strip()}{conj}{parts[0].strip()}"
        
        return sentence  # Return unchanged if no reordering possible
    
    def _passive_active_transformation(self, sentence: str) -> str:
        """Simple passive to active voice transformation (basic patterns only)"""
        # Very basic patterns - NOT comprehensive
        passive_patterns = [
            (r'(\w+) is (\w+ed) by (\w+)', r'\3 \2 \1'),  # "X is done by Y" -> "Y done X"
            (r'(\w+) was (\w+ed) by (\w+)', r'\3 \2 \1'),  # "X was done by Y" -> "Y done X"
        ]
        
        for pattern, replacement in passive_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence  # Return unchanged if no transformation possible
    
    def _word_order_change(self, sentence: str) -> str:
        """Minor word order changes in phrases"""
        # Swap adjacent adjectives or adverbs occasionally
        words = sentence.split()
        
        if len(words) < 3:
            return sentence
        
        # Find adjective pairs and occasionally swap them
        for i in range(len(words) - 1):
            if (words[i].endswith('ly') or words[i].endswith('al') or 
                words[i] in ['good', 'bad', 'big', 'small', 'new', 'old']) and random.random() < 0.2:
                if i + 1 < len(words):
                    words[i], words[i + 1] = words[i + 1], words[i]
                    break
        
        return ' '.join(words)
    
    def _minor_paraphrasing(self, sentence: str) -> str:
        """Apply minor paraphrasing changes"""
        # Simple substitutions for common academic phrases
        substitutions = [
            ('results show', 'findings indicate'),
            ('in order to', 'to'),
            ('it is important to note', 'notably'),
            ('a number of', 'several'),
            ('due to the fact that', 'because'),
            ('in the event that', 'if'),
            ('at this point in time', 'currently'),
        ]
        
        modified = sentence
        for original, replacement in substitutions:
            if original in modified.lower():
                # Case-preserving replacement
                if original.title() in modified:
                    modified = modified.replace(original.title(), replacement.title())
                else:
                    modified = modified.replace(original, replacement)
                break  # Only apply one substitution
        
        return modified
    
    def _are_sufficiently_different(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are sufficiently different for negative pairing"""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return True
        
        jaccard_similarity = intersection / union
        return jaccard_similarity < 0.3  # Less than 30% word overlap
    
    def save_pairs(self, pairs: List[Tuple[str, str, int]]) -> None:
        """Save pairs to CSV file
        
        Args:
            pairs: List of (text_a, text_b, label) tuples
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text_a', 'text_b', 'label'])  # Header
            
            for text_a, text_b, label in pairs:
                writer.writerow([text_a, text_b, label])
        
        logger.info(f"Saved {len(pairs)} pairs to {self.output_path}")
    
    def generate_dataset(self, positive_count: int = 100, negative_count: int = 100) -> None:
        """Generate complete dataset with positive and negative pairs
        
        Args:
            positive_count: Number of positive pairs to generate
            negative_count: Number of negative pairs to generate
        """
        logger.info("Starting synthetic pair generation...")
        
        # Load source texts
        sentences = self.load_source_texts()
        
        if not sentences:
            logger.error("No source sentences found. Cannot generate pairs.")
            return
        
        # Generate pairs
        positive_pairs = self.generate_positive_pairs(sentences, positive_count)
        negative_pairs = self.generate_negative_pairs(sentences, negative_count)
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Save to file
        self.save_pairs(all_pairs)
        
        logger.info(f"Dataset generation complete. Total pairs: {len(all_pairs)}")
        logger.info(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")


def main():
    """Main function for script execution"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Generate synthetic pairs
    generator = SyntheticPairGenerator()
    generator.generate_dataset(positive_count=150, negative_count=150)
    
    print(f"✅ Synthetic pairs generated successfully!")
    print(f"📁 Output file: {generator.output_path}")
    print(f"📝 Use this dataset for fine-tuning with: python fine_tuning/fine_tune_semantic.py")


if __name__ == "__main__":
    main()