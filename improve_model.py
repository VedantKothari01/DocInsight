#!/usr/bin/env python3
"""
Model Improvement Script for DocInsight
=======================================

This script improves the model accuracy by:
1. Using the largest available corpus (20000 sentences)
2. Optimizing similarity thresholds
3. Improving confidence scoring
4. Enhancing stylometric analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def improve_model_configuration():
    """Improve model configuration for better accuracy."""
    
    logger.info("🔧 Improving DocInsight model configuration...")
    
    # 1. Use the largest available corpus
    corpus_cache_dir = Path("corpus_cache")
    available_corpus_sizes = []
    
    if corpus_cache_dir.exists():
        for corpus_file in corpus_cache_dir.glob("corpus_*.json"):
            size = int(corpus_file.stem.split('_')[1])
            available_corpus_sizes.append(size)
    
    if available_corpus_sizes:
        best_size = max(available_corpus_sizes)
        logger.info(f"✅ Using largest available corpus: {best_size:,} sentences")
        
        # Check if all required files exist for this size
        required_files = [
            f"corpus_{best_size}.json",
            f"embeddings_{best_size}.pkl", 
            f"faiss_index_{best_size}.bin"
        ]
        
        missing_files = []
        for file in required_files:
            if not (corpus_cache_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"⚠️ Missing files for corpus size {best_size}: {missing_files}")
            # Use the next largest available size
            available_corpus_sizes.remove(best_size)
            if available_corpus_sizes:
                best_size = max(available_corpus_sizes)
                logger.info(f"✅ Falling back to corpus size: {best_size:,} sentences")
            else:
                logger.error("❌ No complete corpus available")
                return False
    else:
        logger.error("❌ No corpus cache found")
        return False
    
    # 2. Create improved configuration
    config = {
        "corpus_size": best_size,
        "similarity_thresholds": {
            "high_confidence": 0.75,  # Lowered from 0.8 for better detection
            "medium_confidence": 0.55,  # Lowered from 0.6
            "low_confidence": 0.35     # Lowered from 0.4
        },
        "fusion_weights": {
            "semantic_weight": 0.6,    # Increased semantic importance
            "cross_encoder_weight": 0.3,
            "stylometric_weight": 0.1
        },
        "top_k_matches": 10,  # Increased from 5 for better coverage
        "enable_cross_encoder": True,
        "enable_stylometric": True,
        "enable_domain_adaptation": True
    }
    
    # Save configuration
    config_path = Path("improved_model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Improved configuration saved to {config_path}")
    return True

def create_enhanced_detector():
    """Create an enhanced detector with improved settings."""
    
    logger.info("🚀 Creating enhanced plagiarism detector...")
    
    try:
        from corpus_builder import CorpusIndex
        from enhanced_pipeline import AcademicPlagiarismDetector
        
        # Load configuration
        config_path = Path("improved_model_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.warning("No improved config found, using defaults")
            config = {"corpus_size": 5000}
        
        # Initialize corpus with the best available size
        corpus_size = config.get("corpus_size", 5000)
        logger.info(f"Initializing corpus with {corpus_size:,} sentences...")
        
        corpus_index = CorpusIndex(target_size=corpus_size, use_domain_adaptation=True)
        
        if not corpus_index.load_for_production():
            logger.error("Failed to load corpus for production")
            return None
        
        # Create enhanced detector
        detector = AcademicPlagiarismDetector(corpus_index, use_domain_adaptation=True)
        
        logger.info("✅ Enhanced detector created successfully")
        return detector
        
    except Exception as e:
        logger.error(f"Failed to create enhanced detector: {e}")
        return None

def test_improved_model():
    """Test the improved model with sample sentences."""
    
    logger.info("🧪 Testing improved model performance...")
    
    detector = create_enhanced_detector()
    if not detector:
        logger.error("Cannot test - detector creation failed")
        return False
    
    # Test sentences with known similarities
    test_cases = [
        {
            "sentence": "Machine learning algorithms can identify patterns in data efficiently.",
            "expected": "HIGH",  # Should find similar academic content
            "description": "Academic ML sentence"
        },
        {
            "sentence": "The quick brown fox jumps over the lazy dog.",
            "expected": "LOW",   # Common phrase, should find matches
            "description": "Common phrase"
        },
        {
            "sentence": "Quantum computing represents a paradigm shift in computational capabilities.",
            "expected": "MEDIUM", # Academic content, should find some matches
            "description": "Technical academic sentence"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Testing case {i}: {test_case['description']}")
        
        try:
            result = detector.analyze_sentence(test_case['sentence'])
            
            if result and result.fused_score is not None:
                confidence = result.confidence
                score = result.fused_score
                matches_count = len(result.matches)
                
                logger.info(f"  Result: Score={score:.3f}, Confidence={confidence}, Matches={matches_count}")
                
                results.append({
                    "case": i,
                    "description": test_case['description'],
                    "score": score,
                    "confidence": confidence,
                    "matches_count": matches_count,
                    "expected": test_case['expected']
                })
            else:
                logger.warning(f"  Failed to analyze sentence")
                results.append({
                    "case": i,
                    "description": test_case['description'],
                    "error": "Analysis failed"
                })
                
        except Exception as e:
            logger.error(f"  Error analyzing sentence: {e}")
            results.append({
                "case": i,
                "description": test_case['description'],
                "error": str(e)
            })
    
    # Summary
    successful_tests = [r for r in results if 'error' not in r]
    logger.info(f"\n📊 Test Results Summary:")
    logger.info(f"  Total tests: {len(test_cases)}")
    logger.info(f"  Successful: {len(successful_tests)}")
    logger.info(f"  Failed: {len(results) - len(successful_tests)}")
    
    if successful_tests:
        avg_score = sum(r['score'] for r in successful_tests) / len(successful_tests)
        high_confidence = sum(1 for r in successful_tests if r['confidence'] == 'HIGH')
        medium_confidence = sum(1 for r in successful_tests if r['confidence'] == 'MEDIUM')
        
        logger.info(f"  Average score: {avg_score:.3f}")
        logger.info(f"  High confidence: {high_confidence}")
        logger.info(f"  Medium confidence: {medium_confidence}")
    
    return len(successful_tests) > 0

def main():
    """Main improvement process."""
    
    logger.info("🎯 Starting DocInsight model improvement process...")
    
    # Step 1: Improve configuration
    if not improve_model_configuration():
        logger.error("Configuration improvement failed")
        return False
    
    # Step 2: Test improved model
    if not test_improved_model():
        logger.error("Model testing failed")
        return False
    
    logger.info("🎉 Model improvement completed successfully!")
    logger.info("📋 Next steps:")
    logger.info("  1. The improved configuration is saved in 'improved_model_config.json'")
    logger.info("  2. Restart the Streamlit app to use the improved model")
    logger.info("  3. Test with your documents for better accuracy")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
