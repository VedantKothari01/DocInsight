#!/usr/bin/env python3
"""
Test script for DocInsight Phase 1 implementation

This script validates the core functionality without requiring full dependency installation.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import config
        print("✅ config.py imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    try:
        from scoring import core as scoring  # noqa: F401
        print("✅ scoring.core imported successfully")
    except Exception as e:
        print(f"❌ Failed to import scoring.core: {e}")
        return False
    
    try:
        # Import without initializing models
        from enhanced_pipeline import TextExtractor, SentenceProcessor
        print("✅ enhanced_pipeline core classes imported successfully")
    except Exception as e:
        print(f"❌ Failed to import enhanced_pipeline: {e}")
        return False
    
    try:
        import corpus_builder
        print("✅ corpus_builder.py imported successfully")
    except Exception as e:
        print(f"❌ Failed to import corpus_builder: {e}")
        return False
    
    # No explicit return needed

def test_configuration():
    """Test configuration values"""
    print("\nTesting configuration...")
    
    import config
    
    # Test model names
    assert hasattr(config, 'SBERT_MODEL_NAME')
    assert hasattr(config, 'CROSS_ENCODER_MODEL_NAME')
    assert hasattr(config, 'SPACY_MODEL_NAME')
    print("✅ Model names configured")
    
    # Test thresholds
    assert hasattr(config, 'HIGH_RISK_THRESHOLD')
    assert hasattr(config, 'MEDIUM_RISK_THRESHOLD')
    assert 0 < config.MEDIUM_RISK_THRESHOLD < config.HIGH_RISK_THRESHOLD < 1
    print("✅ Risk thresholds configured correctly")
    
    # Test aggregation weights
    assert hasattr(config, 'AGGREGATION_WEIGHTS')
    weights = config.AGGREGATION_WEIGHTS
    assert 'alpha' in weights and 'beta' in weights and 'gamma' in weights
    total_weight = weights['alpha'] + weights['beta'] + weights['gamma']
    assert 0.9 < total_weight < 1.1  # Should sum to approximately 1
    print("✅ Aggregation weights configured correctly")
    
    # No explicit return needed

def test_scoring_module():
    """Test scoring module functionality"""
    print("\nTesting scoring module...")
    
    from scoring.core import SentenceClassifier, SpanClusterer, DocumentScorer
    
    # Test sentence classifier
    classifier = SentenceClassifier()
    assert hasattr(classifier, 'compute_fused_score')
    assert hasattr(classifier, 'classify_sentence')
    print("✅ SentenceClassifier created successfully")
    
    # Test span clusterer
    clusterer = SpanClusterer()
    assert hasattr(clusterer, 'cluster_risk_spans')
    print("✅ SpanClusterer created successfully")
    
    # Test document scorer
    scorer = DocumentScorer()
    assert hasattr(scorer, 'compute_originality_score')
    assert hasattr(scorer, 'get_top_risk_spans')
    print("✅ DocumentScorer created successfully")
    
    # Test empty results handling
    empty_results = []
    analysis = scorer.compute_originality_score(empty_results, [])
    assert analysis['originality_score'] == 1.0
    assert analysis['plagiarized_coverage'] == 0.0
    print("✅ Empty document handling works correctly")
    
    # No explicit return needed

def test_text_extraction():
    """Test text extraction functionality"""
    print("\nTesting text extraction...")
    
    from enhanced_pipeline import TextExtractor, SentenceProcessor
    
    # Test with sample text file
    sample_text = "This is a test document. It contains multiple sentences. Each sentence should be processed correctly."
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        extractor = TextExtractor()
        extracted_text = extractor.extract_text(temp_file)
        assert extracted_text.strip() == sample_text
        print("✅ Text extraction from TXT file works")
        
        # Test sentence processing
        processor = SentenceProcessor()
        sentences = processor.split_sentences(extracted_text)
        assert len(sentences) >= 2  # Should have multiple sentences
        print(f"✅ Sentence processing works (found {len(sentences)} sentences)")
        
    finally:
        os.unlink(temp_file)
    
    # No explicit return needed

def test_corpus_builder():
    """Test corpus builder functionality"""
    print("\nTesting corpus builder...")
    
    from corpus_builder import SimpleCorpusBuilder
    
    builder = SimpleCorpusBuilder()
    
    # Test demo corpus
    demo_corpus = builder.get_demo_corpus()
    assert isinstance(demo_corpus, list)
    assert len(demo_corpus) > 0
    assert all(isinstance(sentence, str) for sentence in demo_corpus)
    print(f"✅ Demo corpus loaded ({len(demo_corpus)} sentences)")
    
    # No explicit return needed

def test_file_structure():
    """Test that required files exist and have correct structure"""
    print("\nTesting file structure...")
    
    required_files = [
        'config.py',
        'enhanced_pipeline.py',
        'streamlit_app.py',
        'corpus_builder.py',
        'requirements.txt',
        '.gitignore',
        'README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Check requirements.txt has key dependencies
    with open('requirements.txt', 'r') as f:
        req_content = f.read()
        required_deps = ['sentence-transformers', 'streamlit', 'faiss-cpu', 'spacy']
        for dep in required_deps:
            if dep in req_content:
                print(f"✅ {dep} found in requirements.txt")
            else:
                print(f"❌ {dep} missing from requirements.txt")
                return False
    
    # No explicit return needed

def main():
    """Run all tests"""
    print("🧪 DocInsight Phase 1 Implementation Tests\n")
    
    tests = [
        test_file_structure,
        test_imports,
        test_configuration,
        test_scoring_module,
        test_text_extraction,
        test_corpus_builder
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 implementation is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)