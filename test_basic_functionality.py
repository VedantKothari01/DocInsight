#!/usr/bin/env python3
"""
Basic functionality test for DocInsight core components

Tests text processing and scoring logic without requiring heavy ML dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_text_processing():
    """Test basic text processing functionality"""
    print("🔤 Testing text processing...")
    
    from enhanced_pipeline import TextExtractor, SentenceProcessor
    
    # Test text extraction
    extractor = TextExtractor()
    text = extractor.extract_text('sample_document.txt')
    assert len(text) > 0
    print("✅ Text extraction works")
    
    # Test sentence processing
    processor = SentenceProcessor()
    sentences = processor.split_sentences(text)
    assert len(sentences) > 0
    print(f"✅ Sentence processing works ({len(sentences)} sentences found)")
    
    return sentences

def test_stylometry_basic():
    """Test basic stylometry without spaCy"""
    print("\n📊 Testing basic stylometry...")
    
    # Import textstat directly for basic readability
    import textstat
    
    test_sentence = "This is a test sentence for readability analysis."
    flesch_score = textstat.flesch_reading_ease(test_sentence)
    assert isinstance(flesch_score, (int, float))
    print(f"✅ Flesch reading ease score: {flesch_score}")
    
    return True

def test_scoring_logic():
    """Test scoring logic with mock data"""
    print("\n🎯 Testing scoring logic...")
    
    from scoring.core import SentenceClassifier, SpanClusterer, DocumentScorer
    
    # Test initialization
    classifier = SentenceClassifier()
    clusterer = SpanClusterer()
    scorer = DocumentScorer()
    print("✅ Scoring components initialized")
    
    # Test with mock sentence results
    mock_sentence_results = [
        {
            'sentence': 'High risk sentence',
            'risk_level': 'HIGH',
            'confidence_score': 0.8
        },
        {
            'sentence': 'Medium risk sentence',
            'risk_level': 'MEDIUM', 
            'confidence_score': 0.5
        },
        {
            'sentence': 'Low risk sentence',
            'risk_level': 'LOW',
            'confidence_score': 0.1
        }
    ]
    
    # Test span clustering
    spans = clusterer.cluster_risk_spans(mock_sentence_results)
    print(f"✅ Span clustering works ({len(spans)} spans found)")
    
    # Test document scoring
    metrics = scorer.compute_originality_score(mock_sentence_results, spans)
    assert 0 <= metrics['originality_score'] <= 1
    print(f"✅ Document scoring works (originality: {metrics['originality_score']:.2%})")
    
    return True

def test_corpus_builder():
    """Test corpus builder functionality"""
    print("\n📚 Testing corpus builder...")
    
    from corpus_builder import SimpleCorpusBuilder
    
    builder = SimpleCorpusBuilder()
    demo_corpus = builder.get_demo_corpus()
    assert len(demo_corpus) > 0
    print(f"✅ Demo corpus loaded ({len(demo_corpus)} sentences)")
    
    return True

def test_configuration():
    """Test configuration values"""
    print("\n⚙️ Testing configuration...")
    
    import config
    
    # Verify key configuration values
    assert hasattr(config, 'AGGREGATION_WEIGHTS')
    assert hasattr(config, 'FUSION_WEIGHTS')
    assert hasattr(config, 'HIGH_RISK_THRESHOLD')
    
    weights = config.AGGREGATION_WEIGHTS
    total = weights['alpha'] + weights['beta'] + weights['gamma']
    assert 0.9 < total < 1.1  # Should sum to ~1
    
    print("✅ Configuration values are valid")
    return True

def main():
    """Run basic functionality tests"""
    print("🧪 DocInsight Basic Functionality Tests\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Text Processing", test_text_processing),
        ("Stylometry Basic", test_stylometry_basic),
        ("Scoring Logic", test_scoring_logic),
        ("Corpus Builder", test_corpus_builder)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED\n")
            else:
                print(f"❌ {test_name} test FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}\n")
            import traceback
            traceback.print_exc()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        print("📋 Core DocInsight functionality is working correctly.")
        print("\n📝 To run full functionality:")
        print("1. Install ML dependencies: pip install sentence-transformers faiss-cpu")
        print("2. Install spaCy model: python -m spacy download en_core_web_sm")
        print("3. Run Streamlit app: streamlit run streamlit_app.py")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)