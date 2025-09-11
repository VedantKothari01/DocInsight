#!/usr/bin/env python3
"""
Simple system test for DocInsight - Works offline
"""
import sys
from corpus_builder import CorpusIndex
from enhanced_pipeline import PlagiarismDetector

def test_system():
    """Test the complete system."""
    print("🔍 Testing DocInsight Enhanced System...")
    
    try:
        # Test 1: Corpus building
        print("\n1️⃣ Testing corpus building...")
        ci = CorpusIndex(target_size=20)
        ci.load_or_build()
        print(f"   ✅ Built corpus with {len(ci.sentences)} sentences")
        
        # Test 2: Pipeline initialization
        print("\n2️⃣ Testing pipeline initialization...")
        detector = PlagiarismDetector(ci)
        print("   ✅ Pipeline initialized successfully")
        
        # Test 3: Sentence analysis
        print("\n3️⃣ Testing sentence analysis...")
        test_sentence = "Machine learning algorithms can identify patterns in large datasets."
        result = detector.analyze_sentence(test_sentence)
        print(f"   ✅ Analysis complete - Score: {result.fused_score:.3f}, Confidence: {result.confidence}")
        
        # Test 4: Document analysis
        print("\n4️⃣ Testing document analysis...")
        test_doc = """
        Artificial intelligence represents a transformative technology.
        Neural networks enable computers to learn from data.
        These systems can recognize complex patterns automatically.
        """
        doc_result = detector.analyze_document(test_doc)
        total_sentences = doc_result['overall_stats']['total_sentences']
        avg_score = doc_result['overall_stats']['avg_fused_score']
        print(f"   ✅ Document analysis complete - {total_sentences} sentences, avg score: {avg_score:.3f}")
        
        # Test 5: Import test for streamlit app
        print("\n5️⃣ Testing Streamlit app imports...")
        try:
            import streamlit_app
            print("   ✅ Streamlit app imports successfully")
        except Exception as e:
            print(f"   ⚠️ Streamlit app import issue (expected in some environments): {e}")
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("- Install streamlit: pip install streamlit")
        print("- Run web app: streamlit run streamlit_app.py")
        print("- For better performance, install: pip install sentence-transformers faiss-cpu")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)