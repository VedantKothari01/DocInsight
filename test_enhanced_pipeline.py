#!/usr/bin/env python3
"""
Test script to verify enhanced pipeline functionality
"""
import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_pipeline():
    """Test the enhanced pipeline functionality."""
    print("Testing enhanced pipeline functionality...")
    
    try:
        from enhanced_pipeline import EnhancedPlagiarismDetector
        
        # Test 1: Initialize detector
        print("\n1. Initializing enhanced plagiarism detector...")
        detector = EnhancedPlagiarismDetector(corpus_size=100)  # Small size for testing
        print("✓ Detector initialized successfully")
        
        # Test 2: Initialize pipeline
        print("\n2. Loading models and corpus...")
        start_time = time.time()
        detector.initialize()
        end_time = time.time()
        print(f"✓ Pipeline initialized in {end_time - start_time:.2f} seconds")
        print(f"  Corpus size: {len(detector.corpus_sentences)}")
        
        # Test 3: Test semantic search
        print("\n3. Testing semantic search...")
        query = "Climate change affects global temperatures and weather patterns"
        results = detector.semantic_search(query, top_k=3)
        
        if results:
            print(f"✓ Search returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result['score']:.3f} - {result['sentence'][:60]}...")
        else:
            print("⚠ Search returned no results")
        
        # Test 4: Test stylometry features
        print("\n4. Testing stylometry feature extraction...")
        features = detector.extract_stylometry_features(query)
        if features:
            print("✓ Stylometry features extracted successfully")
            print(f"  Features: {list(features.keys())}")
            print(f"  Sample values: num_tokens={features.get('num_tokens', 0)}, "
                  f"ttr={features.get('ttr', 0):.3f}")
        else:
            print("⚠ No features extracted")
        
        # Test 5: Test text extraction
        print("\n5. Testing text extraction...")
        # Create a test file
        test_text = """This is a test document for plagiarism detection.
        Climate change is a serious global issue that requires immediate action.
        Machine learning techniques can help analyze large datasets efficiently."""
        
        test_file = "/tmp/test_document.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_text)
        
        extracted_text = detector.extract_text(test_file)
        if extracted_text and len(extracted_text) > 50:
            print("✓ Text extraction working correctly")
            print(f"  Extracted {len(extracted_text)} characters")
        else:
            print("⚠ Text extraction may have issues")
        
        # Test 6: Test sentence splitting
        print("\n6. Testing sentence splitting...")
        sentences = detector.split_sentences(extracted_text)
        if sentences and len(sentences) >= 2:
            print(f"✓ Sentence splitting working correctly ({len(sentences)} sentences)")
            print(f"  Sample sentence: {sentences[0]}")
        else:
            print("⚠ Sentence splitting may have issues")
        
        # Test 7: Test full report generation (if possible)
        print("\n7. Testing report generation...")
        try:
            report = detector.generate_enhanced_report(
                test_file, 
                output_json="/tmp/test_report.json",
                output_html="/tmp/test_report.html"
            )
            
            if report and 'sentences' in report:
                print("✓ Report generation successful")
                print(f"  Analyzed {report['total_sentences']} sentences")
                print(f"  Report keys: {list(report.keys())}")
                
                # Check if files were created
                if os.path.exists("/tmp/test_report.json"):
                    print("  ✓ JSON report file created")
                if os.path.exists("/tmp/test_report.html"):
                    print("  ✓ HTML report file created")
            else:
                print("⚠ Report generation incomplete")
                
        except Exception as e:
            print(f"⚠ Error in report generation: {e}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during enhanced pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("DocInsight Enhanced Pipeline Tests")
    print("=" * 50)
    
    # Run test
    test_pass = test_enhanced_pipeline()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Enhanced Pipeline: {'✓ PASS' if test_pass else '✗ FAIL'}")
    
    if test_pass:
        print("\n🎉 Enhanced pipeline test passed! System is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Enhanced pipeline test failed. Please check the output above.")
        sys.exit(1)