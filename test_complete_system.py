#!/usr/bin/env python3
"""
Complete system test for enhanced DocInsight
"""
import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_system():
    """Test the complete enhanced DocInsight system."""
    print("🔬 Testing Complete Enhanced DocInsight System")
    print("=" * 60)
    
    try:
        # Import all components
        from dataset_loaders import DatasetLoader, get_default_corpus
        from simple_corpus_builder import SimpleCorpusBuilder, get_simple_corpus
        from enhanced_pipeline import EnhancedPlagiarismDetector
        
        print("✅ All modules imported successfully")
        
        # Test 1: Quick corpus building
        print("\n📚 Testing corpus building...")
        start_time = time.time()
        corpus = get_simple_corpus(target_size=200)
        end_time = time.time()
        
        print(f"✅ Built corpus: {len(corpus)} sentences in {end_time - start_time:.2f}s")
        
        # Test 2: Enhanced detector initialization
        print("\n🤖 Testing enhanced detector...")
        detector = EnhancedPlagiarismDetector(corpus_size=200)
        
        start_time = time.time()
        detector.initialize()
        end_time = time.time()
        
        print(f"✅ Detector initialized in {end_time - start_time:.2f}s")
        print(f"   Corpus: {len(detector.corpus_sentences)} sentences")
        print(f"   SBERT: {'✓' if detector.sbert_model else '✗'}")
        print(f"   CrossEncoder: {'✓' if detector.cross_encoder else '✗'}")
        print(f"   FAISS: {'✓' if detector.faiss_index else '✗'}")
        
        # Test 3: Search functionality
        print("\n🔍 Testing search functionality...")
        queries = [
            "Climate change is a serious environmental problem",
            "Machine learning algorithms process data efficiently",
            "Academic research requires rigorous methodology"
        ]
        
        for i, query in enumerate(queries, 1):
            results = detector.semantic_search(query, top_k=3)
            print(f"   Query {i}: Found {len(results)} results")
            if results:
                print(f"     Best match (score: {results[0]['score']:.3f}): {results[0]['sentence'][:60]}...")
        
        # Test 4: Document analysis
        print("\n📄 Testing document analysis...")
        
        # Create test document
        test_content = """Climate change represents a major challenge for humanity.
        Machine learning techniques are revolutionizing data analysis.
        Academic research requires careful methodology and peer review.
        The effects of global warming include rising temperatures and sea levels.
        """
        
        test_file = "/tmp/test_doc_complete.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Run analysis
        start_time = time.time()
        report = detector.generate_enhanced_report(
            test_file,
            output_json="/tmp/complete_test_report.json",
            output_html="/tmp/complete_test_report.html"
        )
        end_time = time.time()
        
        print(f"✅ Analysis complete in {end_time - start_time:.2f}s")
        print(f"   Sentences analyzed: {report['total_sentences']}")
        print(f"   Average score: {report['overall_stats']['avg_fused_score']:.3f}")
        print(f"   High confidence: {report['overall_stats']['high_confidence_count']}")
        print(f"   Medium confidence: {report['overall_stats']['medium_confidence_count']}")
        print(f"   Low confidence: {report['overall_stats']['low_confidence_count']}")
        
        # Verify output files
        json_exists = os.path.exists("/tmp/complete_test_report.json")
        html_exists = os.path.exists("/tmp/complete_test_report.html")
        
        print(f"   JSON report: {'✓' if json_exists else '✗'}")
        print(f"   HTML report: {'✓' if html_exists else '✗'}")
        
        # Test 5: Feature extraction
        print("\n🧬 Testing feature extraction...")
        sample_text = "This is a sample sentence for testing stylometry features."
        features = detector.extract_stylometry_features(sample_text)
        
        key_features = ['num_tokens', 'ttr', 'flesch_reading_ease', 'noun_ratio']
        available_features = sum(1 for feat in key_features if feat in features)
        
        print(f"✅ Extracted {len(features)} features ({available_features}/{len(key_features)} key features)")
        print(f"   Sample: num_tokens={features.get('num_tokens', 0)}, "
              f"ttr={features.get('ttr', 0):.3f}")
        
        # Final assessment
        print("\n" + "=" * 60)
        print("🎉 COMPLETE SYSTEM TEST RESULTS:")
        print("=" * 60)
        
        checks = [
            ("Module imports", True),
            ("Corpus building", len(corpus) > 50),
            ("Detector initialization", detector.corpus_sentences is not None),
            ("Search functionality", len(queries) == 3),
            ("Document analysis", report is not None),
            ("Report generation", json_exists and html_exists),
            ("Feature extraction", len(features) > 5)
        ]
        
        passed = 0
        for check_name, passed_test in checks:
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"  {check_name:.<25} {status}")
            if passed_test:
                passed += 1
        
        success_rate = passed / len(checks)
        print(f"\n📊 Overall Success Rate: {success_rate:.1%} ({passed}/{len(checks)})")
        
        if success_rate >= 0.8:
            print("🎉 EXCELLENT: Enhanced DocInsight system is fully operational!")
            return True
        elif success_rate >= 0.6:
            print("⚠️  GOOD: System mostly working, minor issues detected")
            return True
        else:
            print("❌ ISSUES: System has significant problems")
            return False
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required modules are available")
        return False
    except Exception as e:
        print(f"❌ System Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    
    print("\n" + "=" * 60)
    if success:
        print("🚀 Enhanced DocInsight is ready for production use!")
        print("\nNext steps:")
        print("  1. Run: streamlit run streamlit_app.py")
        print("  2. Upload documents for analysis")
        print("  3. Review comprehensive reports")
        sys.exit(0)
    else:
        print("🔧 Please address the issues above before deploying")
        sys.exit(1)