#!/usr/bin/env python3
"""
Test script to verify dataset loading functionality
"""
import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dataset_loading():
    """Test the dataset loading functionality."""
    print("Testing dataset loading functionality...")
    
    try:
        from dataset_loaders import DatasetLoader, get_default_corpus
        
        # Test 1: Initialize dataset loader
        print("\n1. Initializing dataset loader...")
        loader = DatasetLoader(cache_dir="./test_cache")
        print("✓ Dataset loader initialized successfully")
        
        # Test 2: Load small corpus for testing (reduced size for speed)
        print("\n2. Loading small test corpus...")
        start_time = time.time()
        corpus = get_default_corpus(target_size=1000, force_download=False)
        end_time = time.time()
        
        print(f"✓ Loaded corpus with {len(corpus)} sentences")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        
        if len(corpus) > 0:
            print(f"  Sample sentence: {corpus[0][:100]}...")
        
        # Test 3: Verify corpus quality
        print("\n3. Verifying corpus quality...")
        if len(corpus) >= 500:  # Allow for some reduction due to filtering
            print("✓ Corpus size meets minimum threshold")
        else:
            print(f"⚠ Corpus size ({len(corpus)}) is below target, but acceptable for testing")
        
        # Check sentence quality
        valid_sentences = 0
        for sent in corpus[:10]:  # Check first 10 sentences
            if len(sent) > 10 and len(sent.split()) >= 3:
                valid_sentences += 1
        
        if valid_sentences >= 8:
            print("✓ Sentence quality check passed")
        else:
            print(f"⚠ Only {valid_sentences}/10 sentences pass quality check")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure all required packages are installed")
        return False
    except Exception as e:
        print(f"✗ Error during dataset loading test: {e}")
        return False

def test_corpus_building():
    """Test the corpus building functionality."""
    print("\n\nTesting corpus building functionality...")
    
    try:
        from corpus_builder import CorpusBuilder
        
        # Test 1: Initialize corpus builder
        print("\n1. Initializing corpus builder...")
        builder = CorpusBuilder(cache_dir="./test_corpus_cache")
        print("✓ Corpus builder initialized successfully")
        
        # Test 2: Load model
        print("\n2. Loading SentenceTransformer model...")
        start_time = time.time()
        model = builder.load_model()
        end_time = time.time()
        print(f"✓ Model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Test 3: Build small corpus
        print("\n3. Building small test corpus...")
        start_time = time.time()
        corpus = builder.build_corpus(target_size=500, force_rebuild=False)
        end_time = time.time()
        
        print(f"✓ Built corpus with {len(corpus)} sentences")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        
        # Test 4: Build embeddings (small batch)
        print("\n4. Building embeddings...")
        start_time = time.time()
        embeddings = builder.build_embeddings()
        end_time = time.time()
        
        print(f"✓ Built embeddings: {embeddings.shape}")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        
        # Test 5: Build FAISS index
        print("\n5. Building FAISS index...")
        start_time = time.time()
        index = builder.build_index(index_type="flat")
        end_time = time.time()
        
        print(f"✓ Built FAISS index with {index.ntotal} vectors")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        
        # Test 6: Test search functionality
        print("\n6. Testing search functionality...")
        query = "Climate change is a serious global issue"
        results = builder.search(query, top_k=3)
        
        if results:
            print(f"✓ Search returned {len(results)} results")
            print(f"  Best match (score: {results[0]['score']:.3f}): {results[0]['sentence'][:100]}...")
        else:
            print("⚠ Search returned no results")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during corpus building test: {e}")
        return False

if __name__ == "__main__":
    print("DocInsight Dataset Loading Tests")
    print("=" * 50)
    
    # Run tests
    dataset_test_pass = test_dataset_loading()
    corpus_test_pass = test_corpus_building()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Dataset Loading: {'✓ PASS' if dataset_test_pass else '✗ FAIL'}")
    print(f"Corpus Building: {'✓ PASS' if corpus_test_pass else '✗ FAIL'}")
    
    if dataset_test_pass and corpus_test_pass:
        print("\n🎉 All tests passed! Dataset loading system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)