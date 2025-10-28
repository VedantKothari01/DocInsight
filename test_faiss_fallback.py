"""Test FAISS fallback path using numpy cosine similarity when FAISS unavailable or disabled."""
import os
import types

from enhanced_pipeline import SemanticSearchEngine


def test_fallback_search():
    engine = SemanticSearchEngine()
    # Force FAISS unavailability
    engine.index = None  # ensure fallback path
    assert engine.build_index(["alpha beta gamma", "machine learning models", "climate change impact"])
    results = engine.search("machine learning")
    assert results, "Fallback search returned no results"
    # Top result should mention machine learning or models
    assert any("machine" in r['sentence'] for r in results[:2])

if __name__ == "__main__":
    test_fallback_search()
    print("Fallback semantic search test passed")
