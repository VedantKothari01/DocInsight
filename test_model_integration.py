"""Test semantic model integration metadata.

Ensures SemanticSearchEngine reports model_source and handles fallback gracefully.
"""
import os
from enhanced_pipeline import SemanticSearchEngine


def test_semantic_model_metadata():
    engine = SemanticSearchEngine()
    assert engine.model is not None, "Model failed to load"
    assert engine.model_source in {"base", "fine_tuned"}, f"Unexpected source {engine.model_source}"
    # Build tiny index to ensure no crash
    engine.build_index(["Test sentence one.", "Another test sentence."])
    res = engine.search("Test sentence", top_k=2)
    assert isinstance(res, list)
    return True

if __name__ == "__main__":
    ok = test_semantic_model_metadata()
    print("Model integration test passed" if ok else "Model integration test failed")
