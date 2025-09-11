#!/usr/bin/env python3
"""
DocInsight Demo - Quick Preview Script
Shows what features are available in the Streamlit application
"""

import sys
import os

def show_docinsight_info():
    """Display information about DocInsight features and usage"""
    
    print("🚀 DocInsight - Document Originality Analysis")
    print("=" * 50)
    print()
    
    print("📱 STREAMLIT WEB INTERFACE FEATURES:")
    print("  ✅ Upload documents (PDF, DOCX, TXT)")
    print("  ✅ Real-time originality analysis")
    print("  ✅ Visual risk assessment with color coding")
    print("  ✅ Document-level metrics dashboard")
    print("  ✅ Sentence-by-sentence analysis")
    print("  ✅ Downloadable HTML/JSON reports")
    print("  ✅ Risk span clustering")
    print("  ✅ Interactive filtering and exploration")
    print()
    
    print("🔍 ANALYSIS CAPABILITIES:")
    print("  • Multi-layered similarity detection")
    print("  • Semantic similarity search (SBERT)")
    print("  • Cross-encoder reranking")
    print("  • Stylometric feature analysis")
    print("  • Document-level aggregation scoring")
    print("  • Risk span clustering")
    print()
    
    print("📊 OUTPUT METRICS:")
    print("  • Originality Score (0-100%)")
    print("  • Plagiarized Coverage percentage")
    print("  • Severity Index")
    print("  • Risk level distribution (High/Medium/Low)")
    print("  • Top suspicious sections")
    print()
    
    print("🎯 RISK CLASSIFICATION:")
    print("  🔴 HIGH RISK    - Likely plagiarized content (≥70% similarity)")
    print("  🟡 MEDIUM RISK  - Potentially similar content (≥40% similarity)")
    print("  🟢 LOW RISK     - Original content (<40% similarity)")
    print()
    
    print("🚀 TO RUN DOCINSIGHT:")
    print("  1. One-command setup: bash run_docinsight.sh")
    print("  2. Or manual setup:")
    print("     - pip install -r requirements.txt")
    print("     - streamlit run streamlit_app.py")
    print("  3. Open browser to http://localhost:8501")
    print("  4. Upload a document and get instant analysis!")
    print()
    
    print("💡 SAMPLE WORKFLOW:")
    print("  1. Upload your document")
    print("  2. Wait for analysis (usually 30-60 seconds)")
    print("  3. Review originality score and metrics")
    print("  4. Explore risk spans and suspicious sections")
    print("  5. Download detailed reports")
    print("  6. Make improvements to your document")
    print()
    
    # Check if required files exist
    print("🔍 SYSTEM STATUS:")
    required_files = [
        'streamlit_app.py',
        'enhanced_pipeline.py', 
        'config.py',
        'requirements.txt',
        'run_docinsight.sh'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
    
    print()
    
    # Check if sample documents exist
    sample_docs = ['sample_document.txt', 'sample_data_science.txt', 'sample_ml_text.txt']
    available_samples = [doc for doc in sample_docs if os.path.exists(doc)]
    
    if available_samples:
        print("📄 SAMPLE DOCUMENTS AVAILABLE:")
        for doc in available_samples:
            file_size = os.path.getsize(doc)
            print(f"  📝 {doc} ({file_size} bytes)")
        print("  Use these to test DocInsight functionality!")
    else:
        print("📄 No sample documents found (create your own to test)")
    
    print()
    print("🎉 Ready to analyze documents for originality!")

if __name__ == "__main__":
    show_docinsight_info()