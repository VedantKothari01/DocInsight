#!/bin/bash

# DocInsight - One-Command Setup and Run Script
# This script installs dependencies and runs the Streamlit app

set -e  # Exit on any error

echo "🚀 DocInsight - Document Originality Analysis"
echo "============================================="

# Check Python version
python_version=$(python --version 2>&1 | cut -d" " -f2 | cut -d"." -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Install dependencies with better error handling
echo "📦 Installing dependencies..."
echo "   Using pip with extended timeout and retries..."

# Upgrade pip, setuptools, and wheel first to avoid build issues
echo "   📝 Upgrading build tools..."
pip install --upgrade --timeout 300 setuptools wheel pip || {
    echo "⚠️ Warning: Could not upgrade build tools, continuing with existing versions"
}

# Install dependencies with retries and extended timeout
echo "   📦 Installing package dependencies..."
pip install --timeout 300 --retries 3 -r requirements.txt || {
    echo "❌ Error: Failed to install all dependencies"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "   • Check your internet connection"
    echo "   • Try running: pip install --upgrade pip setuptools wheel"
    echo "   • Try installing packages individually:"
    echo "     pip install streamlit numpy pandas nltk"
    echo "     pip install sentence-transformers transformers"
    echo "     pip install spacy faiss-cpu scikit-learn"
    echo ""
    echo "   • For PyTorch installation issues, visit: https://pytorch.org/get-started/locally/"
    echo ""
    exit 1
}

# Download NLTK data if not present
echo "📚 Setting up NLTK data..."
python -c "
import nltk
import os
try:
    nltk.data.find('tokenizers/punkt')
    print('✅ NLTK punkt data already available')
except LookupError:
    print('📥 Downloading NLTK punkt data...')
    nltk.download('punkt', quiet=True)
    print('✅ NLTK punkt data downloaded')

try:
    nltk.data.find('tokenizers/punkt_tab')
    print('✅ NLTK punkt_tab data already available')
except LookupError:
    print('📥 Downloading NLTK punkt_tab data...')
    nltk.download('punkt_tab', quiet=True)
    print('✅ NLTK punkt_tab data downloaded')
"

# Download spaCy model if not present
echo "🧠 Setting up spaCy model..."
python -c "
import spacy
import sys
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy en_core_web_sm model already available')
except OSError:
    print('📥 Downloading spaCy en_core_web_sm model...')
    try:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True, capture_output=True, timeout=300)
        print('✅ spaCy en_core_web_sm model downloaded')
    except subprocess.TimeoutExpired:
        print('⚠️ Warning: spaCy model download timed out')
        print('   You can manually install it later with: python -m spacy download en_core_web_sm')
        sys.exit(0)  # Continue without failing
    except subprocess.CalledProcessError as e:
        print(f'⚠️ Warning: spaCy model download failed: {e}')
        print('   You can manually install it later with: python -m spacy download en_core_web_sm')
        sys.exit(0)  # Continue without failing
"

# Test imports
echo "🔍 Testing core imports..."
python -c "
import sys
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Test basic imports first
try:
    import streamlit
    print('✅ Streamlit imported successfully')
    print(f'   Version: {streamlit.__version__}')
except Exception as e:
    print(f'❌ Failed to import streamlit: {e}')
    sys.exit(1)

# Test if the main pipeline can be imported
try:
    from enhanced_pipeline import DocumentAnalysisPipeline
    print('✅ DocumentAnalysisPipeline imported successfully')
except Exception as e:
    print(f'⚠️ Warning: Could not import DocumentAnalysisPipeline: {e}')
    print('   This may be due to missing ML dependencies')
    print('   The basic Streamlit app may still work')

print('✅ Core imports completed')
"

echo ""
echo "🎉 Setup complete! Starting DocInsight..."
echo "📱 Open your browser to: http://localhost:8501"
echo "📄 Upload a document to analyze its originality"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run Streamlit app
streamlit run streamlit_app.py