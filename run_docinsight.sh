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

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

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
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy en_core_web_sm model already available')
except OSError:
    print('📥 Downloading spaCy en_core_web_sm model...')
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True, capture_output=True)
    print('✅ spaCy en_core_web_sm model downloaded')
"

# Test imports
echo "🔍 Testing core imports..."
python -c "
from enhanced_pipeline import DocumentAnalysisPipeline
import streamlit
print('✅ All core modules imported successfully')
print(f'✅ Streamlit version: {streamlit.__version__}')
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