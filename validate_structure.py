#!/usr/bin/env python3
"""
Simple validation script for DocInsight Phase 1 structure

This script validates the code structure and basic functionality without 
requiring machine learning dependencies.
"""

import sys
import os
import ast
import tempfile

def validate_python_syntax(file_path):
    """Validate that a Python file has correct syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def check_file_structure():
    """Check that all required files exist"""
    print("📁 Checking file structure...")
    
    required_files = {
        'config.py': 'Configuration module',
        'scoring.py': 'Scoring and analysis module', 
        'enhanced_pipeline.py': 'Main pipeline module',
        'streamlit_app.py': 'Streamlit web interface',
        'corpus_builder.py': 'Corpus building utilities',
        'requirements.txt': 'Python dependencies',
        '.gitignore': 'Git ignore rules',
        'README.md': 'Documentation'
    }
    
    all_exist = True
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"✅ {file} - {description}")
        else:
            print(f"❌ {file} - {description} (MISSING)")
            all_exist = False
    
    return all_exist

def validate_syntax():
    """Validate syntax of all Python files"""
    print("\n🐍 Validating Python syntax...")
    
    python_files = [
        'config.py',
        'scoring.py', 
        'enhanced_pipeline.py',
        'streamlit_app.py',
        'corpus_builder.py',
        'test_implementation.py'
    ]
    
    all_valid = True
    for file in python_files:
        if os.path.exists(file):
            if validate_python_syntax(file):
                print(f"✅ {file}")
            else:
                all_valid = False
        else:
            print(f"⚠️ {file} not found")
    
    return all_valid

def check_config_structure():
    """Check that config.py has required constants"""
    print("\n⚙️ Checking configuration structure...")
    
    required_configs = [
        'SBERT_MODEL_NAME',
        'CROSS_ENCODER_MODEL_NAME', 
        'SPACY_MODEL_NAME',
        'HIGH_RISK_THRESHOLD',
        'MEDIUM_RISK_THRESHOLD',
        'AGGREGATION_WEIGHTS',
        'FUSION_WEIGHTS',
        'SUPPORTED_EXTENSIONS'
    ]
    
    try:
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        all_found = True
        for config in required_configs:
            if config in config_content:
                print(f"✅ {config}")
            else:
                print(f"❌ {config} (MISSING)")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading config.py: {e}")
        return False

def check_requirements():
    """Check requirements.txt has key dependencies"""
    print("\n📦 Checking requirements...")
    
    required_deps = [
        'sentence-transformers',
        'transformers', 
        'faiss-cpu',
        'spacy',
        'streamlit',
        'numpy',
        'nltk',
        'textstat',
        'docx2txt',
        'PyMuPDF'
    ]
    
    try:
        with open('requirements.txt', 'r') as f:
            req_content = f.read().lower()
        
        all_found = True
        for dep in required_deps:
            if dep.lower() in req_content:
                print(f"✅ {dep}")
            else:
                print(f"❌ {dep} (MISSING)")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_gitignore():
    """Check .gitignore has important exclusions"""
    print("\n🚫 Checking .gitignore...")
    
    required_patterns = [
        '__pycache__',
        '*.pyc',
        '*.log',
        '.DS_Store',
        'venv/',
        '.env',
        'corpus_cache/',
        'dataset_cache/'
    ]
    
    try:
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
        
        all_found = True
        for pattern in required_patterns:
            if pattern in gitignore_content:
                print(f"✅ {pattern}")
            else:
                print(f"❌ {pattern} (MISSING)")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading .gitignore: {e}")
        return False

def check_readme():
    """Check README.md has required sections"""
    print("\n📖 Checking README.md...")
    
    required_sections = [
        '# DocInsight',
        'Features',
        'Installation', 
        'Usage',
        'Architecture'
    ]
    
    try:
        with open('README.md', 'r') as f:
            readme_content = f.read()
        
        all_found = True
        for section in required_sections:
            if section in readme_content:
                print(f"✅ {section}")
            else:
                print(f"❌ {section} (MISSING)")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error reading README.md: {e}")
        return False

def validate_phase1_structure():
    """Main validation function"""
    print("🧪 DocInsight Phase 1 Structure Validation\n")
    
    tests = [
        ("File Structure", check_file_structure),
        ("Python Syntax", validate_syntax),
        ("Configuration", check_config_structure),
        ("Requirements", check_requirements),
        ("Git Ignore", check_gitignore),
        ("Documentation", check_readme)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            if test_func():
                print(f"✅ {test_name} validation PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} validation FAILED")
        except Exception as e:
            print(f"❌ {test_name} validation ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure validations passed!")
        print("📋 Phase 1 implementation is structurally complete.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download NLTK data: python -c \"import nltk; nltk.download('punkt')\"") 
        print("3. Run Streamlit app: streamlit run streamlit_app.py")
        return True
    else:
        print("⚠️ Some validations failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = validate_phase1_structure()
    sys.exit(0 if success else 1)