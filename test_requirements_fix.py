#!/usr/bin/env python3
"""
Test script to validate our requirements.txt fixes
This script tests if the issues that caused the original pip error are resolved.
"""

import sys
import subprocess
import tempfile
import os

def test_setuptools_fix():
    """Test that setuptools.build_meta is available"""
    print("🔧 Testing setuptools.build_meta availability...")
    try:
        import setuptools.build_meta
        print("✅ setuptools.build_meta is available")
        return True
    except ImportError as e:
        print(f"❌ setuptools.build_meta not available: {e}")
        return False

def test_requirements_parsing():
    """Test that requirements.txt can be parsed without errors"""
    print("\n📋 Testing requirements.txt parsing...")
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        # Check for problematic patterns that caused the original error
        problematic_patterns = [
            'https://github.com/explosion/spacy-models',  # Direct wheel URLs
            'numpy>=1.21.0,<1.25.0',  # Overly restrictive numpy constraint
            'sentence-transformers==2.2.2',  # Exact old version
            'scikit-learn==1.3.0'  # Exact old version
        ]
        
        issues = []
        for pattern in problematic_patterns:
            if pattern in content:
                issues.append(f"Found problematic pattern: {pattern}")
        
        if issues:
            print("❌ Requirements.txt still has problematic patterns:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ Requirements.txt looks good - no problematic patterns found")
            # Success path
            
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def test_pip_dry_run():
    """Test pip install in dry-run mode to check for obvious issues"""
    print("\n🧪 Testing pip install dry-run...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            '--dry-run', '--quiet', '-r', 'requirements.txt'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Pip dry-run completed successfully")
            # Success path
        else:
            print(f"❌ Pip dry-run failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Pip dry-run timed out (network issues?)")
        return False
    except Exception as e:
        print(f"❌ Error running pip dry-run: {e}")
        return False

def main():
    print("🚀 DocInsight Requirements Validation")
    print("=" * 40)
    
    os.chdir('/home/runner/work/DocInsight/DocInsight')
    
    tests = [
        test_setuptools_fix,
        test_requirements_parsing,
        test_pip_dry_run
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! The requirements.txt fixes should resolve the original error.")
    else:
        print("⚠️ Some tests failed. The original issue may persist.")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)