#!/usr/bin/env python3
"""
Self-check script for DocInsight Phase 2+ implementation

Validates code quality, checks for unused imports, TODOs/FIXMEs,
and performs basic syntax validation.
"""

import os
import re
import sys
import ast
import logging
import importlib
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple

logger = logging.getLogger(__name__)


class DocInsightSelfChecker:
    """Performs comprehensive self-checks on DocInsight codebase"""
    
    def __init__(self, project_root: str = "."):
        """Initialize self-checker
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.issues = []
        self.warnings = []
        self.python_files = []
        
        # Collect all Python files
        self._find_python_files()
    
    def _find_python_files(self) -> None:
        """Find all Python files in the project"""
        exclude_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
            'node_modules', 'dist', 'build', '.vscode', '.idea'
        }
        
        for py_file in self.project_root.glob('**/*.py'):
            # Skip files in excluded directories
            if any(part in exclude_patterns for part in py_file.parts):
                continue
            self.python_files.append(py_file)
        
        logger.info(f"Found {len(self.python_files)} Python files to check")
    
    def check_syntax(self) -> bool:
        """Check syntax of all Python files"""
        print("🔍 Checking Python syntax...")
        syntax_errors = []
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse with AST to check syntax
                ast.parse(content)
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}:{e.lineno}: {e.msg}")
            except Exception as e:
                self.warnings.append(f"Could not check syntax for {py_file}: {e}")
        
        if syntax_errors:
            self.issues.extend(syntax_errors)
            print(f"❌ Found {len(syntax_errors)} syntax errors")
            for error in syntax_errors:
                print(f"  {error}")
            return False
        else:
            print("✅ All Python files have valid syntax")
            return True
    
    def check_imports(self) -> bool:
        """Check for unused imports"""
        print("📦 Checking for unused imports...")
        unused_imports = []
        
        for py_file in self.python_files:
            try:
                unused = self._find_unused_imports(py_file)
                if unused:
                    unused_imports.extend([f"{py_file}: {imp}" for imp in unused])
            except Exception as e:
                self.warnings.append(f"Could not check imports for {py_file}: {e}")
        
        if unused_imports:
            # Only report as warnings, not errors (imports might be used dynamically)
            self.warnings.extend(unused_imports)
            print(f"⚠️ Found {len(unused_imports)} potentially unused imports")
            for warning in unused_imports[-5:]:  # Show last 5
                print(f"  {warning}")
            if len(unused_imports) > 5:
                print(f"  ... and {len(unused_imports) - 5} more")
        else:
            print("✅ No unused imports detected")
        
        return True  # Don't fail on unused imports
    
    def _find_unused_imports(self, py_file: Path) -> List[str]:
        """Find unused imports in a Python file (basic check)"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file
            tree = ast.parse(content)
            
            # Extract imports
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
                    for alias in node.names:
                        imports.add(alias.name)
            
            # Check if imports are used (simple string search)
            unused = []
            for imp in imports:
                # Skip common imports that might be used dynamically
                if imp in {'os', 'sys', 'logging', 'typing', 'Union', 'Optional', 'List', 'Dict', 'Any'}:
                    continue
                
                # Simple check: is the import name mentioned anywhere in the code?
                if imp not in content.replace(f"import {imp}", "").replace(f"from {imp}", ""):
                    unused.append(imp)
            
            return unused
            
        except Exception:
            return []
    
    def check_todos_fixmes(self) -> bool:
        """Check for development task comments"""
        print("📝 Checking for development task comments...")
        task_keywords = ['TODO', 'FIXME']
        comment_pattern = re.compile(r'#.*?\b(' + '|'.join(task_keywords) + r')\b.*', re.IGNORECASE)
        todos_found = []
        
        for py_file in self.python_files:
            # Skip this self-check script to avoid false positives
            if py_file.name == 'run_self_check.py':
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    match = comment_pattern.search(line)
                    if match:
                        todos_found.append(f"{py_file}:{line_num}: {line.strip()}")
            
            except Exception as e:
                self.warnings.append(f"Could not check development tasks in {py_file}: {e}")
        
        if todos_found:
            self.issues.extend(todos_found)
            print(f"❌ Found {len(todos_found)} development task comments")
            for todo in todos_found:
                print(f"  {todo}")
            return False
        else:
            print("✅ No development task comments found")
            return True
    
    def check_module_imports(self) -> bool:
        """Test importing all new modules"""
        print("🔧 Testing module imports...")
        
        # Test only the new modules we created, not existing ones with heavy dependencies
        new_modules = [
            'stylometry.features',
            'scoring.aggregate'
        ]
        
        # Test individual files without module structure for those with dependencies
        standalone_files = [
            'ingestion/citation_mask.py',
            'ingestion/section_parser.py',
            'fine_tuning/dataset_prep.py',
            'fine_tuning/fine_tune_semantic.py',
            'fine_tuning/train_ai_likeness.py'
        ]
        
        import_errors = []
        
        # Save current directory and change to project root
        original_dir = os.getcwd()
        os.chdir(self.project_root)
        
        # Add current directory to Python path for imports
        import sys
        original_path = sys.path[:]
        sys.path.insert(0, str(self.project_root))
        
        try:
            # Test module imports
            for module_name in new_modules:
                try:
                    # Try to import the module
                    module = importlib.import_module(module_name)
                    # Try to reload to catch any issues
                    importlib.reload(module)
                except ImportError as e:
                    import_errors.append(f"{module_name}: {e}")
                except Exception as e:
                    import_errors.append(f"{module_name}: Unexpected error - {e}")
            
            # Test standalone file syntax
            for file_path in standalone_files:
                try:
                    full_path = self.project_root / file_path
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Just check syntax, don't import
                    ast.parse(content)
                except SyntaxError as e:
                    import_errors.append(f"{file_path}: Syntax error - {e}")
                except Exception as e:
                    import_errors.append(f"{file_path}: Could not check - {e}")
                    
        finally:
            os.chdir(original_dir)
            sys.path[:] = original_path
        
        if import_errors:
            self.issues.extend(import_errors)
            print(f"❌ Found {len(import_errors)} import/syntax errors")
            for error in import_errors:
                print(f"  {error}")
            return False
        else:
            print("✅ All testable modules import successfully")
            return True
    
    def check_file_structure(self) -> bool:
        """Check that required files and directories exist"""
        print("📁 Checking file structure...")
        
        required_paths = [
            'ingestion/citation_mask.py',
            'ingestion/section_parser.py',
            'stylometry/__init__.py',
            'stylometry/features.py',
            'scoring/aggregate.py',
            'fine_tuning/__init__.py',
            'fine_tuning/dataset_prep.py',
            'fine_tuning/fine_tune_semantic.py',
            'fine_tuning/train_ai_likeness.py',
            'fine_tuning/data/.gitkeep',
            'scripts/generate_synthetic_pairs.py',
            'scripts/run_fine_tuning.sh',
            'scripts/run_ai_likeness_training.sh'
        ]
        
        missing_files = []
        
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            self.issues.extend([f"Missing file: {f}" for f in missing_files])
            print(f"❌ Found {len(missing_files)} missing files")
            for missing in missing_files:
                print(f"  {missing}")
            return False
        else:
            print("✅ All required files present")
            return True
    
    def check_config_completeness(self) -> bool:
        """Check that config.py has all required Phase 2+ settings"""
        print("⚙️ Checking configuration completeness...")
        
        required_config_vars = [
            'MODEL_FINE_TUNED_PATH',
            'AI_LIKENESS_MODEL_PATH',
            'WEIGHT_SEMANTIC',
            'WEIGHT_STYLO',
            'WEIGHT_AI',
            'CITATION_MASKING_ENABLED',
            'ACADEMIC_SECTIONS',
            'CITATION_PATTERNS'
        ]
        
        missing_configs = []
        
        try:
            config_path = self.project_root / 'config.py'
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            for var in required_config_vars:
                if var not in config_content:
                    missing_configs.append(var)
        
        except Exception as e:
            self.issues.append(f"Could not check config.py: {e}")
            return False
        
        if missing_configs:
            self.issues.extend([f"Missing config variable: {var}" for var in missing_configs])
            print(f"❌ Found {len(missing_configs)} missing config variables")
            for missing in missing_configs:
                print(f"  {missing}")
            return False
        else:
            print("✅ Configuration appears complete")
            return True
    
    def run_all_checks(self) -> bool:
        """Run all self-checks and return overall result"""
        print("🧪 DocInsight Phase 2+ Self-Check")
        print("=" * 40)
        
        checks = [
            ("File Structure", self.check_file_structure),
            ("Configuration", self.check_config_completeness),
            ("Python Syntax", self.check_syntax),
            ("Module Imports", self.check_module_imports),
            ("Import Usage", self.check_imports),
            ("TODO/FIXME", self.check_todos_fixmes)
        ]
        
        results = []
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                print(f"❌ Check '{check_name}' failed: {e}")
                self.issues.append(f"Check failure: {check_name} - {e}")
                results.append(False)
            print()  # Add spacing between checks
        
        # Summary
        print("📊 Self-Check Summary")
        print("=" * 20)
        
        passed_checks = sum(results)
        total_checks = len(results)
        
        print(f"Checks passed: {passed_checks}/{total_checks}")
        print(f"Issues found: {len(self.issues)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.issues:
            print("\n❌ Issues that need attention:")
            for issue in self.issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print(f"\n⚠️ Warnings (first 5 of {len(self.warnings)}):")
            for warning in self.warnings[:5]:
                print(f"  • {warning}")
        
        # Overall result
        success = len(self.issues) == 0
        
        if success:
            print("\n🎉 All checks passed! Code is ready for production.")
        else:
            print(f"\n💥 {len(self.issues)} issues found. Please fix before merge.")
        
        return success


def main():
    """Main function for script execution"""
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs
    
    checker = DocInsightSelfChecker()
    success = checker.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()