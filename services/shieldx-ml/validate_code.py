#!/usr/bin/env python3
"""
Quick validation script for ShieldX ML Service
Checks Python syntax and basic imports without running full tests
"""

import sys
import os
from pathlib import Path
import py_compile
import importlib.util

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}\n")

def check_syntax(file_path: Path) -> bool:
    """Check Python file syntax"""
    try:
        py_compile.compile(str(file_path), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"{Colors.RED}✗ Syntax Error:{Colors.NC} {file_path}")
        print(f"  {e}")
        return False

def check_imports(file_path: Path) -> bool:
    """Check if file can be imported (basic check)"""
    try:
        with open(file_path) as f:
            content = f.read()
        
        # Check for common syntax issues
        if 'import' in content:
            # Basic validation
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    # Check for basic import syntax
                    if stripped.endswith('\\'):
                        continue  # Multi-line import
                    if 'import' in stripped and not any(x in stripped for x in ['(', ',', 'as']):
                        # Simple import
                        parts = stripped.split()
                        if len(parts) < 2:
                            print(f"{Colors.YELLOW}⚠ Warning:{Colors.NC} Line {i}: {stripped}")
        
        return True
    except Exception as e:
        print(f"{Colors.RED}✗ Import Check Failed:{Colors.NC} {file_path}")
        print(f"  {e}")
        return False

def validate_directory(directory: str, pattern: str = "**/*.py") -> tuple:
    """Validate all Python files in directory"""
    path = Path(directory)
    if not path.exists():
        print(f"{Colors.YELLOW}Directory not found: {directory}{Colors.NC}")
        return 0, 0
    
    print(f"\n{Colors.BOLD}Validating: {directory}{Colors.NC}")
    print("-" * 60)
    
    files = list(path.glob(pattern))
    if not files:
        print(f"{Colors.YELLOW}No Python files found{Colors.NC}")
        return 0, 0
    
    passed = 0
    failed = 0
    
    for file in sorted(files):
        # Skip __pycache__ and venv
        if '__pycache__' in str(file) or 'venv' in str(file):
            continue
        
        # Check syntax
        if check_syntax(file):
            # Check imports
            if check_imports(file):
                print(f"{Colors.GREEN}✓{Colors.NC} {file.relative_to(path.parent)}")
                passed += 1
            else:
                failed += 1
        else:
            failed += 1
    
    return passed, failed

def main():
    print_header("ShieldX ML Service - Quick Validation")
    
    os.chdir('/home/vananh/shieldx/services/shieldx-ml')
    print(f"Working directory: {os.getcwd()}\n")
    
    total_passed = 0
    total_failed = 0
    
    # Validate main ML service code
    directories = [
        'ml-service',
        'tests',
        'ml-service/tests'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            passed, failed = validate_directory(directory)
            total_passed += passed
            total_failed += failed
    
    # Summary
    print_header("Validation Summary")
    print(f"Total Files:    {total_passed + total_failed}")
    print(f"{Colors.GREEN}Passed:         {total_passed}{Colors.NC}")
    print(f"{Colors.RED}Failed:         {total_failed}{Colors.NC}")
    
    if total_failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL FILES VALID!{Colors.NC}")
        # Guidance for running full tests
        print(f"\n{Colors.YELLOW}Note: Full test execution requires installing test dependencies.{Colors.NC}")
        print(f"{Colors.YELLOW}Run: pip3 install -r requirements-test.txt{Colors.NC}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ VALIDATION FAILED{Colors.NC}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
