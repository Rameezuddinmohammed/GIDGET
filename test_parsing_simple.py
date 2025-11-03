#!/usr/bin/env python3
"""Simple parsing test."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_intelligence.parsing.parser import CodeParser

def test_parsing():
    """Test parsing functionality."""
    parser = CodeParser()
    
    # Simple Python code
    code = '''def hello():
    return "world"

class TestClass:
    def method(self):
        return hello()
'''
    
    try:
        result = parser.parse_content(code, 'python', 'test.py')
        print(f"✅ Parsed {len(result.elements)} elements")
        
        for elem in result.elements:
            print(f"  {elem.element_type}: {elem.name} (lines {elem.start_line}-{elem.end_line})")
        
        return True
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parsing()
    sys.exit(0 if success else 1)