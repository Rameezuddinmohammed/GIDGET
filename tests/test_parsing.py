"""Tests for code parsing functionality."""

import pytest
from pathlib import Path

from src.code_intelligence.parsing.parser import CodeParser, MultiLanguageParser, ParsingError
from src.code_intelligence.parsing.models import ParsedFile, FunctionElement, ClassElement, ImportElement
from src.code_intelligence.parsing.extractors import PythonExtractor, JavaScriptExtractor


class TestCodeParser:
    """Test CodeParser class."""
    
    def test_init(self):
        """Test parser initialization."""
        parser = CodeParser()
        
        supported = parser.supported_languages()
        expected_languages = ['python', 'javascript', 'typescript']
        
        for lang in expected_languages:
            assert lang in supported
    
    def test_parse_python_content(self, sample_python_code):
        """Test parsing Python code content."""
        parser = CodeParser()
        
        parsed = parser.parse_content(sample_python_code, 'python', 'test.py')
        
        assert isinstance(parsed, ParsedFile)
        assert parsed.language == 'python'
        assert parsed.file_path == 'test.py'
        assert len(parsed.elements) > 0
        
        # Check for expected elements
        functions = parsed.get_functions()
        classes = parsed.get_classes()
        
        assert len(classes) == 1
        assert classes[0].name == 'Calculator'
        
        assert len(functions) >= 3  # __init__, add, multiply, main
        function_names = [f.name for f in functions]
        assert 'add' in function_names
        assert 'multiply' in function_names
        assert 'main' in function_names
    
    def test_parse_javascript_content(self, sample_javascript_code):
        """Test parsing JavaScript code content."""
        parser = CodeParser()
        
        parsed = parser.parse_content(sample_javascript_code, 'javascript', 'test.js')
        
        assert isinstance(parsed, ParsedFile)
        assert parsed.language == 'javascript'
        assert len(parsed.elements) > 0
        
        # Check for expected elements
        classes = parsed.get_classes()
        functions = parsed.get_functions()
        
        assert len(classes) == 1
        assert classes[0].name == 'TaskManager'
        
        # Should find methods as functions
        function_names = [f.name for f in functions]
        assert any('addTask' in name for name in function_names)
    
    def test_detect_language(self):
        """Test language detection from file extensions."""
        parser = CodeParser()
        
        assert parser._detect_language(Path('test.py')) == 'python'
        assert parser._detect_language(Path('test.js')) == 'javascript'
        assert parser._detect_language(Path('test.jsx')) == 'javascript'
        assert parser._detect_language(Path('test.ts')) == 'typescript'
        assert parser._detect_language(Path('test.tsx')) == 'typescript'
        assert parser._detect_language(Path('test.txt')) is None
    
    def test_parse_unsupported_language(self):
        """Test parsing unsupported language."""
        parser = CodeParser()
        
        with pytest.raises(ParsingError):
            parser.parse_content("content", 'unsupported', 'test.unknown')


class TestMultiLanguageParser:
    """Test MultiLanguageParser class."""
    
    def test_init(self):
        """Test multi-language parser initialization."""
        parser = MultiLanguageParser()
        
        assert parser.parser is not None
        assert isinstance(parser.parser, CodeParser)
    
    def test_parse_directory(self, temp_dir, sample_python_code, sample_javascript_code):
        """Test parsing directory with multiple files."""
        # Create test files
        (temp_dir / "test.py").write_text(sample_python_code)
        (temp_dir / "test.js").write_text(sample_javascript_code)
        (temp_dir / "README.md").write_text("# Test")  # Should be ignored
        
        parser = MultiLanguageParser()
        parsed_files = parser.parse_directory(str(temp_dir))
        
        assert len(parsed_files) == 2  # Only .py and .js files
        
        languages = [f.language for f in parsed_files]
        assert 'python' in languages
        assert 'javascript' in languages
    
    def test_parse_files(self, temp_dir, sample_python_code):
        """Test parsing specific files."""
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)
        
        parser = MultiLanguageParser()
        parsed_files = parser.parse_files([str(test_file)])
        
        assert len(parsed_files) == 1
        assert parsed_files[0].language == 'python'
        assert len(parsed_files[0].elements) > 0


class TestPythonExtractor:
    """Test PythonExtractor class."""
    
    def test_extract_imports(self, sample_python_code):
        """Test extracting import statements."""
        parser = CodeParser()
        tree = parser._get_parser('python').parse(sample_python_code.encode('utf-8'))
        
        extractor = PythonExtractor()
        parsed = extractor.extract_elements(tree, sample_python_code, 'test.py', 'python')
        
        assert len(parsed.imports) >= 2  # import os, from typing import ...
        
        import_modules = [imp.module_name for imp in parsed.imports]
        assert 'os' in import_modules
        assert 'typing' in import_modules
    
    def test_extract_classes(self, sample_python_code):
        """Test extracting class definitions."""
        parser = CodeParser()
        tree = parser._get_parser('python').parse(sample_python_code.encode('utf-8'))
        
        extractor = PythonExtractor()
        parsed = extractor.extract_elements(tree, sample_python_code, 'test.py', 'python')
        
        classes = parsed.get_classes()
        assert len(classes) == 1
        
        calc_class = classes[0]
        assert calc_class.name == 'Calculator'
        assert calc_class.docstring == 'A simple calculator class.'
        assert len(calc_class.methods) >= 2  # __init__, add, multiply
    
    def test_extract_functions(self, sample_python_code):
        """Test extracting function definitions."""
        parser = CodeParser()
        tree = parser._get_parser('python').parse(sample_python_code.encode('utf-8'))
        
        extractor = PythonExtractor()
        parsed = extractor.extract_elements(tree, sample_python_code, 'test.py', 'python')
        
        functions = parsed.get_functions()
        assert len(functions) >= 3
        
        function_names = [f.name for f in functions]
        assert 'add' in function_names
        assert 'multiply' in function_names
        assert 'main' in function_names
        
        # Check function details
        add_func = next(f for f in functions if f.name == 'add')
        assert len(add_func.parameters) >= 2  # self, a, b
        assert add_func.docstring == 'Add two numbers.'


class TestJavaScriptExtractor:
    """Test JavaScriptExtractor class."""
    
    def test_extract_imports(self, sample_javascript_code):
        """Test extracting import statements."""
        parser = CodeParser()
        tree = parser._get_parser('javascript').parse(sample_javascript_code.encode('utf-8'))
        
        extractor = JavaScriptExtractor()
        parsed = extractor.extract_elements(tree, sample_javascript_code, 'test.js', 'javascript')
        
        assert len(parsed.imports) >= 2  # EventEmitter, fs
        
        import_modules = [imp.module_name for imp in parsed.imports]
        assert 'events' in import_modules
        assert 'fs' in import_modules
    
    def test_extract_classes(self, sample_javascript_code):
        """Test extracting class definitions."""
        parser = CodeParser()
        tree = parser._get_parser('javascript').parse(sample_javascript_code.encode('utf-8'))
        
        extractor = JavaScriptExtractor()
        parsed = extractor.extract_elements(tree, sample_javascript_code, 'test.js', 'javascript')
        
        classes = parsed.get_classes()
        assert len(classes) == 1
        
        task_class = classes[0]
        assert task_class.name == 'TaskManager'
        assert 'EventEmitter' in task_class.base_classes
    
    def test_extract_functions(self, sample_javascript_code):
        """Test extracting function definitions."""
        parser = CodeParser()
        tree = parser._get_parser('javascript').parse(sample_javascript_code.encode('utf-8'))
        
        extractor = JavaScriptExtractor()
        parsed = extractor.extract_elements(tree, sample_javascript_code, 'test.js', 'javascript')
        
        functions = parsed.get_functions()
        assert len(functions) >= 2
        
        # Check for async function
        async_functions = [f for f in functions if f.is_async]
        assert len(async_functions) >= 1