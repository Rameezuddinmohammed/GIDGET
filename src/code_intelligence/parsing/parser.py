"""Core parsing functionality using tree-sitter."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Type
import logging

import tree_sitter
from tree_sitter import Language, Parser

from ..exceptions import CodeIntelligenceError
from .models import ParsedFile, CodeElement
from .extractors import BaseExtractor, PythonExtractor, JavaScriptExtractor, TypeScriptExtractor


logger = logging.getLogger(__name__)


class ParsingError(CodeIntelligenceError):
    """Code parsing related errors."""
    pass


class CodeParser:
    """Parses source code using tree-sitter."""
    
    def __init__(self):
        """Initialize parser with language support."""
        self._languages: Dict[str, Language] = {}
        self._parsers: Dict[str, Parser] = {}
        self._extractors: Dict[str, BaseExtractor] = {}
        
        self._setup_languages()
    
    def _setup_languages(self) -> None:
        """Set up tree-sitter languages and parsers."""
        try:
            # Try to load pre-built languages
            self._load_languages()
        except Exception as e:
            logger.warning(f"Failed to load pre-built languages: {e}")
            # Fall back to building languages
            self._build_languages()
    
    def _load_languages(self) -> None:
        """Load pre-built tree-sitter languages."""
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            import tree_sitter_typescript
            
            # Load languages - the Language constructor only takes the language object
            self._languages['python'] = Language(tree_sitter_python.language())
            self._languages['javascript'] = Language(tree_sitter_javascript.language())
            # TypeScript module has different function names
            self._languages['typescript'] = Language(tree_sitter_typescript.language_typescript())
            
        except ImportError as e:
            raise ParsingError(f"Tree-sitter language modules not available: {e}")
        except Exception as e:
            raise ParsingError(f"Failed to load tree-sitter languages: {e}")
    
    def _build_languages(self) -> None:
        """Build tree-sitter languages from source (fallback)."""
        # This would require building from source, which is complex
        # For now, we'll raise an error and require pre-built languages
        raise ParsingError(
            "Pre-built tree-sitter languages not available. "
            "Please install tree-sitter-python, tree-sitter-javascript, and tree-sitter-typescript."
        )
    
    def _get_parser(self, language: str) -> Parser:
        """Get or create parser for language."""
        if language not in self._parsers:
            if language not in self._languages:
                raise ParsingError(f"Unsupported language: {language}")
            
            parser = Parser()
            parser.language = self._languages[language]
            self._parsers[language] = parser
        
        return self._parsers[language]
    
    def _get_extractor(self, language: str) -> BaseExtractor:
        """Get or create extractor for language."""
        if language not in self._extractors:
            extractor_classes = {
                'python': PythonExtractor,
                'javascript': JavaScriptExtractor,
                'typescript': TypeScriptExtractor,
            }
            
            if language not in extractor_classes:
                raise ParsingError(f"No extractor available for language: {language}")
            
            self._extractors[language] = extractor_classes[language]()
        
        return self._extractors[language]
    
    def parse_file(self, file_path: str, content: Optional[str] = None) -> ParsedFile:
        """Parse a source code file."""
        file_path = Path(file_path)
        
        # Determine language from file extension
        language = self._detect_language(file_path)
        if not language:
            raise ParsingError(f"Unsupported file type: {file_path.suffix}")
        
        # Read file content if not provided
        if content is None:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                raise ParsingError(f"Failed to read file {file_path}: {e}")
        
        # Parse with tree-sitter
        parser = self._get_parser(language)
        tree = parser.parse(content.encode('utf-8'))
        
        # Extract code elements
        extractor = self._get_extractor(language)
        parsed_file = extractor.extract_elements(
            tree=tree,
            content=content,
            file_path=str(file_path),
            language=language
        )
        
        return parsed_file
    
    def parse_content(self, content: str, language: str, file_path: str = "<string>") -> ParsedFile:
        """Parse source code content directly."""
        if language not in self._languages:
            raise ParsingError(f"Unsupported language: {language}")
        
        parser = self._get_parser(language)
        tree = parser.parse(content.encode('utf-8'))
        
        extractor = self._get_extractor(language)
        return extractor.extract_elements(
            tree=tree,
            content=content,
            file_path=file_path,
            language=language
        )
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
        }
        
        return extension_map.get(file_path.suffix.lower())
    
    def supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self._languages.keys())


class MultiLanguageParser:
    """Parses multiple files across different languages."""
    
    def __init__(self):
        """Initialize multi-language parser."""
        self.parser = CodeParser()
    
    def parse_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[ParsedFile]:
        """Parse all supported files in a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise ParsingError(f"Directory does not exist: {directory}")
        
        parsed_files = []
        
        # Default patterns
        if include_patterns is None:
            include_patterns = ['*.py', '*.js', '*.jsx', '*.ts', '*.tsx']
        if exclude_patterns is None:
            exclude_patterns = ['node_modules/**', '.git/**', '__pycache__/**', '*.pyc']
        
        # Find files to parse
        files_to_parse = []
        if recursive:
            for pattern in include_patterns:
                files_to_parse.extend(directory.rglob(pattern))
        else:
            for pattern in include_patterns:
                files_to_parse.extend(directory.glob(pattern))
        
        # Filter out excluded files
        filtered_files = []
        for file_path in files_to_parse:
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if file_path.match(exclude_pattern):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_files.append(file_path)
        
        # Parse each file
        for file_path in filtered_files:
            try:
                parsed_file = self.parser.parse_file(str(file_path))
                parsed_files.append(parsed_file)
                logger.debug(f"Parsed {file_path}: {len(parsed_file.elements)} elements")
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                # Create empty parsed file with error
                parsed_files.append(ParsedFile(
                    file_path=str(file_path),
                    language=self.parser._detect_language(file_path) or "unknown",
                    parse_errors=[str(e)]
                ))
        
        return parsed_files
    
    def parse_files(self, file_paths: List[str]) -> List[ParsedFile]:
        """Parse a list of specific files."""
        parsed_files = []
        
        for file_path in file_paths:
            try:
                parsed_file = self.parser.parse_file(file_path)
                parsed_files.append(parsed_file)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                parsed_files.append(ParsedFile(
                    file_path=file_path,
                    language="unknown",
                    parse_errors=[str(e)]
                ))
        
        return parsed_files