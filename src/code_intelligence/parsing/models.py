"""Data models for code parsing and analysis."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field
import hashlib


class CodeElementType(str, Enum):
    """Types of code elements."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    CONSTANT = "constant"


class CodeElement(BaseModel):
    """Base class for all code elements."""
    name: str
    element_type: CodeElementType
    file_path: str
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    language: str
    signature_hash: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __post_init__(self):
        """Generate signature hash after initialization."""
        if not self.signature_hash:
            self.signature_hash = self.generate_signature_hash()
    
    def generate_signature_hash(self) -> str:
        """Generate a hash representing the element's signature."""
        signature_parts = [
            self.name,
            self.element_type.value,
            str(self.start_line),
            str(self.end_line),
        ]
        
        # Add type-specific signature components
        if hasattr(self, 'parameters'):
            signature_parts.extend([str(p) for p in getattr(self, 'parameters', [])])
        if hasattr(self, 'return_type'):
            signature_parts.append(str(getattr(self, 'return_type', '')))
        
        signature_str = '|'.join(signature_parts)
        return hashlib.md5(signature_str.encode()).hexdigest()


class FunctionElement(CodeElement):
    """Represents a function or method."""
    element_type: CodeElementType = CodeElementType.FUNCTION
    parameters: List[str] = Field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    decorators: List[str] = Field(default_factory=list)
    docstring: Optional[str] = None
    complexity: int = 1  # Cyclomatic complexity
    calls: Set[str] = Field(default_factory=set)  # Functions called by this function


class ClassElement(CodeElement):
    """Represents a class or interface."""
    element_type: CodeElementType = CodeElementType.CLASS
    base_classes: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)  # Method names
    attributes: List[str] = Field(default_factory=list)
    decorators: List[str] = Field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False


class ImportElement(CodeElement):
    """Represents an import statement."""
    element_type: CodeElementType = CodeElementType.IMPORT
    module_name: str
    imported_names: List[str] = Field(default_factory=list)
    alias: Optional[str] = None
    is_from_import: bool = False


class VariableElement(CodeElement):
    """Represents a variable or constant."""
    element_type: CodeElementType = CodeElementType.VARIABLE
    variable_type: Optional[str] = None
    is_constant: bool = False
    initial_value: Optional[str] = None


class DependencyRelation(BaseModel):
    """Represents a dependency relationship between code elements."""
    source_element: str  # Element name or identifier
    target_element: str
    relation_type: str  # "calls", "imports", "inherits", "uses", etc.
    file_path: str
    line_number: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedFile(BaseModel):
    """Represents a parsed source code file."""
    file_path: str
    language: str
    elements: List[CodeElement] = Field(default_factory=list)
    dependencies: List[DependencyRelation] = Field(default_factory=list)
    imports: List[ImportElement] = Field(default_factory=list)
    parse_errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_elements_by_type(self, element_type: CodeElementType) -> List[CodeElement]:
        """Get all elements of a specific type."""
        return [elem for elem in self.elements if elem.element_type == element_type]
    
    def get_functions(self) -> List[FunctionElement]:
        """Get all function elements."""
        return [elem for elem in self.elements if isinstance(elem, FunctionElement)]
    
    def get_classes(self) -> List[ClassElement]:
        """Get all class elements."""
        return [elem for elem in self.elements if isinstance(elem, ClassElement)]