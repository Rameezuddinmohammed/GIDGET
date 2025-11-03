"""Language-specific code element extractors."""

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict, Any
import re

from tree_sitter import Tree, Node

from .models import (
    ParsedFile, CodeElement, FunctionElement, ClassElement, 
    ImportElement, VariableElement, DependencyRelation, CodeElementType
)


class BaseExtractor(ABC):
    """Base class for language-specific code extractors."""
    
    @abstractmethod
    def extract_elements(self, tree: Tree, content: str, file_path: str, language: str) -> ParsedFile:
        """Extract code elements from parsed tree."""
        pass
    
    def _get_node_text(self, node: Node, content: str) -> str:
        """Get text content of a tree-sitter node."""
        return content[node.start_byte:node.end_byte]
    
    def _get_node_position(self, node: Node) -> tuple:
        """Get (start_line, end_line, start_column, end_column) of node."""
        return (
            node.start_point[0] + 1,  # Convert to 1-based line numbers
            node.end_point[0] + 1,
            node.start_point[1],
            node.end_point[1]
        )


class PythonExtractor(BaseExtractor):
    """Extracts code elements from Python source code."""
    
    def extract_elements(self, tree: Tree, content: str, file_path: str, language: str) -> ParsedFile:
        """Extract Python code elements."""
        elements = []
        dependencies = []
        imports = []
        
        root_node = tree.root_node
        
        # Extract imports
        imports.extend(self._extract_imports(root_node, content, file_path))
        
        # Extract classes and functions
        elements.extend(self._extract_classes(root_node, content, file_path))
        elements.extend(self._extract_functions(root_node, content, file_path))
        
        # Extract dependencies
        dependencies.extend(self._extract_dependencies(root_node, content, file_path))
        
        return ParsedFile(
            file_path=file_path,
            language=language,
            elements=elements,
            dependencies=dependencies,
            imports=imports
        )
    
    def _extract_imports(self, node: Node, content: str, file_path: str) -> List[ImportElement]:
        """Extract import statements."""
        imports = []
        
        for child in node.children:
            if child.type == 'import_statement':
                # import module
                module_node = child.child_by_field_name('name')
                if module_node:
                    start_line, end_line, start_col, end_col = self._get_node_position(child)
                    module_name = self._get_node_text(module_node, content)
                    
                    imports.append(ImportElement(
                        name=module_name,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_col,
                        end_column=end_col,
                        language='python',
                        module_name=module_name,
                        is_from_import=False
                    ))
            
            elif child.type == 'import_from_statement':
                # from module import names
                module_node = child.child_by_field_name('module_name')
                names_node = child.child_by_field_name('name')
                
                if module_node:
                    start_line, end_line, start_col, end_col = self._get_node_position(child)
                    module_name = self._get_node_text(module_node, content)
                    
                    imported_names = []
                    if names_node:
                        if names_node.type == 'dotted_name':
                            imported_names.append(self._get_node_text(names_node, content))
                        elif names_node.type == 'import_list':
                            for name_node in names_node.children:
                                if name_node.type == 'dotted_name':
                                    imported_names.append(self._get_node_text(name_node, content))
                    
                    imports.append(ImportElement(
                        name=f"from {module_name}",
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_col,
                        end_column=end_col,
                        language='python',
                        module_name=module_name,
                        imported_names=imported_names,
                        is_from_import=True
                    ))
        
        return imports
    
    def _extract_classes(self, node: Node, content: str, file_path: str) -> List[ClassElement]:
        """Extract class definitions."""
        classes = []
        
        def visit_node(n: Node):
            if n.type == 'class_definition':
                name_node = n.child_by_field_name('name')
                if name_node:
                    start_line, end_line, start_col, end_col = self._get_node_position(n)
                    class_name = self._get_node_text(name_node, content)
                    
                    # Extract base classes
                    base_classes = []
                    superclasses_node = n.child_by_field_name('superclasses')
                    if superclasses_node:
                        for child in superclasses_node.children:
                            if child.type == 'identifier':
                                base_classes.append(self._get_node_text(child, content))
                    
                    # Extract methods
                    methods = []
                    body_node = n.child_by_field_name('body')
                    if body_node:
                        for child in body_node.children:
                            if child.type == 'function_definition':
                                method_name_node = child.child_by_field_name('name')
                                if method_name_node:
                                    methods.append(self._get_node_text(method_name_node, content))
                    
                    # Extract docstring
                    docstring = self._extract_docstring(n, content)
                    
                    classes.append(ClassElement(
                        name=class_name,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_col,
                        end_column=end_col,
                        language='python',
                        base_classes=base_classes,
                        methods=methods,
                        docstring=docstring
                    ))
            
            # Recursively visit children
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return classes
    
    def _extract_functions(self, node: Node, content: str, file_path: str) -> List[FunctionElement]:
        """Extract function definitions."""
        functions = []
        
        def visit_node(n: Node):
            if n.type == 'function_definition':
                name_node = n.child_by_field_name('name')
                if name_node:
                    start_line, end_line, start_col, end_col = self._get_node_position(n)
                    func_name = self._get_node_text(name_node, content)
                    
                    # Extract parameters
                    parameters = []
                    params_node = n.child_by_field_name('parameters')
                    if params_node:
                        for child in params_node.children:
                            if child.type == 'identifier':
                                parameters.append(self._get_node_text(child, content))
                            elif child.type == 'typed_parameter':
                                param_name_node = child.child_by_field_name('pattern')
                                if param_name_node:
                                    parameters.append(self._get_node_text(param_name_node, content))
                    
                    # Check if async
                    is_async = False
                    for child in n.children:
                        if child.type == 'async' or self._get_node_text(child, content) == 'async':
                            is_async = True
                            break
                    
                    # Extract docstring
                    docstring = self._extract_docstring(n, content)
                    
                    # Extract function calls
                    calls = self._extract_function_calls(n, content)
                    
                    functions.append(FunctionElement(
                        name=func_name,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_col,
                        end_column=end_col,
                        language='python',
                        parameters=parameters,
                        is_async=is_async,
                        docstring=docstring,
                        calls=calls
                    ))
            
            # Recursively visit children
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return functions
    
    def _extract_dependencies(self, node: Node, content: str, file_path: str) -> List[DependencyRelation]:
        """Extract dependency relationships."""
        dependencies = []
        
        def visit_node(n: Node):
            if n.type == 'call':
                # Function call dependency
                func_node = n.child_by_field_name('function')
                if func_node:
                    func_name = self._get_node_text(func_node, content)
                    start_line, _, _, _ = self._get_node_position(n)
                    
                    dependencies.append(DependencyRelation(
                        source_element="<current_context>",
                        target_element=func_name,
                        relation_type="calls",
                        file_path=file_path,
                        line_number=start_line
                    ))
            
            # Recursively visit children
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return dependencies
    
    def _extract_docstring(self, node: Node, content: str) -> Optional[str]:
        """Extract docstring from function or class."""
        body_node = node.child_by_field_name('body')
        if body_node and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == 'expression_statement':
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == 'string':
                    docstring = self._get_node_text(expr, content)
                    # Remove quotes and clean up
                    return docstring.strip('"\'').strip()
        return None
    
    def _extract_function_calls(self, node: Node, content: str) -> Set[str]:
        """Extract function calls within a function."""
        calls = set()
        
        def visit_node(n: Node):
            if n.type == 'call':
                func_node = n.child_by_field_name('function')
                if func_node:
                    func_name = self._get_node_text(func_node, content)
                    # Extract just the function name (not full dotted path)
                    if '.' in func_name:
                        func_name = func_name.split('.')[-1]
                    calls.add(func_name)
            
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return calls


class JavaScriptExtractor(BaseExtractor):
    """Extracts code elements from JavaScript source code."""
    
    def extract_elements(self, tree: Tree, content: str, file_path: str, language: str) -> ParsedFile:
        """Extract JavaScript code elements."""
        elements = []
        dependencies = []
        imports = []
        
        root_node = tree.root_node
        
        # Extract imports
        imports.extend(self._extract_imports(root_node, content, file_path))
        
        # Extract functions and classes
        elements.extend(self._extract_functions(root_node, content, file_path))
        elements.extend(self._extract_classes(root_node, content, file_path))
        
        return ParsedFile(
            file_path=file_path,
            language=language,
            elements=elements,
            dependencies=dependencies,
            imports=imports
        )
    
    def _extract_imports(self, node: Node, content: str, file_path: str) -> List[ImportElement]:
        """Extract import statements."""
        imports = []
        
        def visit_node(n: Node):
            if n.type == 'import_statement':
                start_line, end_line, start_col, end_col = self._get_node_position(n)
                
                # Extract module name
                source_node = n.child_by_field_name('source')
                module_name = ""
                if source_node:
                    module_name = self._get_node_text(source_node, content).strip('"\'')
                
                # Extract imported names
                imported_names = []
                import_clause = n.child_by_field_name('import')
                if import_clause:
                    # Handle different import patterns
                    for child in import_clause.children:
                        if child.type == 'identifier':
                            imported_names.append(self._get_node_text(child, content))
                        elif child.type == 'import_specifier':
                            name_node = child.child_by_field_name('name')
                            if name_node:
                                imported_names.append(self._get_node_text(name_node, content))
                
                imports.append(ImportElement(
                    name=f"import from {module_name}",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    start_column=start_col,
                    end_column=end_col,
                    language='javascript',
                    module_name=module_name,
                    imported_names=imported_names
                ))
            
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return imports
    
    def _extract_functions(self, node: Node, content: str, file_path: str) -> List[FunctionElement]:
        """Extract function definitions."""
        functions = []
        
        def visit_node(n: Node):
            if n.type in ['function_declaration', 'function_expression', 'arrow_function']:
                name = self._get_function_name(n, content)
                if name:
                    start_line, end_line, start_col, end_col = self._get_node_position(n)
                    
                    # Extract parameters
                    parameters = []
                    params_node = n.child_by_field_name('parameters')
                    if params_node:
                        for child in params_node.children:
                            if child.type == 'identifier':
                                parameters.append(self._get_node_text(child, content))
                    
                    # Check if async
                    is_async = any(
                        self._get_node_text(child, content) == 'async' 
                        for child in n.children
                    )
                    
                    functions.append(FunctionElement(
                        name=name,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_col,
                        end_column=end_col,
                        language='javascript',
                        parameters=parameters,
                        is_async=is_async
                    ))
            
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return functions
    
    def _extract_classes(self, node: Node, content: str, file_path: str) -> List[ClassElement]:
        """Extract class definitions."""
        classes = []
        
        def visit_node(n: Node):
            if n.type == 'class_declaration':
                name_node = n.child_by_field_name('name')
                if name_node:
                    start_line, end_line, start_col, end_col = self._get_node_position(n)
                    class_name = self._get_node_text(name_node, content)
                    
                    # Extract superclass
                    base_classes = []
                    superclass_node = n.child_by_field_name('superclass')
                    if superclass_node:
                        base_classes.append(self._get_node_text(superclass_node, content))
                    
                    # Extract methods
                    methods = []
                    body_node = n.child_by_field_name('body')
                    if body_node:
                        for child in body_node.children:
                            if child.type == 'method_definition':
                                method_name_node = child.child_by_field_name('name')
                                if method_name_node:
                                    methods.append(self._get_node_text(method_name_node, content))
                    
                    classes.append(ClassElement(
                        name=class_name,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_col,
                        end_column=end_col,
                        language='javascript',
                        base_classes=base_classes,
                        methods=methods
                    ))
            
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return classes
    
    def _get_function_name(self, node: Node, content: str) -> Optional[str]:
        """Extract function name from various function node types."""
        if node.type == 'function_declaration':
            name_node = node.child_by_field_name('name')
            if name_node:
                return self._get_node_text(name_node, content)
        elif node.type in ['function_expression', 'arrow_function']:
            # For expressions, try to find assignment or property name
            parent = node.parent
            if parent and parent.type == 'variable_declarator':
                name_node = parent.child_by_field_name('name')
                if name_node:
                    return self._get_node_text(name_node, content)
            elif parent and parent.type == 'assignment_expression':
                left_node = parent.child_by_field_name('left')
                if left_node:
                    return self._get_node_text(left_node, content)
        
        return None


class TypeScriptExtractor(JavaScriptExtractor):
    """Extracts code elements from TypeScript source code."""
    
    def extract_elements(self, tree: Tree, content: str, file_path: str, language: str) -> ParsedFile:
        """Extract TypeScript code elements."""
        # Use JavaScript extractor as base, then add TypeScript-specific features
        parsed_file = super().extract_elements(tree, content, file_path, language)
        parsed_file.language = 'typescript'
        
        # Add TypeScript-specific elements like interfaces
        root_node = tree.root_node
        interfaces = self._extract_interfaces(root_node, content, file_path)
        parsed_file.elements.extend(interfaces)
        
        return parsed_file
    
    def _extract_interfaces(self, node: Node, content: str, file_path: str) -> List[ClassElement]:
        """Extract TypeScript interface definitions."""
        interfaces = []
        
        def visit_node(n: Node):
            if n.type == 'interface_declaration':
                name_node = n.child_by_field_name('name')
                if name_node:
                    start_line, end_line, start_col, end_col = self._get_node_position(n)
                    interface_name = self._get_node_text(name_node, content)
                    
                    # Extract extends clause
                    base_classes = []
                    extends_node = n.child_by_field_name('extends')
                    if extends_node:
                        for child in extends_node.children:
                            if child.type == 'identifier':
                                base_classes.append(self._get_node_text(child, content))
                    
                    interfaces.append(ClassElement(
                        name=interface_name,
                        element_type=CodeElementType.INTERFACE,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        start_column=start_col,
                        end_column=end_col,
                        language='typescript',
                        base_classes=base_classes
                    ))
            
            for child in n.children:
                visit_node(child)
        
        visit_node(node)
        return interfaces