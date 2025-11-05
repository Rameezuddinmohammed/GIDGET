"""Data models for semantic search system."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib


class EmbeddingModel(Enum):
    """Supported embedding models for code."""
    CODEBERT = "microsoft/codebert-base"
    GRAPHCODEBERT = "microsoft/graphcodebert-base"
    UNIXCODER = "microsoft/unixcoder-base"
    CODET5 = "Salesforce/codet5-base"


class CodeElementType(Enum):
    """Types of code elements that can be embedded."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    VARIABLE = "variable"
    COMMENT = "comment"


@dataclass
class CodeElement:
    """Represents a code element for embedding generation."""
    
    element_type: CodeElementType
    name: str
    code_snippet: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Generate content hash for the code element."""
        content = f"{self.element_type.value}:{self.name}:{self.code_snippet}"
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def signature(self) -> str:
        """Get a unique signature for this code element."""
        return f"{self.file_path}:{self.element_type.value}:{self.name}:{self.start_line}"


@dataclass
class CodeEmbedding:
    """Represents a code embedding with metadata."""
    
    element: CodeElement
    embedding: List[float]
    model_name: str
    embedding_dimension: int
    confidence_score: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate embedding dimensions."""
        if len(self.embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(self.embedding)}"
            )


@dataclass
class SearchResult:
    """Represents a semantic search result."""
    
    element: CodeElement
    similarity_score: float
    embedding_distance: float
    rank: int
    explanation: Optional[str] = None
    
    @property
    def confidence(self) -> float:
        """Get confidence score based on similarity."""
        return min(self.similarity_score, 1.0)


@dataclass
class HybridSearchResult:
    """Represents a hybrid search result combining semantic and structural search."""
    
    semantic_result: Optional[SearchResult]
    structural_matches: List[Dict[str, Any]]
    combined_score: float
    rank: int
    explanation: str
    
    @property
    def best_element(self) -> Optional[CodeElement]:
        """Get the best matching code element."""
        if self.semantic_result:
            return self.semantic_result.element
        elif self.structural_matches:
            # Return first structural match as CodeElement if available
            match = self.structural_matches[0]
            if "element" in match:
                return match["element"]
        return None


@dataclass
class EmbeddingBatch:
    """Represents a batch of code elements for processing."""
    
    elements: List[CodeElement]
    batch_id: str
    repository_id: str
    commit_sha: str
    
    def __post_init__(self) -> None:
        """Generate batch hash."""
        content = f"{self.repository_id}:{self.commit_sha}:{len(self.elements)}"
        self.batch_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.elements)
    
    def split(self, max_batch_size: int) -> List["EmbeddingBatch"]:
        """Split batch into smaller batches."""
        if self.size <= max_batch_size:
            return [self]
        
        batches = []
        for i in range(0, self.size, max_batch_size):
            batch_elements = self.elements[i:i + max_batch_size]
            batch = EmbeddingBatch(
                elements=batch_elements,
                batch_id=f"{self.batch_id}_{i // max_batch_size}",
                repository_id=self.repository_id,
                commit_sha=self.commit_sha
            )
            batches.append(batch)
        
        return batches


@dataclass
class SearchQuery:
    """Represents a semantic search query."""
    
    query_text: str
    repository_id: str
    element_types: Optional[List[CodeElementType]] = None
    file_patterns: Optional[List[str]] = None
    language_filter: Optional[str] = None
    max_results: int = 10
    similarity_threshold: float = 0.7
    include_explanation: bool = True
    
    def __post_init__(self) -> None:
        """Generate query hash."""
        content = f"{self.query_text}:{self.repository_id}:{self.max_results}"
        self.query_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class EmbeddingQuality:
    """Represents embedding quality metrics."""
    
    model_name: str
    dimension: int
    avg_magnitude: float
    std_magnitude: float
    sparsity_ratio: float  # Ratio of near-zero values
    consistency_score: float  # Consistency across similar code
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        # Higher consistency and lower sparsity indicate better quality
        return (self.consistency_score * 0.7) + ((1 - self.sparsity_ratio) * 0.3)