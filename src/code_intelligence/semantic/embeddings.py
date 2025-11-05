"""Code embedding generation pipeline using specialized models."""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..logging import get_logger
from ..exceptions import EmbeddingError
from .models import (
    CodeElement, CodeEmbedding, EmbeddingModel, EmbeddingBatch, 
    EmbeddingQuality, CodeElementType
)

logger = get_logger(__name__)


class CodeEmbeddingGenerator:
    """Generates embeddings for code elements using specialized models."""
    
    def __init__(
        self, 
        model: EmbeddingModel = EmbeddingModel.CODEBERT,
        device: str = "auto",
        max_batch_size: int = 32,
        max_sequence_length: int = 512
    ) -> None:
        """Initialize the embedding generator.
        
        Args:
            model: The embedding model to use
            device: Device to run on ('cpu', 'cuda', or 'auto')
            max_batch_size: Maximum batch size for processing
            max_sequence_length: Maximum sequence length for tokenization
        """
        self.model_name = model.value
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using mock embeddings")
            self._use_mock = True
            self.embedding_dimension = 768  # Standard dimension
        else:
            self._use_mock = False
            self._setup_model(device)
    
    def _setup_model(self, device: str) -> None:
        """Setup the embedding model and tokenizer."""
        try:
            logger.info("Loading embedding model", model=self.model_name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Determine device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                dummy_input = self.tokenizer("test", return_tensors="pt", padding=True)
                dummy_output = self.model(**dummy_input.to(self.device))
                self.embedding_dimension = dummy_output.last_hidden_state.shape[-1]
            
            logger.info(
                "Model loaded successfully", 
                model=self.model_name,
                device=self.device,
                dimension=self.embedding_dimension
            )
            
        except Exception as e:
            logger.error("Failed to load embedding model", model=self.model_name, error=str(e))
            raise EmbeddingError(f"Model loading failed: {e}")
    
    async def generate_embedding(self, element: CodeElement) -> CodeEmbedding:
        """Generate embedding for a single code element."""
        if self._use_mock:
            return self._generate_mock_embedding(element)
        
        try:
            # Prepare code text for embedding
            code_text = self._prepare_code_text(element)
            
            # Generate embedding
            embedding = await self._encode_text(code_text)
            
            return CodeEmbedding(
                element=element,
                embedding=embedding,
                model_name=self.model_name,
                embedding_dimension=self.embedding_dimension,
                confidence_score=1.0
            )
            
        except Exception as e:
            logger.error(
                "Failed to generate embedding", 
                element=element.signature, 
                error=str(e)
            )
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    async def generate_batch_embeddings(self, batch: EmbeddingBatch) -> List[CodeEmbedding]:
        """Generate embeddings for a batch of code elements."""
        logger.info("Generating batch embeddings", batch_size=batch.size, batch_id=batch.batch_id)
        
        if self._use_mock:
            return [self._generate_mock_embedding(element) for element in batch.elements]
        
        try:
            # Split into smaller batches if needed
            sub_batches = batch.split(self.max_batch_size)
            all_embeddings = []
            
            for sub_batch in sub_batches:
                embeddings = await self._process_sub_batch(sub_batch)
                all_embeddings.extend(embeddings)
            
            logger.info(
                "Batch embeddings generated", 
                total_embeddings=len(all_embeddings),
                batch_id=batch.batch_id
            )
            
            return all_embeddings
            
        except Exception as e:
            logger.error("Failed to generate batch embeddings", batch_id=batch.batch_id, error=str(e))
            raise EmbeddingError(f"Batch embedding generation failed: {e}")
    
    async def _process_sub_batch(self, batch: EmbeddingBatch) -> List[CodeEmbedding]:
        """Process a sub-batch of elements."""
        # Prepare all texts
        texts = [self._prepare_code_text(element) for element in batch.elements]
        
        # Generate embeddings in parallel
        embeddings = await self._encode_batch(texts)
        
        # Create CodeEmbedding objects
        result = []
        for element, embedding in zip(batch.elements, embeddings):
            code_embedding = CodeEmbedding(
                element=element,
                embedding=embedding,
                model_name=self.model_name,
                embedding_dimension=self.embedding_dimension,
                confidence_score=1.0
            )
            result.append(code_embedding)
        
        return result
    
    def _prepare_code_text(self, element: CodeElement) -> str:
        """Prepare code text for embedding generation."""
        # Create a structured representation of the code element
        parts = []
        
        # Add element type and name
        parts.append(f"{element.element_type.value}: {element.name}")
        
        # Add language context
        parts.append(f"Language: {element.language}")
        
        # Add the actual code
        parts.append(element.code_snippet.strip())
        
        # Add relevant metadata
        if element.metadata:
            if "docstring" in element.metadata:
                parts.append(f"Documentation: {element.metadata['docstring']}")
            if "parameters" in element.metadata:
                parts.append(f"Parameters: {element.metadata['parameters']}")
            if "return_type" in element.metadata:
                parts.append(f"Returns: {element.metadata['return_type']}")
        
        return "\n".join(parts)
    
    async def _encode_text(self, text: str) -> List[float]:
        """Encode a single text into embedding."""
        loop = asyncio.get_event_loop()
        
        def _encode():
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embedding
                outputs = self.model(**inputs)
                
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings.cpu().numpy()[0].tolist()
        
        # Run in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _encode)
    
    async def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into embeddings."""
        loop = asyncio.get_event_loop()
        
        def _encode_batch():
            with torch.no_grad():
                # Tokenize all texts
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                outputs = self.model(**inputs)
                
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings.cpu().numpy().tolist()
        
        # Run in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _encode_batch)
    
    def _generate_mock_embedding(self, element: CodeElement) -> CodeEmbedding:
        """Generate a mock embedding for testing purposes."""
        # Create a deterministic but varied embedding based on element content
        np.random.seed(hash(element.signature) % (2**32))
        embedding = np.random.normal(0, 1, self.embedding_dimension).tolist()
        
        # Normalize
        magnitude = np.linalg.norm(embedding)
        if magnitude > 0:
            embedding = (np.array(embedding) / magnitude).tolist()
        
        return CodeEmbedding(
            element=element,
            embedding=embedding,
            model_name=f"mock-{self.model_name}",
            embedding_dimension=self.embedding_dimension,
            confidence_score=0.8  # Lower confidence for mock
        )
    
    async def validate_embedding_quality(
        self, 
        embeddings: List[CodeEmbedding]
    ) -> EmbeddingQuality:
        """Validate the quality of generated embeddings."""
        if not embeddings:
            raise ValueError("No embeddings provided for quality validation")
        
        # Extract embedding vectors
        vectors = np.array([emb.embedding for emb in embeddings])
        
        # Calculate statistics
        magnitudes = np.linalg.norm(vectors, axis=1)
        avg_magnitude = float(np.mean(magnitudes))
        std_magnitude = float(np.std(magnitudes))
        
        # Calculate sparsity (ratio of near-zero values)
        near_zero_threshold = 1e-6
        sparsity_ratio = float(np.mean(np.abs(vectors) < near_zero_threshold))
        
        # Calculate consistency score (similarity between similar code elements)
        consistency_score = await self._calculate_consistency_score(embeddings)
        
        quality = EmbeddingQuality(
            model_name=self.model_name,
            dimension=self.embedding_dimension,
            avg_magnitude=avg_magnitude,
            std_magnitude=std_magnitude,
            sparsity_ratio=sparsity_ratio,
            consistency_score=consistency_score
        )
        
        logger.info(
            "Embedding quality validated",
            quality_score=quality.quality_score,
            avg_magnitude=avg_magnitude,
            sparsity_ratio=sparsity_ratio,
            consistency_score=consistency_score
        )
        
        return quality
    
    async def _calculate_consistency_score(self, embeddings: List[CodeEmbedding]) -> float:
        """Calculate consistency score based on similar code elements."""
        if len(embeddings) < 2:
            return 1.0
        
        # Group embeddings by element type and language
        groups = {}
        for emb in embeddings:
            key = (emb.element.element_type, emb.element.language)
            if key not in groups:
                groups[key] = []
            groups[key].append(emb)
        
        # Calculate average similarity within groups
        total_similarity = 0.0
        total_pairs = 0
        
        for group_embeddings in groups.values():
            if len(group_embeddings) < 2:
                continue
            
            vectors = np.array([emb.embedding for emb in group_embeddings])
            
            # Calculate pairwise cosine similarities
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    similarity = np.dot(vectors[i], vectors[j])
                    total_similarity += similarity
                    total_pairs += 1
        
        if total_pairs == 0:
            return 1.0
        
        return float(total_similarity / total_pairs)
    
    async def update_embeddings_for_changes(
        self,
        repository_id: str,
        changed_elements: List[CodeElement],
        commit_sha: str
    ) -> List[CodeEmbedding]:
        """Update embeddings for changed code elements."""
        logger.info(
            "Updating embeddings for changes",
            repository_id=repository_id,
            changed_count=len(changed_elements),
            commit_sha=commit_sha
        )
        
        # Create batch for changed elements
        batch = EmbeddingBatch(
            elements=changed_elements,
            batch_id=f"update_{commit_sha[:8]}",
            repository_id=repository_id,
            commit_sha=commit_sha
        )
        
        # Generate new embeddings
        new_embeddings = await self.generate_batch_embeddings(batch)
        
        logger.info(
            "Embeddings updated for changes",
            repository_id=repository_id,
            updated_count=len(new_embeddings)
        )
        
        return new_embeddings