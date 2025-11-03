"""Graph population utilities for Neo4j."""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from ..database.neo4j_client import Neo4jClient
from ..git.models import CommitInfo, FileChange
from ..parsing.models import ParsedFile, CodeElement, FunctionElement, ClassElement, ImportElement
from .models import GraphBatch, GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


class GraphPopulator:
    """Populates Neo4j graph database with code analysis data."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """Initialize with Neo4j client."""
        self.neo4j = neo4j_client
        self._batch_size = 1000
    
    def create_repository_node(self, repository_id: str, repository_info: Dict[str, Any]) -> str:
        """Create or update repository node."""
        query = """
        MERGE (r:Repository {id: $repo_id})
        SET r += $properties
        SET r.last_updated = datetime()
        RETURN r.id as id
        """
        
        result = self.neo4j.execute_query_sync(query, {
            'repo_id': repository_id,
            'properties': repository_info
        })
        
        return result[0]['id'] if result else repository_id
    
    def create_commit_node(self, commit_info: CommitInfo, repository_id: str) -> str:
        """Create commit node with temporal relationships."""
        query = """
        MATCH (r:Repository {id: $repo_id})
        MERGE (c:Commit {sha: $sha})
        SET c += $properties
        MERGE (r)-[:HAS_COMMIT]->(c)
        
        // Link to parent commits
        WITH c
        UNWIND $parents as parent_sha
        MATCH (parent:Commit {sha: parent_sha})
        MERGE (parent)-[:PARENT_OF]->(c)
        
        RETURN c.sha as sha
        """
        
        properties = {
            'sha': commit_info.sha,
            'message': commit_info.message,
            'author_name': commit_info.author_name,
            'author_email': commit_info.author_email,
            'committer_name': commit_info.committer_name,
            'committer_email': commit_info.committer_email,
            'authored_date': commit_info.authored_date.isoformat(),
            'committed_date': commit_info.committed_date.isoformat(),
            'insertions': commit_info.stats.get('insertions', 0),
            'deletions': commit_info.stats.get('deletions', 0),
            'files_changed': commit_info.stats.get('files', 0)
        }
        
        result = self.neo4j.execute_query_sync(query, {
            'repo_id': repository_id,
            'sha': commit_info.sha,
            'properties': properties,
            'parents': commit_info.parents
        })
        
        return result[0]['sha'] if result else commit_info.sha
    
    def ingest_parsed_files(self, parsed_files: List[ParsedFile], commit_sha: str, repository_id: str) -> Dict[str, int]:
        """Ingest parsed files into graph database."""
        stats = {'nodes': 0, 'relationships': 0}
        
        # Process files in batches
        for i in range(0, len(parsed_files), self._batch_size):
            batch_files = parsed_files[i:i + self._batch_size]
            batch = self._create_graph_batch(batch_files, commit_sha, repository_id)
            
            batch_stats = self._ingest_batch(batch)
            stats['nodes'] += batch_stats['nodes']
            stats['relationships'] += batch_stats['relationships']
        
        return stats
    
    def _create_graph_batch(self, parsed_files: List[ParsedFile], commit_sha: str, repository_id: str) -> GraphBatch:
        """Create a graph batch from parsed files."""
        batch = GraphBatch(commit_sha=commit_sha, repository_id=repository_id)
        
        for parsed_file in parsed_files:
            # Create file node
            file_key = f"{repository_id}:{parsed_file.file_path}"
            batch.add_node(
                labels=['File'],
                properties={
                    'path': parsed_file.file_path,
                    'language': parsed_file.language,
                    'repository_id': repository_id,
                    'element_count': len(parsed_file.elements),
                    'has_errors': len(parsed_file.parse_errors) > 0,
                    'parse_errors': parsed_file.parse_errors
                },
                unique_key=file_key
            )
            
            # Link file to commit
            batch.add_relationship(
                source_key=f"commit:{commit_sha}",
                target_key=file_key,
                rel_type='CHANGED_IN',
                properties={'commit_sha': commit_sha}
            )
            
            # Process code elements
            for element in parsed_file.elements:
                element_key = self._create_element_node(element, batch, repository_id)
                
                # Link element to file
                batch.add_relationship(
                    source_key=file_key,
                    target_key=element_key,
                    rel_type='CONTAINS'
                )
                
                # Link element to commit for temporal tracking
                batch.add_relationship(
                    source_key=element_key,
                    target_key=f"commit:{commit_sha}",
                    rel_type='CHANGED_IN',
                    properties={'commit_sha': commit_sha}
                )
            
            # Process imports
            for import_elem in parsed_file.imports:
                import_key = self._create_import_node(import_elem, batch, repository_id)
                
                # Link import to file
                batch.add_relationship(
                    source_key=file_key,
                    target_key=import_key,
                    rel_type='IMPORTS'
                )
            
            # Process dependencies
            for dep in parsed_file.dependencies:
                self._create_dependency_relationship(dep, batch, file_key)
        
        return batch
    
    def _create_element_node(self, element: CodeElement, batch: GraphBatch, repository_id: str) -> str:
        """Create node for code element."""
        element_key = f"{repository_id}:{element.file_path}:{element.name}:{element.start_line}"
        
        # Base properties
        properties = {
            'name': element.name,
            'file_path': element.file_path,
            'start_line': element.start_line,
            'end_line': element.end_line,
            'start_column': element.start_column,
            'end_column': element.end_column,
            'language': element.language,
            'signature_hash': element.signature_hash,
            'repository_id': repository_id
        }
        
        # Add type-specific properties and labels
        labels = ['CodeElement']
        
        if isinstance(element, FunctionElement):
            labels.append('Function')
            properties.update({
                'parameters': element.parameters,
                'return_type': element.return_type,
                'is_async': element.is_async,
                'is_generator': element.is_generator,
                'decorators': element.decorators,
                'docstring': element.docstring,
                'complexity': element.complexity,
                'calls_count': len(element.calls)
            })
            
            # Create call relationships
            for called_func in element.calls:
                called_key = f"function_call:{called_func}"
                batch.add_relationship(
                    source_key=element_key,
                    target_key=called_key,
                    rel_type='CALLS',
                    properties={'function_name': called_func}
                )
        
        elif isinstance(element, ClassElement):
            labels.append('Class')
            properties.update({
                'base_classes': element.base_classes,
                'methods': element.methods,
                'attributes': element.attributes,
                'decorators': element.decorators,
                'docstring': element.docstring,
                'is_abstract': element.is_abstract,
                'method_count': len(element.methods)
            })
            
            # Create inheritance relationships
            for base_class in element.base_classes:
                base_key = f"class:{base_class}"
                batch.add_relationship(
                    source_key=element_key,
                    target_key=base_key,
                    rel_type='INHERITS_FROM',
                    properties={'base_class': base_class}
                )
        
        batch.add_node(labels=labels, properties=properties, unique_key=element_key)
        return element_key
    
    def _create_import_node(self, import_elem: ImportElement, batch: GraphBatch, repository_id: str) -> str:
        """Create node for import statement."""
        import_key = f"{repository_id}:{import_elem.file_path}:import:{import_elem.start_line}"
        
        properties = {
            'module_name': import_elem.module_name,
            'imported_names': import_elem.imported_names,
            'alias': import_elem.alias,
            'is_from_import': import_elem.is_from_import,
            'file_path': import_elem.file_path,
            'start_line': import_elem.start_line,
            'language': import_elem.language,
            'repository_id': repository_id
        }
        
        batch.add_node(labels=['Import'], properties=properties, unique_key=import_key)
        return import_key
    
    def _create_dependency_relationship(self, dep, batch: GraphBatch, file_key: str):
        """Create dependency relationship."""
        # For now, create a simple dependency relationship
        # This could be enhanced to link to actual code elements
        batch.add_relationship(
            source_key=file_key,
            target_key=f"dependency:{dep.target_element}",
            rel_type='DEPENDS_ON',
            properties={
                'relation_type': dep.relation_type,
                'line_number': dep.line_number,
                'target_element': dep.target_element
            }
        )
    
    def _ingest_batch(self, batch: GraphBatch) -> Dict[str, int]:
        """Ingest a batch of nodes and relationships."""
        stats = {'nodes': 0, 'relationships': 0}
        
        try:
            # Create nodes first
            if batch.nodes:
                node_stats = self._create_nodes_batch(batch.nodes)
                stats['nodes'] = node_stats
            
            # Then create relationships
            if batch.relationships:
                rel_stats = self._create_relationships_batch(batch.relationships)
                stats['relationships'] = rel_stats
                
        except Exception as e:
            logger.error(f"Failed to ingest batch: {e}")
            raise
        
        return stats
    
    def _create_nodes_batch(self, nodes: List[GraphNode]) -> int:
        """Create nodes in batch."""
        if not nodes:
            return 0
        
        # Group nodes by labels for efficient creation
        nodes_by_labels = {}
        for node in nodes:
            labels_key = ':'.join(sorted(node.labels))
            if labels_key not in nodes_by_labels:
                nodes_by_labels[labels_key] = []
            nodes_by_labels[labels_key].append(node)
        
        total_created = 0
        
        for labels_key, label_nodes in nodes_by_labels.items():
            labels = labels_key.split(':')
            
            # Create MERGE query for this label combination
            label_str = ':'.join(labels)
            query = f"""
            UNWIND $nodes as node
            MERGE (n:{label_str} {{unique_key: node.unique_key}})
            SET n += node.properties
            RETURN count(n) as created
            """
            
            node_data = [
                {
                    'unique_key': node.unique_key,
                    'properties': {**node.properties, 'unique_key': node.unique_key}
                }
                for node in label_nodes
            ]
            
            result = self.neo4j.execute_query_sync(query, {'nodes': node_data})
            created = result[0]['created'] if result else 0
            total_created += created
        
        return total_created
    
    def _create_relationships_batch(self, relationships: List[GraphRelationship]) -> int:
        """Create relationships in batch."""
        if not relationships:
            return 0
        
        # Group relationships by type for efficient creation
        rels_by_type = {}
        for rel in relationships:
            if rel.relationship_type not in rels_by_type:
                rels_by_type[rel.relationship_type] = []
            rels_by_type[rel.relationship_type].append(rel)
        
        total_created = 0
        
        for rel_type, type_rels in rels_by_type.items():
            query = f"""
            UNWIND $relationships as rel
            MATCH (source {{unique_key: rel.source_key}})
            MATCH (target {{unique_key: rel.target_key}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r += rel.properties
            RETURN count(r) as created
            """
            
            rel_data = [
                {
                    'source_key': rel.source_key,
                    'target_key': rel.target_key,
                    'properties': rel.properties
                }
                for rel in type_rels
            ]
            
            result = self.neo4j.execute_query_sync(query, {'relationships': rel_data})
            created = result[0]['created'] if result else 0
            total_created += created
        
        return total_created
    
    def cleanup_old_data(self, repository_id: str, keep_commits: int = 100):
        """Clean up old commit data to manage database size."""
        query = """
        MATCH (r:Repository {id: $repo_id})-[:HAS_COMMIT]->(c:Commit)
        WITH c ORDER BY c.committed_date DESC
        SKIP $keep_commits
        
        // Delete old commits and their relationships
        DETACH DELETE c
        
        RETURN count(c) as deleted
        """
        
        result = self.neo4j.execute_query_sync(query, {
            'repo_id': repository_id,
            'keep_commits': keep_commits
        })
        
        deleted = result[0]['deleted'] if result else 0
        logger.info(f"Cleaned up {deleted} old commits for repository {repository_id}")
        return deleted