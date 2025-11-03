"""Analyst Agent for deep code analysis using Neo4j and semantic search."""

import ast
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

from ..database.neo4j_client import Neo4jClient
from ..database.supabase_client import SupabaseClient
from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, CodeElement, Citation


logger = get_logger(__name__)


class AnalystAgent(BaseAgent):
    """Agent responsible for deep code analysis using graph queries and semantic search."""
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize the Analyst Agent."""
        if config is None:
            config = AgentConfig(
                name="analyst",
                description="Performs deep structural and semantic code analysis using Neo4j and vector search"
            )
        super().__init__(config, **kwargs)
        
        # Initialize database clients
        self.neo4j_client = None
        self.supabase_client = None
        
        # Register analysis templates
        self._register_templates()
        
    def _register_templates(self) -> None:
        """Register prompt templates for code analysis."""
        structural_analysis_template = PromptTemplate(
            """Analyze the following code structure and relationships for deep insights.

Target Elements: {target_elements}
Graph Data: {graph_data}
Semantic Matches: {semantic_matches}

Provide comprehensive analysis including:
1. Structural relationships and dependencies
2. Architectural patterns and design insights
3. Code quality and complexity assessment
4. Impact analysis and change propagation
5. Recommendations for improvement

Respond in JSON format:
{{
    "structural_analysis": {{
        "dependencies": [
            {{
                "from": "element_name",
                "to": "element_name", 
                "relationship": "calls|imports|inherits|uses",
                "strength": 0.0-1.0,
                "description": "relationship description"
            }}
        ],
        "complexity_metrics": {{
            "cyclomatic_complexity": 0,
            "coupling_score": 0.0-1.0,
            "cohesion_score": 0.0-1.0
        }},
        "architectural_patterns": [
            {{
                "pattern": "pattern_name",
                "confidence": 0.0-1.0,
                "elements": ["element1", "element2"],
                "description": "pattern description"
            }}
        ]
    }},
    "semantic_analysis": {{
        "conceptual_clusters": [
            {{
                "concept": "concept_name",
                "elements": ["element1", "element2"],
                "similarity_score": 0.0-1.0
            }}
        ],
        "code_smells": [
            {{
                "smell_type": "smell_name",
                "elements": ["element1"],
                "severity": "low|medium|high",
                "description": "smell description"
            }}
        ]
    }},
    "impact_analysis": {{
        "change_propagation": [
            {{
                "source": "element_name",
                "affected": ["element1", "element2"],
                "impact_level": "low|medium|high",
                "reasoning": "why these elements are affected"
            }}
        ]
    }},
    "recommendations": [
        {{
            "type": "refactoring|optimization|design_improvement",
            "priority": "low|medium|high",
            "description": "recommendation description",
            "affected_elements": ["element1", "element2"]
        }}
    ]
}}""",
            variables=["target_elements", "graph_data", "semantic_matches"]
        )
        
        dependency_analysis_template = PromptTemplate(
            """Analyze dependency relationships and trace impact across the codebase.

Focus Elements: {focus_elements}
Dependency Graph: {dependency_graph}
Analysis Scope: {analysis_scope}

Trace dependencies and provide:
1. Direct and transitive dependencies
2. Circular dependency detection
3. Dependency strength and criticality
4. Impact propagation paths
5. Refactoring opportunities

Format as detailed dependency analysis with specific citations.""",
            variables=["focus_elements", "dependency_graph", "analysis_scope"]
        )
        
        self.structural_analysis_template = structural_analysis_template
        self.dependency_analysis_template = dependency_analysis_template
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute analyst logic."""
        self._log_execution_start(state)
        
        if not self._validate_state(state):
            state.add_error("Invalid state for analyst", self.config.name)
            return state
            
        try:
            # Initialize database connections
            await self._initialize_database_clients()
            
            # Get target elements for analysis
            target_elements = await self._resolve_target_elements(state)
            
            # Check if we have working code from historian
            working_code_info = self._extract_working_code_from_state(state)
            repository_path = state.repository.get("path", "")
            
            if working_code_info:
                # Analyze integration requirements for working code
                integration_analysis = await self._analyze_integration_requirements(
                    working_code_info, repository_path, state
                )
                
                # Perform enhanced structural analysis with real code
                structural_data = await self._perform_enhanced_structural_analysis(
                    state, target_elements, working_code_info
                )
            else:
                # Fallback to graph-based analysis
                structural_data = await self._perform_structural_analysis(state, target_elements)
                integration_analysis = {"available": False}
            
            # Perform semantic search and analysis
            semantic_data = await self._perform_semantic_analysis(state, target_elements)
            
            # Analyze dependencies and impact
            dependency_data = await self._analyze_dependencies(state, target_elements)
            
            # Generate comprehensive analysis insights
            insights = await self._generate_analysis_insights(
                state, structural_data, semantic_data, dependency_data
            )
            
            # Add integration analysis if available
            if integration_analysis["available"]:
                integration_insight = {
                    "type": "integration_analysis",
                    "content": f"Integration analysis for working code: {len(integration_analysis['dependencies'])} dependencies identified, {len(integration_analysis['integration_steps'])} steps required",
                    "confidence": integration_analysis["confidence"],
                    "citations": [],
                    "metadata": integration_analysis
                }
                insights.append(integration_insight)
            
            # Create analyst findings
            for insight in insights:
                finding = self._create_finding(
                    finding_type=insight["type"],
                    content=insight["content"],
                    confidence=insight["confidence"],
                    citations=insight.get("citations", []),
                    metadata=insight.get("metadata", {})
                )
                state.add_finding(self.config.name, finding)
                
            # Update state with analysis data
            state.analysis["structural_analysis"] = structural_data
            state.analysis["semantic_analysis"] = semantic_data
            state.analysis["dependency_analysis"] = dependency_data
            state.analysis["integration_analysis"] = integration_analysis
            
            self._log_execution_end(state, True)
            return state
            
        except Exception as e:
            error_context = {
                "session_id": state.session_id,
                "target_elements_count": len(state.analysis.get("target_elements", [])),
                "neo4j_available": self.neo4j_client is not None,
                "supabase_available": self.supabase_client is not None,
                "error_type": type(e).__name__
            }
            self.logger.error(f"Analyst execution failed: {str(e)}", extra=error_context)
            state.add_error(f"Analyst failed: {str(e)}", self.config.name)
            self._log_execution_end(state, False)
            return state
            
    async def _initialize_database_clients(self) -> None:
        """Initialize database clients if not already done."""
        if not self.neo4j_client:
            try:
                self.neo4j_client = Neo4jClient()
                # Test connection
                await self.neo4j_client.execute_query("RETURN 1 as test")
                self.logger.info("Neo4j connection established successfully")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Neo4j: {str(e)}")
                self.neo4j_client = None
            
        if not self.supabase_client:
            try:
                self.supabase_client = SupabaseClient()
                # Test connection would go here if SupabaseClient had a test method
                self.logger.info("Supabase client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Supabase client: {str(e)}")
                self.supabase_client = None
            
    async def _resolve_target_elements(self, state: AgentState) -> List[CodeElement]:
        """Resolve target elements from analysis state."""
        target_elements_data = state.analysis.get("target_elements", [])
        target_elements = []
        
        for elem_data in target_elements_data:
            # If file_path is missing, try to resolve it
            if not elem_data.get("file_path") and elem_data.get("name"):
                resolved_elements = await self._find_elements_by_name(elem_data["name"])
                target_elements.extend(resolved_elements)
            else:
                target_elements.append(CodeElement(**elem_data))
                
        return target_elements
        
    async def _find_elements_by_name(self, name: str) -> List[CodeElement]:
        """Find code elements by name using Neo4j."""
        if not self.neo4j_client:
            return []
            
        try:
            # Query Neo4j for elements with matching names
            query = """
            MATCH (n)
            WHERE n.name CONTAINS $name OR n.name = $name
            RETURN n.name as name, labels(n)[0] as type, 
                   n.file_path as file_path, n.start_line as start_line,
                   n.end_line as end_line, n.signature_hash as signature_hash
            LIMIT 10
            """
            
            results = await self.neo4j_client.execute_query(query, {"name": name})
            
            elements = []
            for record in results:
                elements.append(CodeElement(
                    name=record["name"],
                    type=record["type"].lower() if record["type"] else "unknown",
                    file_path=record["file_path"] or "",
                    start_line=record["start_line"],
                    end_line=record["end_line"],
                    signature_hash=record["signature_hash"]
                ))
                
            return elements
            
        except Exception as e:
            self.logger.warning(f"Failed to find elements by name '{name}': {str(e)}")
            return []
            
    async def _perform_structural_analysis(
        self, 
        state: AgentState, 
        target_elements: List[CodeElement]
    ) -> Dict[str, Any]:
        """Perform structural analysis using Neo4j graph queries."""
        if not self.neo4j_client or not target_elements:
            return {"dependencies": [], "metrics": {}, "patterns": []}
            
        try:
            structural_data = {
                "dependencies": [],
                "metrics": {},
                "patterns": [],
                "graph_statistics": {}
            }
            
            # Analyze dependencies for target elements (batch processing)
            all_dependencies = await self._analyze_batch_dependencies(target_elements)
            structural_data["dependencies"].extend(all_dependencies)
                
            # Calculate complexity metrics
            structural_data["metrics"] = await self._calculate_complexity_metrics(target_elements)
            
            # Identify architectural patterns
            structural_data["patterns"] = await self._identify_architectural_patterns(target_elements)
            
            # Get graph statistics
            structural_data["graph_statistics"] = await self._get_graph_statistics()
            
            return structural_data
            
        except Exception as e:
            self.logger.error(f"Structural analysis failed: {str(e)}")
            return {"dependencies": [], "metrics": {}, "patterns": []}
            
    async def _analyze_element_dependencies(self, element: CodeElement) -> List[Dict[str, Any]]:
        """Analyze dependencies for a specific element."""
        if not self.neo4j_client:
            return []
            
        try:
            # Query for outgoing dependencies
            outgoing_query = """
            MATCH (source {name: $name, file_path: $file_path})-[r]->(target)
            RETURN source.name as from_name, target.name as to_name,
                   type(r) as relationship, target.file_path as to_file,
                   target.start_line as to_line
            LIMIT 20
            """
            
            # Query for incoming dependencies
            incoming_query = """
            MATCH (source)-[r]->(target {name: $name, file_path: $file_path})
            RETURN source.name as from_name, target.name as to_name,
                   type(r) as relationship, source.file_path as from_file,
                   source.start_line as from_line
            LIMIT 20
            """
            
            params = {"name": element.name, "file_path": element.file_path}
            
            outgoing_results = await self.neo4j_client.execute_query(outgoing_query, params)
            incoming_results = await self.neo4j_client.execute_query(incoming_query, params)
            
            dependencies = []
            
            # Process outgoing dependencies
            for record in outgoing_results:
                dependencies.append({
                    "from": record["from_name"],
                    "to": record["to_name"],
                    "relationship": record["relationship"].lower(),
                    "direction": "outgoing",
                    "target_file": record["to_file"],
                    "target_line": record["to_line"]
                })
                
            # Process incoming dependencies
            for record in incoming_results:
                dependencies.append({
                    "from": record["from_name"],
                    "to": record["to_name"],
                    "relationship": record["relationship"].lower(),
                    "direction": "incoming",
                    "source_file": record["from_file"],
                    "source_line": record["from_line"]
                })
                
            return dependencies
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze dependencies for {element.name}: {str(e)}")
            return []
            
    async def _calculate_complexity_metrics(self, target_elements: List[CodeElement]) -> Dict[str, Any]:
        """Calculate complexity metrics for target elements."""
        if not self.neo4j_client:
            return {}
            
        try:
            metrics = {
                "total_elements": len(target_elements),
                "coupling_scores": {},
                "fan_in_out": {},
                "depth_metrics": {}
            }
            
            for element in target_elements:
                # Calculate fan-in and fan-out
                fan_metrics = await self._calculate_fan_metrics(element)
                metrics["fan_in_out"][element.name] = fan_metrics
                
                # Calculate coupling score
                coupling_score = await self._calculate_coupling_score(element)
                metrics["coupling_scores"][element.name] = coupling_score
                
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate complexity metrics: {str(e)}")
            return {}
            
    async def _calculate_fan_metrics(self, element: CodeElement) -> Dict[str, int]:
        """Calculate fan-in and fan-out metrics for an element."""
        if not self.neo4j_client:
            return {"fan_in": 0, "fan_out": 0}
            
        try:
            # Fan-out: number of elements this element depends on
            fan_out_query = """
            MATCH (source {name: $name, file_path: $file_path})-[r]->(target)
            RETURN count(DISTINCT target) as fan_out
            """
            
            # Fan-in: number of elements that depend on this element
            fan_in_query = """
            MATCH (source)-[r]->(target {name: $name, file_path: $file_path})
            RETURN count(DISTINCT source) as fan_in
            """
            
            params = {"name": element.name, "file_path": element.file_path}
            
            fan_out_result = await self.neo4j_client.execute_query(fan_out_query, params)
            fan_in_result = await self.neo4j_client.execute_query(fan_in_query, params)
            
            return {
                "fan_in": fan_in_result[0]["fan_in"] if fan_in_result else 0,
                "fan_out": fan_out_result[0]["fan_out"] if fan_out_result else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate fan metrics for {element.name}: {str(e)}")
            return {"fan_in": 0, "fan_out": 0}
            
    async def _calculate_coupling_score(self, element: CodeElement) -> float:
        """Calculate coupling score for an element."""
        fan_metrics = await self._calculate_fan_metrics(element)
        
        # Simple coupling score based on fan-in and fan-out
        fan_in = fan_metrics["fan_in"]
        fan_out = fan_metrics["fan_out"]
        
        if fan_in + fan_out == 0:
            return 0.0
            
        # Normalize to 0-1 scale (higher values indicate higher coupling)
        coupling_score = min((fan_in + fan_out) / 20.0, 1.0)
        return coupling_score
        
    async def _identify_architectural_patterns(self, target_elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Identify architectural patterns in the code."""
        patterns = []
        
        # Simple pattern detection based on naming and structure
        element_names = [elem.name for elem in target_elements]
        
        # Factory pattern detection
        factory_elements = [name for name in element_names if "factory" in name.lower()]
        if factory_elements:
            patterns.append({
                "pattern": "factory_pattern",
                "confidence": 0.7,
                "elements": factory_elements,
                "description": f"Detected factory pattern in {len(factory_elements)} elements"
            })
            
        # Service pattern detection
        service_elements = [name for name in element_names if "service" in name.lower()]
        if service_elements:
            patterns.append({
                "pattern": "service_pattern",
                "confidence": 0.8,
                "elements": service_elements,
                "description": f"Detected service pattern in {len(service_elements)} elements"
            })
            
        # Repository pattern detection
        repo_elements = [name for name in element_names if "repository" in name.lower() or "repo" in name.lower()]
        if repo_elements:
            patterns.append({
                "pattern": "repository_pattern",
                "confidence": 0.8,
                "elements": repo_elements,
                "description": f"Detected repository pattern in {len(repo_elements)} elements"
            })
            
        return patterns
        
    async def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        if not self.neo4j_client:
            return {}
            
        try:
            stats_query = """
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
            ORDER BY count DESC
            """
            
            results = await self.neo4j_client.execute_query(stats_query)
            
            node_counts = {}
            total_nodes = 0
            
            for record in results:
                node_type = record["node_type"] or "unknown"
                count = record["count"]
                node_counts[node_type] = count
                total_nodes += count
                
            return {
                "total_nodes": total_nodes,
                "node_type_distribution": node_counts
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get graph statistics: {str(e)}")
            return {}
            
    async def _perform_semantic_analysis(
        self, 
        state: AgentState, 
        target_elements: List[CodeElement]
    ) -> Dict[str, Any]:
        """Perform semantic analysis using vector search."""
        if not self.supabase_client or not target_elements:
            return {"semantic_matches": [], "clusters": [], "similarities": {}}
            
        try:
            semantic_data = {
                "semantic_matches": [],
                "conceptual_clusters": [],
                "similarity_scores": {},
                "code_smells": []
            }
            
            # Perform semantic search for each target element
            for element in target_elements:
                matches = await self._find_semantic_matches(element)
                semantic_data["semantic_matches"].extend(matches)
                
            # Identify conceptual clusters
            semantic_data["conceptual_clusters"] = await self._identify_conceptual_clusters(target_elements)
            
            # Detect code smells
            semantic_data["code_smells"] = await self._detect_code_smells(target_elements)
            
            return semantic_data
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {str(e)}")
            return {"semantic_matches": [], "clusters": [], "similarities": {}}
            
    async def _find_semantic_matches(self, element: CodeElement) -> List[Dict[str, Any]]:
        """Find semantically similar code elements."""
        # This would use pgvector for semantic search
        # For now, return placeholder data
        return [
            {
                "element": element.name,
                "similar_elements": [],
                "similarity_scores": {},
                "semantic_concepts": []
            }
        ]
        
    async def _identify_conceptual_clusters(self, target_elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Identify conceptual clusters of related elements."""
        # Group elements by conceptual similarity
        clusters = []
        
        # Simple clustering based on naming patterns
        name_groups = {}
        for element in target_elements:
            # Extract base concept from name
            base_name = re.sub(r'(Service|Repository|Factory|Manager|Handler)$', '', element.name)
            base_name = re.sub(r'([A-Z])', r' \1', base_name).strip().lower()
            
            if base_name not in name_groups:
                name_groups[base_name] = []
            name_groups[base_name].append(element.name)
            
        for concept, elements in name_groups.items():
            if len(elements) > 1:
                clusters.append({
                    "concept": concept,
                    "elements": elements,
                    "similarity_score": 0.8  # Placeholder
                })
                
        return clusters
        
    async def _detect_code_smells(self, target_elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Detect potential code smells."""
        smells = []
        
        # Long parameter list detection (placeholder)
        for element in target_elements:
            if "(" in element.name and element.name.count(",") > 5:
                smells.append({
                    "smell_type": "long_parameter_list",
                    "elements": [element.name],
                    "severity": "medium",
                    "description": f"Function {element.name} may have too many parameters"
                })
                
        return smells
        
    async def _analyze_dependencies(
        self, 
        state: AgentState, 
        target_elements: List[CodeElement]
    ) -> Dict[str, Any]:
        """Analyze dependencies and impact propagation."""
        dependency_data = {
            "dependency_graph": {},
            "circular_dependencies": [],
            "impact_analysis": {},
            "critical_paths": []
        }
        
        # Build dependency graph
        for element in target_elements:
            element_deps = await self._analyze_element_dependencies(element)
            dependency_data["dependency_graph"][element.name] = element_deps
            
        # Detect circular dependencies
        dependency_data["circular_dependencies"] = await self._detect_circular_dependencies(target_elements)
        
        # Analyze impact propagation
        dependency_data["impact_analysis"] = await self._analyze_impact_propagation(target_elements)
        
        return dependency_data
        
    async def _detect_circular_dependencies(self, target_elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Detect circular dependencies."""
        if not self.neo4j_client:
            return []
            
        try:
            # Query for circular dependencies
            circular_query = """
            MATCH path = (a)-[*2..10]->(a)
            WHERE a.name IN $element_names
            RETURN [node in nodes(path) | node.name] as cycle
            LIMIT 10
            """
            
            element_names = [elem.name for elem in target_elements]
            results = await self.neo4j_client.execute_query(circular_query, {"element_names": element_names})
            
            circular_deps = []
            for record in results:
                cycle = record["cycle"]
                if len(cycle) > 2:  # Valid cycle
                    circular_deps.append({
                        "cycle": cycle,
                        "length": len(cycle),
                        "severity": "high" if len(cycle) <= 3 else "medium"
                    })
                    
            return circular_deps
            
        except Exception as e:
            self.logger.warning(f"Failed to detect circular dependencies: {str(e)}")
            return []
            
    async def _analyze_impact_propagation(self, target_elements: List[CodeElement]) -> Dict[str, Any]:
        """Analyze how changes would propagate through dependencies."""
        impact_analysis = {}
        
        for element in target_elements:
            # Calculate impact radius
            impact_radius = await self._calculate_impact_radius(element)
            impact_analysis[element.name] = {
                "direct_impact": impact_radius.get("direct", []),
                "transitive_impact": impact_radius.get("transitive", []),
                "impact_score": impact_radius.get("score", 0.0)
            }
            
        return impact_analysis
        
    async def _calculate_impact_radius(self, element: CodeElement) -> Dict[str, Any]:
        """Calculate the impact radius of changes to an element."""
        if not self.neo4j_client:
            return {"direct": [], "transitive": [], "score": 0.0}
            
        try:
            # Find all elements that would be affected by changes to this element
            impact_query = """
            MATCH (source {name: $name, file_path: $file_path})<-[*1..3]-(affected)
            RETURN DISTINCT affected.name as affected_name, 
                   affected.file_path as affected_file,
                   length(shortestPath((source)<-[*]-(affected))) as distance
            ORDER BY distance
            LIMIT 20
            """
            
            params = {"name": element.name, "file_path": element.file_path}
            results = await self.neo4j_client.execute_query(impact_query, params)
            
            direct_impact = []
            transitive_impact = []
            
            for record in results:
                affected_element = {
                    "name": record["affected_name"],
                    "file": record["affected_file"],
                    "distance": record["distance"]
                }
                
                if record["distance"] == 1:
                    direct_impact.append(affected_element)
                else:
                    transitive_impact.append(affected_element)
                    
            # Calculate impact score based on number of affected elements
            total_affected = len(direct_impact) + len(transitive_impact)
            impact_score = min(total_affected / 10.0, 1.0)  # Normalize to 0-1
            
            return {
                "direct": direct_impact,
                "transitive": transitive_impact,
                "score": impact_score
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate impact radius for {element.name}: {str(e)}")
            return {"direct": [], "transitive": [], "score": 0.0}
            
    async def _generate_analysis_insights(
        self,
        state: AgentState,
        structural_data: Dict[str, Any],
        semantic_data: Dict[str, Any],
        dependency_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive analysis insights."""
        insights = []
        
        # Structural insights
        dependencies = structural_data.get("dependencies", [])
        if dependencies:
            citations = []
            for dep in dependencies[:5]:  # Top 5 dependencies
                citations.append(self._create_citation(
                    file_path=dep.get("target_file", "unknown"),
                    description=f"{dep['from']} {dep['relationship']} {dep['to']}",
                    line_number=dep.get("target_line")
                ))
                
            insights.append({
                "type": "structural_dependencies",
                "content": f"Identified {len(dependencies)} dependency relationships. "
                          f"Key patterns include {len(structural_data.get('patterns', []))} architectural patterns.",
                "confidence": 0.9,
                "citations": citations,
                "metadata": {"dependency_count": len(dependencies)}
            })
            
        # Complexity insights
        metrics = structural_data.get("metrics", {})
        if metrics:
            insights.append({
                "type": "complexity_analysis",
                "content": f"Analyzed complexity metrics for {metrics.get('total_elements', 0)} elements. "
                          f"Average coupling scores and fan-in/out metrics calculated.",
                "confidence": 0.8,
                "citations": [],
                "metadata": metrics
            })
            
        # Semantic insights
        clusters = semantic_data.get("conceptual_clusters", [])
        if clusters:
            insights.append({
                "type": "semantic_clustering",
                "content": f"Identified {len(clusters)} conceptual clusters of related code elements.",
                "confidence": 0.7,
                "citations": [],
                "metadata": {"clusters": clusters}
            })
            
        # Dependency insights
        circular_deps = dependency_data.get("circular_dependencies", [])
        if circular_deps:
            insights.append({
                "type": "circular_dependencies",
                "content": f"Detected {len(circular_deps)} circular dependency cycles that may indicate design issues.",
                "confidence": 0.95,
                "citations": [],
                "metadata": {"cycles": circular_deps}
            })
            
        return insights
            
    async def _analyze_batch_dependencies(self, target_elements: List[CodeElement]) -> List[Dict[str, Any]]:
        """Analyze dependencies for multiple elements in batch."""
        if not self.neo4j_client or not target_elements:
            return []
            
        try:
            # Batch query for all elements
            element_names = [elem.name for elem in target_elements]
            element_paths = [elem.file_path for elem in target_elements]
            
            # Single query for all outgoing dependencies
            outgoing_query = """
            MATCH (source)-[r]->(target)
            WHERE source.name IN $names AND source.file_path IN $paths
            RETURN source.name as from_name, target.name as to_name,
                   type(r) as relationship, target.file_path as to_file,
                   target.start_line as to_line, source.file_path as from_file
            LIMIT 100
            """
            
            # Single query for all incoming dependencies
            incoming_query = """
            MATCH (source)-[r]->(target)
            WHERE target.name IN $names AND target.file_path IN $paths
            RETURN source.name as from_name, target.name as to_name,
                   type(r) as relationship, source.file_path as from_file,
                   source.start_line as from_line, target.file_path as to_file
            LIMIT 100
            """
            
            params = {"names": element_names, "paths": element_paths}
            
            outgoing_results = await self.neo4j_client.execute_query(outgoing_query, params)
            incoming_results = await self.neo4j_client.execute_query(incoming_query, params)
            
            dependencies = []
            
            # Process outgoing dependencies
            for record in outgoing_results:
                dependencies.append({
                    "from": record["from_name"],
                    "to": record["to_name"],
                    "relationship": record["relationship"].lower(),
                    "direction": "outgoing",
                    "target_file": record["to_file"],
                    "target_line": record.get("to_line"),
                    "source_file": record["from_file"]
                })
                
            # Process incoming dependencies
            for record in incoming_results:
                dependencies.append({
                    "from": record["from_name"],
                    "to": record["to_name"],
                    "relationship": record["relationship"].lower(),
                    "direction": "incoming",
                    "source_file": record["from_file"],
                    "source_line": record.get("from_line"),
                    "target_file": record["to_file"]
                })
                
            return dependencies
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze batch dependencies: {str(e)}")
            # Fallback to individual processing
            all_deps = []
            for element in target_elements:
                element_deps = await self._analyze_element_dependencies(element)
                all_deps.extend(element_deps)
            return all_deps
            
    def _extract_working_code_from_state(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Extract working code information from historian findings."""
        
        historian_findings = state.get_findings_by_agent("historian")
        
        for finding in historian_findings:
            if finding.finding_type == "working_code_extraction":
                return {
                    "code_content": finding.metadata.get("code_content", ""),
                    "file_path": finding.metadata.get("file_path", ""),
                    "commit_sha": finding.metadata.get("commit_sha", ""),
                    "confidence": finding.confidence
                }
                
        return None
        
    async def _analyze_integration_requirements(
        self, 
        working_code_info: Dict[str, Any], 
        repository_path: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """Analyze what's needed to integrate working code into current codebase."""
        
        result = {
            "available": True,
            "dependencies": [],
            "compatibility": [],
            "integration_steps": [],
            "confidence": 0.0
        }
        
        try:
            code_content = working_code_info["code_content"]
            file_path = working_code_info["file_path"]
            
            # Extract dependencies from working code
            dependencies = self._extract_dependencies_from_code(code_content, file_path)
            result["dependencies"] = dependencies
            
            # Check current codebase compatibility
            compatibility = await self._check_codebase_compatibility(dependencies, repository_path)
            result["compatibility"] = compatibility
            
            # Generate integration steps
            integration_steps = self._generate_integration_steps(dependencies, compatibility)
            result["integration_steps"] = integration_steps
            
            # Calculate confidence
            if dependencies:
                compatible_count = len([c for c in compatibility if c.get("compatible", False)])
                result["confidence"] = min(0.9, 0.6 + (compatible_count / len(dependencies)) * 0.3)
            else:
                result["confidence"] = 0.7  # Medium confidence when no dependencies
                
        except Exception as e:
            self.logger.error(f"Integration analysis failed: {str(e)}")
            result["confidence"] = 0.3
            
        return result
        
    def _extract_dependencies_from_code(self, code_content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract dependencies from actual code content."""
        
        dependencies = []
        
        try:
            # Determine language and extract dependencies accordingly
            if file_path.endswith('.py'):
                dependencies = self._extract_python_dependencies(code_content)
            elif file_path.endswith('.java'):
                dependencies = self._extract_java_dependencies(code_content)
            elif file_path.endswith(('.js', '.ts')):
                dependencies = self._extract_javascript_dependencies(code_content)
            else:
                # Generic extraction
                dependencies = self._extract_generic_dependencies(code_content)
                
        except Exception as e:
            self.logger.error(f"Failed to extract dependencies from {file_path}: {str(e)}")
            
        return dependencies
        
    def _extract_python_dependencies(self, code_content: str) -> List[Dict[str, Any]]:
        """Extract Python dependencies using AST parsing."""
        
        dependencies = []
        
        try:
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append({
                            "name": alias.name,
                            "type": "import",
                            "line": node.lineno,
                            "usage": "direct_import"
                        })
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append({
                            "name": node.module,
                            "type": "from_import",
                            "line": node.lineno,
                            "usage": "from_import",
                            "items": [alias.name for alias in node.names]
                        })
                        
        except SyntaxError as e:
            self.logger.warning(f"Python syntax error in code: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to parse Python code: {str(e)}")
            
        return dependencies
        
    def _extract_java_dependencies(self, code_content: str) -> List[Dict[str, Any]]:
        """Extract Java dependencies using regex patterns."""
        
        dependencies = []
        
        # Import statements
        import_pattern = r'import\s+(static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?);'
        imports = re.finditer(import_pattern, code_content)
        
        for match in imports:
            is_static = match.group(1) is not None
            import_name = match.group(2)
            dependencies.append({
                "name": import_name,
                "type": "static_import" if is_static else "import",
                "line": code_content[:match.start()].count('\n') + 1,
                "usage": "import"
            })
            
        return dependencies
        
    def _extract_javascript_dependencies(self, code_content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript dependencies using regex patterns."""
        
        dependencies = []
        
        # Import statements
        patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)',
            r'import\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, code_content)
            for match in matches:
                dependencies.append({
                    "name": match.group(1),
                    "type": "import",
                    "line": code_content[:match.start()].count('\n') + 1,
                    "usage": "import"
                })
                
        return dependencies
        
    def _extract_generic_dependencies(self, code_content: str) -> List[Dict[str, Any]]:
        """Extract dependencies using generic patterns."""
        
        dependencies = []
        
        # Look for common dependency patterns
        patterns = [
            r'#include\s*[<"]([^>"]+)[>"]',  # C/C++
            r'using\s+([a-zA-Z_][a-zA-Z0-9_.]*);',  # C#
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, code_content)
            for match in matches:
                dependencies.append({
                    "name": match.group(1),
                    "type": "dependency",
                    "line": code_content[:match.start()].count('\n') + 1,
                    "usage": "include"
                })
                
        return dependencies
        
    async def _check_codebase_compatibility(self, dependencies: List[Dict[str, Any]], repository_path: str) -> List[Dict[str, Any]]:
        """Check if dependencies are compatible with current codebase."""
        
        compatibility = []
        
        for dep in dependencies:
            dep_name = dep["name"]
            
            # Check if dependency exists in current codebase
            exists_in_codebase = await self._dependency_exists_in_codebase(dep_name, repository_path)
            
            compatibility.append({
                "dependency": dep_name,
                "compatible": exists_in_codebase,
                "type": dep["type"],
                "status": "available" if exists_in_codebase else "missing"
            })
            
        return compatibility
        
    async def _dependency_exists_in_codebase(self, dep_name: str, repository_path: str) -> bool:
        """Check if a dependency exists in the current codebase."""
        
        try:
            # Check common dependency files
            dependency_files = [
                "requirements.txt", "setup.py", "pyproject.toml",  # Python
                "pom.xml", "build.gradle",  # Java
                "package.json",  # JavaScript/Node.js
                "Cargo.toml",  # Rust
                "go.mod"  # Go
            ]
            
            for dep_file in dependency_files:
                file_path = os.path.join(repository_path, dep_file)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if dep_name in content:
                            return True
                            
            # Check if dependency is imported/used in source files
            for root, dirs, files in os.walk(repository_path):
                # Skip .git and other hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if any(file.endswith(ext) for ext in ['.py', '.java', '.js', '.ts', '.cpp', '.c', '.cs']):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if dep_name in content:
                                    return True
                        except Exception:
                            continue  # Skip files that can't be read
                            
        except Exception as e:
            self.logger.error(f"Error checking dependency {dep_name}: {str(e)}")
            
        return False
        
    def _generate_integration_steps(self, dependencies: List[Dict[str, Any]], compatibility: List[Dict[str, Any]]) -> List[str]:
        """Generate steps needed to integrate the working code."""
        
        steps = []
        
        # Check for missing dependencies
        missing_deps = [c for c in compatibility if not c["compatible"]]
        
        if missing_deps:
            steps.append(f"Install {len(missing_deps)} missing dependencies:")
            for dep in missing_deps:
                steps.append(f"  - Add {dep['dependency']} to project dependencies")
                
        # Add integration steps
        steps.append("Copy the working code implementation")
        steps.append("Update imports and package references if needed")
        steps.append("Test the integration with existing codebase")
        
        if not missing_deps:
            steps.append("No additional dependencies required - code should integrate cleanly")
            
        return steps
        
    async def _perform_enhanced_structural_analysis(
        self, 
        state: AgentState, 
        target_elements: List[CodeElement],
        working_code_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform enhanced structural analysis with working code context."""
        
        # Start with regular structural analysis
        base_analysis = await self._perform_structural_analysis(state, target_elements)
        
        # Enhance with working code insights
        code_content = working_code_info["code_content"]
        file_path = working_code_info["file_path"]
        
        # Analyze code structure
        code_structure = self._analyze_code_structure(code_content, file_path)
        
        # Combine analyses
        enhanced_analysis = {
            **base_analysis,
            "working_code_structure": code_structure,
            "enhancement_confidence": min(0.9, base_analysis.get("confidence", 0.5) + 0.2)
        }
        
        return enhanced_analysis
        
    def _analyze_code_structure(self, code_content: str, file_path: str) -> Dict[str, Any]:
        """Analyze the structure of working code."""
        
        structure = {
            "file_type": file_path.split('.')[-1] if '.' in file_path else "unknown",
            "lines_of_code": len(code_content.splitlines()),
            "functions": [],
            "classes": [],
            "complexity": "medium"
        }
        
        try:
            if file_path.endswith('.py'):
                structure.update(self._analyze_python_structure(code_content))
            elif file_path.endswith('.java'):
                structure.update(self._analyze_java_structure(code_content))
            elif file_path.endswith(('.js', '.ts')):
                structure.update(self._analyze_javascript_structure(code_content))
                
        except Exception as e:
            self.logger.error(f"Failed to analyze code structure: {str(e)}")
            
        return structure
        
    def _analyze_python_structure(self, code_content: str) -> Dict[str, Any]:
        """Analyze Python code structure using AST."""
        
        structure = {"functions": [], "classes": []}
        
        try:
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args)
                    })
                elif isinstance(node, ast.ClassDef):
                    structure["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
                    
        except Exception as e:
            self.logger.error(f"Failed to parse Python AST: {str(e)}")
            
        return structure
        
    def _analyze_java_structure(self, code_content: str) -> Dict[str, Any]:
        """Analyze Java code structure using regex patterns."""
        
        structure = {"functions": [], "classes": []}
        
        # Find classes
        class_pattern = r'(?:public\s+|private\s+|protected\s+)?class\s+(\w+)'
        classes = re.finditer(class_pattern, code_content)
        
        for match in classes:
            structure["classes"].append({
                "name": match.group(1),
                "line": code_content[:match.start()].count('\n') + 1
            })
            
        # Find methods
        method_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*{'
        methods = re.finditer(method_pattern, code_content)
        
        for match in methods:
            structure["functions"].append({
                "name": match.group(1),
                "line": code_content[:match.start()].count('\n') + 1
            })
            
        return structure
        
    def _analyze_javascript_structure(self, code_content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript code structure using regex patterns."""
        
        structure = {"functions": [], "classes": []}
        
        # Find functions
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
            r'(\w+)\s*:\s*(?:async\s+)?function\s*\('
        ]
        
        for pattern in function_patterns:
            functions = re.finditer(pattern, code_content)
            for match in functions:
                structure["functions"].append({
                    "name": match.group(1),
                    "line": code_content[:match.start()].count('\n') + 1
                })
                
        # Find classes
        class_pattern = r'class\s+(\w+)'
        classes = re.finditer(class_pattern, code_content)
        
        for match in classes:
            structure["classes"].append({
                "name": match.group(1),
                "line": code_content[:match.start()].count('\n') + 1
            })
            
        return structure