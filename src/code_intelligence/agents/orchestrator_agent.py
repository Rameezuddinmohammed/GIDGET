"""Orchestrator Agent for query parsing and workflow management."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, ParsedQuery, QueryScope, CodeElement
from .config import get_agent_config


logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """Agent responsible for query parsing and workflow coordination."""
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize the Orchestrator Agent."""
        if config is None:
            config = AgentConfig(
                name="orchestrator",
                description="Parses natural language queries and coordinates multi-agent workflows"
            )
        super().__init__(config, **kwargs)
        
        # Register query parsing templates
        self._register_templates()
        
    def _register_templates(self) -> None:
        """Register prompt templates for query parsing."""
        query_parsing_template = PromptTemplate(
            """You are an expert code intelligence query parser. Parse the following natural language query about code analysis.

Query: "{query}"
Repository: {repository_path}

Extract the following information and respond in JSON format:

{{
    "intent": "primary intent (e.g., 'find_changes', 'analyze_function', 'trace_dependencies')",
    "entities": ["list", "of", "code", "entities", "mentioned"],
    "time_range": "time range if specified (e.g., 'last_week', 'since_commit_abc123', null)",
    "scope": "analysis scope (function|class|file|module|repository)",
    "keywords": ["key", "terms", "extracted"],
    "requires_history": true/false,
    "requires_semantic_search": true/false,
    "complexity": "low|medium|high"
}}

Examples:
- "What changed in the login function since last week?" → intent: "find_changes", entities: ["login"], time_range: "last_week", scope: "function"
- "Find all functions that call authenticate()" → intent: "trace_dependencies", entities: ["authenticate"], scope: "repository"
- "How did the UserService class evolve?" → intent: "analyze_evolution", entities: ["UserService"], scope: "class"

Parse the query and provide only the JSON response:""",
            variables=["query", "repository_path"]
        )
        
        workflow_planning_template = PromptTemplate(
            """Based on the parsed query, determine the optimal workflow for analysis.

Parsed Query: {parsed_query}
Available Agents: historian, analyst, synthesizer, verifier

Determine:
1. Which agents are needed
2. Execution order and dependencies
3. Expected analysis focus areas
4. Estimated complexity and time

Respond in JSON format:
{{
    "required_agents": ["list", "of", "required", "agents"],
    "execution_plan": [
        {{"agent": "agent_name", "priority": 1, "focus": "what to analyze"}},
        {{"agent": "agent_name", "priority": 2, "focus": "what to analyze"}}
    ],
    "parallel_execution": true/false,
    "estimated_duration": "time estimate",
    "complexity_score": 0.0-1.0
}}""",
            variables=["parsed_query"]
        )
        
        self.query_parsing_template = query_parsing_template
        self.workflow_planning_template = workflow_planning_template
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute orchestrator logic."""
        self._log_execution_start(state)
        
        if not self._validate_state(state):
            state.add_error("Invalid state for orchestrator", self.config.name)
            return state
            
        try:
            # Parse the natural language query
            parsed_query = await self._parse_query(state)
            state.query["parsed"] = parsed_query.model_dump()
            
            # Plan the workflow
            workflow_plan = await self._plan_workflow(state, parsed_query)
            state.analysis["workflow_plan"] = workflow_plan
            
            # Initialize analysis context
            await self._initialize_analysis_context(state, parsed_query)
            
            # Create orchestrator finding
            finding = self._create_finding(
                finding_type="query_orchestration",
                content=f"Parsed query with intent: {parsed_query.intent}. "
                       f"Planned workflow with {len(workflow_plan.get('required_agents', []))} agents.",
                confidence=0.95,
                metadata={
                    "parsed_query": parsed_query.model_dump(),
                    "workflow_plan": workflow_plan
                }
            )
            state.add_finding(self.config.name, finding)
            
            self._log_execution_end(state, True)
            return state
            
        except Exception as e:
            error_context = {
                "session_id": state.session_id,
                "query": state.query.get("original", ""),
                "repository_path": state.repository.get("path", ""),
                "error_type": type(e).__name__
            }
            self.logger.error(f"Orchestrator execution failed: {str(e)}", extra=error_context)
            state.add_error(f"Orchestrator failed: {str(e)}", self.config.name)
            self._log_execution_end(state, False)
            return state
            
    async def _parse_query(self, state: AgentState) -> ParsedQuery:
        """Parse natural language query using LLM."""
        query = state.query.get("original", "")
        repository_path = state.repository.get("path", "")
        
        prompt = self.query_parsing_template.format(
            query=query,
            repository_path=repository_path
        )
        
        system_prompt = (
            "You are an expert at parsing natural language queries about code analysis. "
            "Extract structured information and respond only with valid JSON."
        )
        
        response = await self._call_llm(prompt, system_prompt)
        
        try:
            # Extract JSON from response with sanitization
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Sanitize JSON string to prevent code injection
                json_str = self._sanitize_json_string(json_str)
                parsed_data = json.loads(json_str)
            else:
                # Fallback parsing
                parsed_data = self._fallback_query_parsing(query)
                
            return ParsedQuery(**parsed_data)
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse LLM response, using fallback: {str(e)}")
            return self._fallback_query_parsing(query)
            
    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Fallback query parsing using rule-based approach."""
        query_lower = query.lower()
        
        # Determine intent
        intent = "general_analysis"
        if any(word in query_lower for word in ["change", "diff", "evolution", "history"]):
            intent = "find_changes"
        elif any(word in query_lower for word in ["call", "depend", "use", "reference"]):
            intent = "trace_dependencies"
        elif any(word in query_lower for word in ["function", "method"]):
            intent = "analyze_function"
        elif any(word in query_lower for word in ["class"]):
            intent = "analyze_class"
            
        # Extract entities (simple approach)
        entities = []
        # Look for camelCase or snake_case identifiers with timeout protection
        entity_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:[A-Z][a-zA-Z0-9_]*)*\b'
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Regex operation timed out")
            
            # Set timeout for regex operation
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2)  # 2 second timeout
            
            potential_entities = re.findall(entity_pattern, query)
            
            signal.alarm(0)  # Cancel timeout
        except (TimeoutError, AttributeError):
            # Fallback to simple word extraction on timeout or Windows (no signal)
            potential_entities = [word for word in query.split() if word.isalnum() and len(word) > 2]
        config = get_agent_config()
        entities = [e for e in potential_entities if len(e) > 2 and e not in ["the", "and", "for", "with"]]
        
        # Determine scope
        scope = QueryScope.REPOSITORY
        if "function" in query_lower or "method" in query_lower:
            scope = QueryScope.FUNCTION
        elif "class" in query_lower:
            scope = QueryScope.CLASS
        elif "file" in query_lower:
            scope = QueryScope.FILE
            
        # Check for time range
        time_range = None
        if any(word in query_lower for word in ["since", "last", "recent", "history"]):
            time_range = "recent"
            
        return {
            "intent": intent,
            "entities": entities[:config.limits.max_elements_per_analysis],  # Configurable limit
            "time_range": time_range,
            "scope": scope.value,
            "keywords": query.split()[:10],  # First 10 words as keywords
            "requires_history": "history" in query_lower or "change" in query_lower,
            "requires_semantic_search": "find" in query_lower or "search" in query_lower,
            "complexity": "medium"
        }
        
    async def _plan_workflow(self, state: AgentState, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Plan the optimal workflow based on parsed query."""
        prompt = self.workflow_planning_template.format(
            parsed_query=parsed_query.model_dump_json()
        )
        
        system_prompt = (
            "You are an expert at planning multi-agent workflows for code analysis. "
            "Determine the optimal sequence and configuration of agents."
        )
        
        response = await self._call_llm(prompt, system_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_workflow_planning(parsed_query)
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse workflow plan, using fallback: {str(e)}")
            return self._fallback_workflow_planning(parsed_query)
            
    def _fallback_workflow_planning(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Fallback workflow planning using rule-based approach."""
        required_agents = ["analyst", "synthesizer", "verifier"]
        execution_plan = []
        
        # Add historian if temporal analysis is needed
        if parsed_query.time_range or "history" in parsed_query.intent:
            required_agents.insert(0, "historian")
            execution_plan.append({
                "agent": "historian",
                "priority": 1,
                "focus": "temporal_analysis"
            })
            
        # Add analyst for code structure analysis
        execution_plan.append({
            "agent": "analyst", 
            "priority": 2,
            "focus": "structural_analysis"
        })
        
        # Add synthesizer for result compilation
        execution_plan.append({
            "agent": "synthesizer",
            "priority": 3, 
            "focus": "result_synthesis"
        })
        
        # Add verifier for validation
        execution_plan.append({
            "agent": "verifier",
            "priority": 4,
            "focus": "finding_validation"
        })
        
        return {
            "required_agents": required_agents,
            "execution_plan": execution_plan,
            "parallel_execution": len(required_agents) <= 3,
            "estimated_duration": "2-5 minutes",
            "complexity_score": 0.6
        }
        
    async def _initialize_analysis_context(self, state: AgentState, parsed_query: ParsedQuery) -> None:
        """Initialize analysis context for other agents."""
        # Set target elements based on entities
        target_elements = []
        for entity in parsed_query.entities:
            target_elements.append(CodeElement(
                name=entity,
                type="unknown",  # Will be determined by analyst
                file_path="",    # Will be resolved by analyst
            ))
            
        state.analysis.update({
            "target_elements": [elem.model_dump() for elem in target_elements],
            "scope": parsed_query.scope.value,
            "focus_areas": self._determine_focus_areas(parsed_query),
            "analysis_depth": self._determine_analysis_depth(parsed_query)
        })
        
    def _determine_focus_areas(self, parsed_query: ParsedQuery) -> List[str]:
        """Determine focus areas for analysis."""
        focus_areas = []
        
        if "change" in parsed_query.intent or "evolution" in parsed_query.intent:
            focus_areas.append("temporal_changes")
            
        if "depend" in parsed_query.intent or "call" in parsed_query.intent:
            focus_areas.append("dependency_analysis")
            
        if "function" in parsed_query.intent or parsed_query.scope == QueryScope.FUNCTION:
            focus_areas.append("function_analysis")
            
        if "class" in parsed_query.intent or parsed_query.scope == QueryScope.CLASS:
            focus_areas.append("class_analysis")
            
        # Default focus areas
        if not focus_areas:
            focus_areas = ["structural_analysis", "semantic_analysis"]
            
        return focus_areas
        
    def _determine_analysis_depth(self, parsed_query: ParsedQuery) -> str:
        """Determine the depth of analysis required."""
        if parsed_query.scope in [QueryScope.FUNCTION, QueryScope.CLASS]:
            return "deep"
        elif parsed_query.scope == QueryScope.FILE:
            return "medium"
        else:
            return "broad"
            
    def get_user_preferences(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user preferences for query processing."""
        # Default preferences - in a real implementation, this would
        # fetch from a user preferences database
        return {
            "preferred_detail_level": "medium",
            "include_citations": True,
            "max_results": 50,
            "confidence_threshold": 0.7,
            "preferred_agents": ["historian", "analyst", "synthesizer", "verifier"],
            "timeout_seconds": 300
        }
        
    def integrate_user_context(self, state: AgentState, user_id: Optional[str] = None) -> None:
        """Integrate user context and preferences into the analysis."""
        preferences = self.get_user_preferences(user_id)
        
        # Update query options with user preferences
        state.query.setdefault("options", {}).update({
            "user_preferences": preferences,
            "detail_level": preferences["preferred_detail_level"],
            "confidence_threshold": preferences["confidence_threshold"]
        })
        
        # Update analysis configuration
        state.analysis.setdefault("configuration", {}).update({
            "max_results": preferences["max_results"],
            "include_citations": preferences["include_citations"],
            "timeout_seconds": preferences["timeout_seconds"]
        })
        
    def _sanitize_json_string(self, json_str: str) -> str:
        """Sanitize JSON string to prevent code injection."""
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess',
            r'os\.',
            r'sys\.',
            r'open\s*\(',
            r'file\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            json_str = re.sub(pattern, '', json_str, flags=re.IGNORECASE)
            
        return json_str