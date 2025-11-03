"""Developer Query Orchestrator for handling real-world developer scenarios."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, ParsedQuery, QueryScope, CodeElement
from .solution_verifier import SolutionVerifier


logger = get_logger(__name__)


class DeveloperQueryOrchestrator(BaseAgent):
    """Orchestrator designed for real developer queries requiring actionable solutions."""
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize the Developer Query Orchestrator."""
        if config is None:
            config = AgentConfig(
                name="developer_orchestrator",
                description="Orchestrates multi-agent workflows for real developer problem-solving"
            )
        super().__init__(config, **kwargs)
        
        # Initialize solution verifier
        self.solution_verifier = SolutionVerifier()
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute developer-focused orchestration."""
        self._log_execution_start(state)
        
        if not self._validate_state(state):
            state.add_error("Invalid state for developer orchestrator", self.config.name)
            return state
            
        try:
            # Parse developer query for actionable requirements
            developer_intent = await self._parse_developer_query(state)
            state.query["developer_intent"] = developer_intent
            
            # Plan solution-oriented workflow
            solution_plan = await self._plan_solution_workflow(state, developer_intent)
            state.analysis["solution_plan"] = solution_plan
            
            # Set success criteria
            success_criteria = self._define_success_criteria(developer_intent)
            state.analysis["success_criteria"] = success_criteria
            
            # Create orchestrator finding
            finding = self._create_finding(
                finding_type="developer_query_orchestration",
                content=f"Parsed developer query: {developer_intent['problem_type']}. "
                       f"Solution plan: {solution_plan['approach']}. "
                       f"Success threshold: {success_criteria['confidence_threshold']:.0%}",
                confidence=0.95,
                metadata={
                    "developer_intent": developer_intent,
                    "solution_plan": solution_plan,
                    "success_criteria": success_criteria
                }
            )
            state.add_finding(self.config.name, finding)
            
            self._log_execution_end(state, True)
            return state
            
        except Exception as e:
            self.logger.error(f"Developer orchestrator execution failed: {str(e)}")
            state.add_error(f"Developer orchestrator failed: {str(e)}", self.config.name)
            self._log_execution_end(state, False)
            return state
            
    async def _parse_developer_query(self, state: AgentState) -> Dict[str, Any]:
        """Parse developer query to understand the actual problem and required solution."""
        
        query = state.query.get("original", "")
        query_lower = query.lower()
        
        developer_intent = {
            "problem_type": "unknown",
            "urgency": "medium",
            "deliverables_needed": [],
            "context": {},
            "success_definition": ""
        }
        
        # Identify problem type
        if any(phrase in query_lower for phrase in ["was working", "used to work", "worked before"]):
            developer_intent["problem_type"] = "regression_analysis"
            developer_intent["urgency"] = "high"  # Regressions are urgent
            
        elif any(phrase in query_lower for phrase in ["how to implement", "how do i", "need to build"]):
            developer_intent["problem_type"] = "implementation_guidance"
            
        elif any(phrase in query_lower for phrase in ["find where", "locate", "which version"]):
            developer_intent["problem_type"] = "code_archaeology"
            
        elif any(phrase in query_lower for phrase in ["deadlock", "performance", "slow", "hanging"]):
            developer_intent["problem_type"] = "performance_issue"
            developer_intent["urgency"] = "high"
            
        # Identify required deliverables
        deliverables = []
        
        if any(phrase in query_lower for phrase in ["give me code", "show me", "extract"]):
            deliverables.append("working_code")
            
        if any(phrase in query_lower for phrase in ["dependencies", "what needs", "requirements"]):
            deliverables.append("dependency_list")
            
        if any(phrase in query_lower for phrase in ["how to integrate", "integration", "steps"]):
            deliverables.append("integration_steps")
            
        if any(phrase in query_lower for phrase in ["which version", "when", "commit"]):
            deliverables.append("version_identification")
            
        if any(phrase in query_lower for phrase in ["why", "what changed", "difference"]):
            deliverables.append("root_cause_analysis")
            
        developer_intent["deliverables_needed"] = deliverables
        
        # Extract context
        context = {}
        
        # Feature context
        feature_match = re.search(r'(?:feature|functionality|component)\s+(\w+)', query_lower)
        if feature_match:
            context["target_feature"] = feature_match.group(1)
            
        # Problem context  
        if "deadlock" in query_lower:
            context["problem_symptom"] = "deadlock"
        elif any(word in query_lower for word in ["broken", "not working", "failing"]):
            context["problem_symptom"] = "functionality_broken"
            
        # Timeline context
        if any(phrase in query_lower for phrase in ["today", "now", "current"]):
            context["timeline"] = "immediate"
        elif any(phrase in query_lower for phrase in ["past", "before", "previously"]):
            context["timeline"] = "historical"
            
        developer_intent["context"] = context
        
        # Define success criteria
        if developer_intent["problem_type"] == "regression_analysis":
            developer_intent["success_definition"] = "Identify working version, extract code, provide integration steps"
        elif developer_intent["problem_type"] == "implementation_guidance":
            developer_intent["success_definition"] = "Provide working code example with clear implementation steps"
        else:
            developer_intent["success_definition"] = "Answer developer question with actionable information"
            
        return developer_intent
        
    async def _plan_solution_workflow(self, state: AgentState, developer_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Plan workflow focused on delivering actionable solutions."""
        
        problem_type = developer_intent["problem_type"]
        deliverables = developer_intent["deliverables_needed"]
        
        solution_plan = {
            "approach": "unknown",
            "agent_sequence": [],
            "validation_strategy": "solution_verification",
            "delivery_threshold": 0.8  # 80% confidence minimum
        }
        
        # Plan based on problem type
        if problem_type == "regression_analysis":
            solution_plan["approach"] = "historical_code_recovery"
            solution_plan["agent_sequence"] = [
                {"agent": "historian", "focus": "find_working_version", "priority": 1},
                {"agent": "analyst", "focus": "extract_dependencies", "priority": 2},
                {"agent": "synthesizer", "focus": "integration_plan", "priority": 3},
                {"agent": "solution_verifier", "focus": "validate_completeness", "priority": 4}
            ]
            solution_plan["delivery_threshold"] = 0.85  # Higher threshold for regressions
            
        elif problem_type == "code_archaeology":
            solution_plan["approach"] = "historical_investigation"
            solution_plan["agent_sequence"] = [
                {"agent": "historian", "focus": "timeline_analysis", "priority": 1},
                {"agent": "analyst", "focus": "code_location", "priority": 2},
                {"agent": "synthesizer", "focus": "findings_compilation", "priority": 3},
                {"agent": "solution_verifier", "focus": "validate_accuracy", "priority": 4}
            ]
            
        elif problem_type == "performance_issue":
            solution_plan["approach"] = "performance_analysis"
            solution_plan["agent_sequence"] = [
                {"agent": "analyst", "focus": "dependency_analysis", "priority": 1},
                {"agent": "historian", "focus": "change_analysis", "priority": 2},
                {"agent": "synthesizer", "focus": "root_cause_report", "priority": 3},
                {"agent": "solution_verifier", "focus": "validate_diagnosis", "priority": 4}
            ]
            
        else:
            # Default workflow
            solution_plan["approach"] = "general_investigation"
            solution_plan["agent_sequence"] = [
                {"agent": "analyst", "focus": "structural_analysis", "priority": 1},
                {"agent": "historian", "focus": "historical_context", "priority": 2},
                {"agent": "synthesizer", "focus": "comprehensive_report", "priority": 3},
                {"agent": "solution_verifier", "focus": "validate_quality", "priority": 4}
            ]
            
        return solution_plan
        
    def _define_success_criteria(self, developer_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Define what constitutes success for this developer query."""
        
        deliverables = developer_intent["deliverables_needed"]
        problem_type = developer_intent["problem_type"]
        
        success_criteria = {
            "confidence_threshold": 0.8,  # 80% minimum
            "required_deliverables": deliverables,
            "quality_gates": []
        }
        
        # Set higher thresholds for critical problems
        if developer_intent["urgency"] == "high":
            success_criteria["confidence_threshold"] = 0.85  # 85% for urgent issues
            
        # Define quality gates
        quality_gates = []
        
        if "working_code" in deliverables:
            quality_gates.append({
                "gate": "code_extraction",
                "requirement": "Must provide actual code with file references",
                "validation": "verify_code_citations"
            })
            
        if "version_identification" in deliverables:
            quality_gates.append({
                "gate": "version_accuracy", 
                "requirement": "Must identify specific commit/version with high confidence",
                "validation": "verify_historical_accuracy"
            })
            
        if "integration_steps" in deliverables:
            quality_gates.append({
                "gate": "actionable_steps",
                "requirement": "Must provide clear, executable integration steps",
                "validation": "verify_step_completeness"
            })
            
        success_criteria["quality_gates"] = quality_gates
        
        return success_criteria
        
    async def validate_solution_delivery(self, state: AgentState) -> Dict[str, Any]:
        """Validate if the solution meets developer requirements for delivery."""
        
        # Use solution verifier
        verification_state = await self.solution_verifier.execute(state)
        
        # Extract verification results
        verification_data = verification_state.verification
        solution_approved = verification_data.get("delivery_approved", False)
        confidence = verification_data.get("solution_confidence", 0.0)
        
        # Generate delivery decision
        delivery_decision = {
            "approved": solution_approved,
            "confidence": confidence,
            "ready_for_developer": solution_approved and confidence >= 0.8,
            "recommendation": ""
        }
        
        if delivery_decision["ready_for_developer"]:
            delivery_decision["recommendation"] = "DELIVER: Solution meets developer requirements"
        elif solution_approved:
            delivery_decision["recommendation"] = "CONDITIONAL: Solution partially meets requirements"
        else:
            delivery_decision["recommendation"] = "REJECT: Solution does not meet minimum requirements"
            
        return delivery_decision