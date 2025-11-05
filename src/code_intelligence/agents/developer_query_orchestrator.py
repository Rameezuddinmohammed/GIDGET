"""Developer Query Orchestrator for handling real-world developer scenarios."""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..database.supabase_client import supabase_client
from ..caching.cache_manager import cache_manager
from ..monitoring.agent_monitor import agent_monitor
from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, ParsedQuery, QueryScope, CodeElement


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
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute developer-focused orchestration with intelligent caching."""
        execution_id = await agent_monitor.start_execution(
            self.config.name, state.session_id
        )
        
        self._log_execution_start(state)
        
        if not self._validate_state(state):
            state.add_error("Invalid state for developer orchestrator", self.config.name)
            await agent_monitor.record_execution(
                execution_id, False, error_message="Invalid state"
            )
            return state
            
        try:
            # Check for cached results first
            cached_result = await self._check_cache(state)
            if cached_result:
                logger.info("Using cached result for developer query")
                state = await self._apply_cached_result(state, cached_result)
                await agent_monitor.record_execution(
                    execution_id, True, 
                    findings_count=len(state.get_all_findings()),
                    confidence_score=cached_result.get("confidence_score", 0.0)
                )
                return state
            
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
            
            await agent_monitor.record_execution(
                execution_id, True,
                findings_count=len(state.get_all_findings()),
                confidence_score=finding.confidence
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Developer orchestrator execution failed: {str(e)}")
            state.add_error(f"Developer orchestrator failed: {str(e)}", self.config.name)
            self._log_execution_end(state, False)
            
            await agent_monitor.record_execution(
                execution_id, False, error_message=str(e)
            )
            
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
        
        # Simple validation based on findings and confidence
        all_findings = state.get_all_findings()
        if not all_findings:
            return {
                "approved": False,
                "confidence": 0.0,
                "ready_for_developer": False,
                "recommendation": "REJECT: No findings available"
            }
        
        # Calculate average confidence
        avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings)
        solution_approved = avg_confidence >= 0.7
        
        # Generate delivery decision
        delivery_decision = {
            "approved": solution_approved,
            "confidence": avg_confidence,
            "ready_for_developer": solution_approved and avg_confidence >= 0.8,
            "recommendation": ""
        }
        
        if delivery_decision["ready_for_developer"]:
            delivery_decision["recommendation"] = "DELIVER: Solution meets developer requirements"
        elif solution_approved:
            delivery_decision["recommendation"] = "CONDITIONAL: Solution partially meets requirements"
        else:
            delivery_decision["recommendation"] = "REJECT: Solution does not meet minimum requirements"
            
        return delivery_decision
        
    async def _check_cache(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Check for cached results for the current query."""
        try:
            query = state.query.get("original", "")
            repository_id = state.repository.get("id") or state.repository.get("path", "")
            options = state.query.get("options", {})
            
            if not query or not repository_id:
                return None
                
            cached_result = await cache_manager.get_cached_result(
                query=query,
                repository_id=repository_id,
                options=options
            )
            
            return cached_result
            
        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            return None
            
    async def _apply_cached_result(
        self, 
        state: AgentState, 
        cached_result: Dict[str, Any]
    ) -> AgentState:
        """Apply cached result to the current state."""
        try:
            result_data = cached_result.get("result_data", {})
            
            # Apply cached findings
            if "findings" in result_data:
                for agent_name, findings in result_data["findings"].items():
                    for finding_data in findings:
                        # Reconstruct finding object
                        finding = self._create_finding(
                            finding_type=finding_data.get("finding_type", "cached"),
                            content=finding_data.get("content", ""),
                            confidence=finding_data.get("confidence", 0.0),
                            metadata=finding_data.get("metadata", {}),
                            citations=finding_data.get("citations", [])
                        )
                        # Set agent_name for the finding
                        finding.agent_name = agent_name
                        state.add_finding(agent_name, finding)
            
            # Apply cached analysis data
            if "analysis" in result_data:
                for key, value in result_data["analysis"].items():
                    state.analysis[key] = value
                    
            # Apply cached verification data
            if "verification" in result_data:
                for key, value in result_data["verification"].items():
                    state.verification[key] = value
                    
            # Mark as cached result
            state.analysis["from_cache"] = True
            state.analysis["cache_timestamp"] = cached_result.get("created_at")
            state.verification["cache_confidence"] = cached_result.get("confidence_score", 0.0)
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to apply cached result: {str(e)}")
            return state
            
    async def store_final_result(self, state: AgentState) -> bool:
        """Store the final verified result in cache."""
        try:
            query = state.query.get("original", "")
            repository_id = state.repository.get("id") or state.repository.get("path", "")
            options = state.query.get("options", {})
            
            if not query or not repository_id:
                return False
                
            # Calculate overall confidence
            all_findings = state.get_all_findings()
            if not all_findings:
                return False
                
            overall_confidence = state.verification.get(
                "overall_confidence",
                sum(f.confidence for f in all_findings) / len(all_findings)
            )
            
            # Prepare result data for caching
            result_data = {
                "findings": {},
                "analysis": state.analysis,
                "verification": state.verification,
                "query_metadata": {
                    "session_id": state.session_id,
                    "execution_time": state.progress.get("execution_time"),
                    "agent_count": len(state.agent_results)
                }
            }
            
            # Serialize findings by agent
            for agent_name, findings in state.agent_results.items():
                result_data["findings"][agent_name] = [
                    {
                        "finding_type": f.finding_type,
                        "content": f.content,
                        "confidence": f.confidence,
                        "metadata": f.metadata,
                        "citations": f.citations
                    }
                    for f in findings
                ]
            
            # Check if result should be cached
            if not await cache_manager.should_cache_result(overall_confidence, result_data):
                logger.debug("Result not suitable for caching", confidence=overall_confidence)
                return False
                
            # Store in cache
            success = await cache_manager.store_result(
                query=query,
                repository_id=repository_id,
                result_data=result_data,
                confidence_score=overall_confidence,
                options=options
            )
            
            if success:
                logger.info("Final result stored in cache", confidence=overall_confidence)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store final result: {str(e)}")
            return False