"""LangGraph-based agent orchestration framework."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Union

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from ..logging import get_logger
from .state import AgentState, AgentFinding
from .base import BaseAgent


logger = get_logger(__name__)


class OrchestrationConfig(BaseModel):
    """Configuration for agent orchestration."""
    max_execution_time_seconds: int = 300  # 5 minutes
    agent_timeout_seconds: int = 60  # 1 minute per agent
    max_retries: int = 1
    enable_parallel_execution: bool = True
    graceful_degradation: bool = True


class AgentOrchestrator:
    """LangGraph-based orchestrator for multi-agent workflows."""
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        """Initialize the orchestrator."""
        self.config = config or OrchestrationConfig()
        self.agents: Dict[str, BaseAgent] = {}
        self.graph: Optional[StateGraph] = None
        self.checkpointer = MemorySaver()
        self._build_graph()
        
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")
        
    def _build_graph(self) -> None:
        """Build the LangGraph state machine."""
        # Create the state graph
        self.graph = StateGraph(AgentState)
        
        # Add nodes for each phase
        self.graph.add_node("initialize", self._initialize_workflow)
        self.graph.add_node("orchestrate", self._orchestrate_query)
        self.graph.add_node("analyze_history", self._analyze_history)
        self.graph.add_node("analyze_code", self._analyze_code)
        self.graph.add_node("synthesize", self._synthesize_results)
        self.graph.add_node("verify", self._verify_findings)
        self.graph.add_node("finalize", self._finalize_results)
        self.graph.add_node("handle_error", self._handle_error)
        
        # Set entry point
        self.graph.set_entry_point("initialize")
        
        # Add conditional edges
        self.graph.add_conditional_edges(
            "initialize",
            self._route_after_init,
            {
                "orchestrate": "orchestrate",
                "error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "orchestrate", 
            self._route_after_orchestrate,
            {
                "analyze_history": "analyze_history",
                "analyze_code": "analyze_code", 
                "error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "analyze_history",
            self._route_after_history,
            {
                "analyze_code": "analyze_code",
                "synthesize": "synthesize",
                "error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "analyze_code",
            self._route_after_analysis,
            {
                "synthesize": "synthesize",
                "error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "synthesize",
            self._route_after_synthesis,
            {
                "verify": "verify",
                "error": "handle_error"
            }
        )
        
        self.graph.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {
                "finalize": "finalize",
                "error": "handle_error"
            }
        )
        
        # Terminal nodes
        self.graph.add_edge("finalize", END)
        self.graph.add_edge("handle_error", END)
        
        # Compile the graph
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        
    async def execute_query(
        self, 
        query: str, 
        repository_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Execute a query using the multi-agent workflow."""
        session_id = str(uuid.uuid4())
        
        # Initialize state
        initial_state = AgentState(
            session_id=session_id,
            query={
                "original": query,
                "options": options or {}
            },
            repository={
                "path": repository_path
            }
        )
        
        logger.info(f"Starting query execution: {session_id}")
        
        try:
            # Execute the workflow with timeout
            config = {"configurable": {"thread_id": session_id}}
            
            result = await asyncio.wait_for(
                self.compiled_graph.ainvoke(initial_state, config=config),
                timeout=self.config.max_execution_time_seconds
            )
            
            logger.info(f"Query execution completed: {session_id}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Query execution timed out: {session_id}")
            initial_state.add_error("Query execution timed out")
            return initial_state
            
        except Exception as e:
            logger.error(f"Query execution failed: {session_id}, error: {str(e)}")
            initial_state.add_error(f"Execution failed: {str(e)}")
            return initial_state
    
    async def _initialize_workflow(self, state: AgentState) -> AgentState:
        """Initialize the workflow."""
        state.update_progress("orchestrator", "initializing", "processing")
        
        # Validate inputs
        if not state.query.get("original"):
            state.add_error("No query provided")
            return state
            
        if not state.repository.get("path"):
            state.add_error("No repository path provided")
            return state
            
        logger.info(f"Initialized workflow for session: {state.session_id}")
        return state
        
    async def _orchestrate_query(self, state: AgentState) -> AgentState:
        """Orchestrate the query parsing and workflow planning."""
        state.update_progress("orchestrator", "parsing_query", "processing")
        
        if "orchestrator" in self.agents:
            try:
                result = await self._execute_agent_with_timeout(
                    "orchestrator", state
                )
                state = result
            except Exception as e:
                state.add_error(f"Orchestrator failed: {str(e)}", "orchestrator")
        
        # Also execute other registered agents that aren't part of the main workflow
        for agent_name in self.agents:
            if agent_name not in ["orchestrator", "historian", "analyst", "synthesizer", "verifier"]:
                try:
                    result = await self._execute_agent_with_timeout(agent_name, state)
                    state = result
                except Exception as e:
                    state.add_error(f"Agent {agent_name} failed: {str(e)}", agent_name)
                
        return state
        
    async def _analyze_history(self, state: AgentState) -> AgentState:
        """Analyze git history."""
        state.update_progress("historian", "analyzing_history", "processing")
        
        if "historian" in self.agents:
            try:
                result = await self._execute_agent_with_timeout(
                    "historian", state
                )
                state = result
            except Exception as e:
                state.add_error(f"Historian failed: {str(e)}", "historian")
                
        return state
        
    async def _analyze_code(self, state: AgentState) -> AgentState:
        """Analyze code structure and semantics."""
        state.update_progress("analyst", "analyzing_code", "processing")
        
        if "analyst" in self.agents:
            try:
                result = await self._execute_agent_with_timeout(
                    "analyst", state
                )
                state = result
            except Exception as e:
                state.add_error(f"Analyst failed: {str(e)}", "analyst")
                
        return state
        
    async def _synthesize_results(self, state: AgentState) -> AgentState:
        """Synthesize results from multiple agents."""
        state.update_progress("synthesizer", "synthesizing_results", "processing")
        
        if "synthesizer" in self.agents:
            try:
                result = await self._execute_agent_with_timeout(
                    "synthesizer", state
                )
                state = result
            except Exception as e:
                state.add_error(f"Synthesizer failed: {str(e)}", "synthesizer")
                
        return state
        
    async def _verify_findings(self, state: AgentState) -> AgentState:
        """Verify findings against source data."""
        state.update_progress("verifier", "verifying_findings", "processing")
        
        if "verifier" in self.agents:
            try:
                result = await self._execute_agent_with_timeout(
                    "verifier", state
                )
                state = result
            except Exception as e:
                state.add_error(f"Verifier failed: {str(e)}", "verifier")
                
        return state
        
    async def _finalize_results(self, state: AgentState) -> AgentState:
        """Finalize and format results."""
        state.update_progress("orchestrator", "finalizing", "completed")
        
        # Calculate final confidence scores
        all_findings = state.get_all_findings()
        if all_findings:
            avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings)
            state.verification["overall_confidence"] = avg_confidence
            
        logger.info(f"Finalized results for session: {state.session_id}")
        return state
        
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors and implement graceful degradation."""
        state.update_progress("orchestrator", "handling_error", "error")
        
        if self.config.graceful_degradation and state.get_all_findings():
            # If we have some findings, continue with reduced confidence
            state.add_warning("Partial results due to agent failures")
            state.verification["degraded"] = True
            return await self._finalize_results(state)
        else:
            logger.error(f"Workflow failed for session: {state.session_id}")
            return state
            
    async def _execute_agent_with_timeout(
        self, 
        agent_name: str, 
        state: AgentState
    ) -> AgentState:
        """Execute an agent with timeout handling."""
        agent = self.agents[agent_name]
        
        try:
            result = await asyncio.wait_for(
                agent.execute(state),
                timeout=self.config.agent_timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            state.add_error(f"Agent {agent_name} timed out", agent_name)
            return state
            
    def _route_after_init(self, state: AgentState) -> str:
        """Route after initialization."""
        if state.has_errors():
            return "error"
        return "orchestrate"
        
    def _route_after_orchestrate(self, state: AgentState) -> str:
        """Route after orchestration."""
        if state.has_errors():
            return "error"
        
        # Determine if we need history analysis
        query_data = state.query.get("parsed", {})
        if query_data.get("time_range") or "history" in state.query.get("original", "").lower():
            return "analyze_history"
        else:
            return "analyze_code"
            
    def _route_after_history(self, state: AgentState) -> str:
        """Route after history analysis."""
        if state.has_errors() and not self.config.graceful_degradation:
            return "error"
        return "analyze_code"
        
    def _route_after_analysis(self, state: AgentState) -> str:
        """Route after code analysis."""
        if state.has_errors() and not self.config.graceful_degradation:
            return "error"
        return "synthesize"
        
    def _route_after_synthesis(self, state: AgentState) -> str:
        """Route after synthesis."""
        if state.has_errors() and not self.config.graceful_degradation:
            return "error"
        return "verify"
        
    def _route_after_verification(self, state: AgentState) -> str:
        """Route after verification."""
        if state.has_errors() and not self.config.graceful_degradation:
            return "error"
        return "finalize"