"""Base agent architecture and shared utilities."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, Protocol, Callable

from pydantic import BaseModel

from ..logging import get_logger
from ..llm.client import LLMClient
from .state import AgentState, AgentFinding, Citation


logger = get_logger(__name__)


class AgentTool(BaseModel):
    """Base class for agent tools."""
    name: str
    description: str
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool."""
        pass


class LLMConfig(BaseModel):
    """Configuration for LLM interactions."""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: int = 30


class AgentConfig(BaseModel):
    """Base configuration for agents."""
    name: str
    description: str
    llm_config: LLMConfig = LLMConfig()
    tools: List[str] = []
    max_retries: int = 2
    timeout_seconds: int = 60


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, config: AgentConfig, llm_client: Optional[LLMClient] = None):
        """Initialize the agent."""
        self.config = config
        self.tools: Dict[str, AgentTool] = {}
        self.logger = get_logger(f"{__name__}.{config.name}")
        self._llm_client = llm_client
        
    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool with the agent."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
        
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's main logic."""
        pass
        
    async def _execute_with_retry(
        self, 
        func: Callable[..., Any], 
        *args, 
        **kwargs
    ) -> Any:
        """Execute a function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying: {str(e)}"
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All attempts failed: {str(e)}")
                    
        raise last_exception
        
    def _create_finding(
        self,
        finding_type: str,
        content: str,
        confidence: float,
        citations: Optional[List[Citation]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentFinding:
        """Create a standardized finding."""
        return AgentFinding(
            agent_name=self.config.name,
            finding_type=finding_type,
            content=content,
            confidence=confidence,
            citations=citations or [],
            metadata=metadata or {}
        )
        
    def _create_citation(
        self,
        file_path: str,
        description: str,
        line_number: Optional[int] = None,
        commit_sha: Optional[str] = None,
        url: Optional[str] = None
    ) -> Citation:
        """Create a standardized citation."""
        return Citation(
            file_path=file_path,
            line_number=line_number,
            commit_sha=commit_sha,
            url=url,
            description=description
        )
        
    async def _call_llm(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Call LLM with standardized configuration using dependency injection."""
        if not self._llm_client:
            # Lazy initialization if no client provided
            from ..llm.azure_client import AzureOpenAIClient
            self._llm_client = AzureOpenAIClient()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self._llm_client.chat_completion(
                messages=messages,
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                **kwargs
            )
            
            self.logger.info(f"LLM call successful for agent {self.config.name}")
            return response
            
        except Exception as e:
            self.logger.error(f"LLM call failed for agent {self.config.name}: {str(e)}")
            # Return a fallback response to prevent agent failure
            return f"LLM unavailable - fallback response for: {prompt[:100]}..."
        
    def _validate_state(self, state: AgentState) -> bool:
        """Validate that the state contains required information."""
        if not state.session_id:
            self.logger.error("State missing session_id")
            return False
            
        if not state.query:
            self.logger.error("State missing query information")
            return False
            
        return True
        
    def _log_execution_start(self, state: AgentState) -> None:
        """Log the start of agent execution."""
        self.logger.info(
            f"Starting execution for session {state.session_id}"
        )
        
    def _log_execution_end(self, state: AgentState, success: bool = True) -> None:
        """Log the end of agent execution."""
        status = "completed" if success else "failed"
        findings_count = len(state.get_findings_by_agent(self.config.name))
        
        self.logger.info(
            f"Execution {status} for session {state.session_id}, "
            f"generated {findings_count} findings"
        )


class PromptTemplate:
    """Template for managing agent prompts."""
    
    def __init__(self, template: str, variables: Optional[List[str]] = None):
        """Initialize the prompt template."""
        self.template = template
        self.variables = variables or []
        
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
            
    def validate_variables(self, **kwargs: Any) -> bool:
        """Validate that all required variables are provided."""
        for var in self.variables:
            if var not in kwargs:
                return False
        return True


class PromptManager:
    """Manager for agent prompts and templates."""
    
    def __init__(self):
        """Initialize the prompt manager."""
        self.templates: Dict[str, PromptTemplate] = {}
        
    def register_template(self, name: str, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self.templates[name] = template
        
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.templates.get(name)
        
    def format_prompt(self, template_name: str, **kwargs: Any) -> str:
        """Format a prompt using a registered template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
            
        return template.format(**kwargs)
        
    def clear_templates(self) -> None:
        """Clear all registered templates."""
        self.templates.clear()
        
    def get_template_count(self) -> int:
        """Get number of registered templates."""
        return len(self.templates)


class AgentMonitor:
    """Monitor for tracking agent performance and health."""
    
    def __init__(self):
        """Initialize the monitor."""
        self.execution_times: Dict[str, List[float]] = {}
        self.success_rates: Dict[str, List[bool]] = {}
        self.error_counts: Dict[str, int] = {}
        self._max_history_per_agent: int = 1000  # Prevent unbounded growth
        
    def record_execution(
        self, 
        agent_name: str, 
        execution_time: float, 
        success: bool
    ) -> None:
        """Record an agent execution."""
        if agent_name not in self.execution_times:
            self.execution_times[agent_name] = []
            self.success_rates[agent_name] = []
            self.error_counts[agent_name] = 0
            
        self.execution_times[agent_name].append(execution_time)
        self.success_rates[agent_name].append(success)
        
        if not success:
            self.error_counts[agent_name] += 1
            
    def get_average_execution_time(self, agent_name: str) -> Optional[float]:
        """Get average execution time for an agent."""
        times = self.execution_times.get(agent_name, [])
        return sum(times) / len(times) if times else None
        
    def get_success_rate(self, agent_name: str) -> Optional[float]:
        """Get success rate for an agent."""
        successes = self.success_rates.get(agent_name, [])
        return sum(successes) / len(successes) if successes else None
        
    def get_error_count(self, agent_name: str) -> int:
        """Get error count for an agent."""
        return self.error_counts.get(agent_name, 0)
        
    def get_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get health summary for all agents."""
        summary = {}
        
        for agent_name in self.execution_times.keys():
            summary[agent_name] = {
                "avg_execution_time": self.get_average_execution_time(agent_name),
                "success_rate": self.get_success_rate(agent_name),
                "error_count": self.get_error_count(agent_name),
                "total_executions": len(self.execution_times[agent_name])
            }
            
        return summary
        
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.execution_times.clear()
        self.success_rates.clear()
        self.error_counts.clear()
        
    def cleanup_old_metrics(self, max_entries: int = 1000) -> None:
        """Remove old metrics to prevent memory growth."""
        for agent_name in list(self.execution_times.keys()):
            if len(self.execution_times[agent_name]) > max_entries:
                # Keep only the most recent entries
                self.execution_times[agent_name] = self.execution_times[agent_name][-max_entries:]
                self.success_rates[agent_name] = self.success_rates[agent_name][-max_entries:]


# Global instances
prompt_manager = PromptManager()
agent_monitor = AgentMonitor()


# Common prompt templates
SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    """You are {agent_name}, a specialized AI agent for code intelligence analysis.

Your role: {agent_description}

Current task: {task_description}

Guidelines:
- Provide accurate, evidence-based analysis
- Include specific citations (file paths, line numbers, commit SHAs)
- Assign confidence scores based on evidence strength
- Flag uncertainties clearly
- Focus on actionable insights

Context:
Repository: {repository_path}
Query: {original_query}
Session: {session_id}
""",
    variables=["agent_name", "agent_description", "task_description", 
              "repository_path", "original_query", "session_id"]
)

ANALYSIS_PROMPT_TEMPLATE = PromptTemplate(
    """Analyze the following code elements and provide insights:

Code Elements:
{code_elements}

Analysis Focus:
{analysis_focus}

Previous Findings:
{previous_findings}

Provide your analysis in the following format:
1. Key Findings
2. Evidence and Citations
3. Confidence Assessment
4. Recommendations
""",
    variables=["code_elements", "analysis_focus", "previous_findings"]
)

# Register default templates
prompt_manager.register_template("system", SYSTEM_PROMPT_TEMPLATE)
prompt_manager.register_template("analysis", ANALYSIS_PROMPT_TEMPLATE)