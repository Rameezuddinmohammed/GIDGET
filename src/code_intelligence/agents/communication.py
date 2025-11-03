"""Agent communication and coordination protocols."""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from ..logging import get_logger
from .state import AgentState, AgentFinding, Citation


logger = get_logger(__name__)


class MessageType(str, Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    COORDINATION = "coordination"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessage(BaseModel):
    """Message between agents."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str
    recipient: str
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    requires_response: bool = False
    response_timeout: Optional[int] = None


class AgentDependency(BaseModel):
    """Dependency relationship between agents."""
    dependent: str  # Agent that depends on another
    dependency: str  # Agent that is depended upon
    dependency_type: str  # Type of dependency (data, coordination, etc.)
    required: bool = True  # Whether this dependency is required


class ConflictResolution(BaseModel):
    """Conflict resolution result."""
    conflict_id: str
    agents_involved: List[str]
    resolution_strategy: str
    resolved_finding: Optional[AgentFinding] = None
    confidence_adjustment: float = 0.0
    resolution_notes: str


class AgentCommunicationProtocol:
    """Protocol for agent communication and coordination."""
    
    def __init__(self):
        """Initialize the communication protocol."""
        self.message_queue: Dict[str, List[AgentMessage]] = {}
        self.dependencies: List[AgentDependency] = []
        self.execution_order: List[str] = []
        self.conflicts: Dict[str, List[AgentFinding]] = {}
        self.resolutions: List[ConflictResolution] = []
        
    def register_dependency(self, dependency: AgentDependency) -> None:
        """Register a dependency between agents."""
        self.dependencies.append(dependency)
        logger.info(
            f"Registered dependency: {dependency.dependent} -> {dependency.dependency}"
        )
        
    def calculate_execution_order(self, agents: List[str]) -> List[str]:
        """Calculate optimal execution order based on dependencies."""
        # Topological sort to determine execution order
        in_degree = {agent: 0 for agent in agents}
        graph = {agent: [] for agent in agents}
        
        # Build dependency graph
        for dep in self.dependencies:
            if dep.dependent in agents and dep.dependency in agents:
                graph[dep.dependency].append(dep.dependent)
                in_degree[dep.dependent] += 1
                
        # Topological sort
        queue = [agent for agent in agents if in_degree[agent] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        # Check for circular dependencies
        if len(result) != len(agents):
            logger.warning("Circular dependencies detected, using fallback order")
            return agents
            
        self.execution_order = result
        logger.info(f"Calculated execution order: {result}")
        return result
        
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to an agent."""
        if message.recipient not in self.message_queue:
            self.message_queue[message.recipient] = []
            
        self.message_queue[message.recipient].append(message)
        
        logger.debug(
            f"Message sent from {message.sender} to {message.recipient}: "
            f"{message.message_type}"
        )
        
    async def receive_messages(self, agent_name: str) -> List[AgentMessage]:
        """Receive messages for an agent."""
        messages = self.message_queue.get(agent_name, [])
        self.message_queue[agent_name] = []
        
        # Sort by priority and timestamp
        messages.sort(
            key=lambda m: (
                self._priority_value(m.priority), 
                m.timestamp
            )
        )
        
        return messages
        
    def _priority_value(self, priority: MessagePriority) -> int:
        """Convert priority to numeric value for sorting."""
        priority_map = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3
        }
        return priority_map.get(priority, 2)
        
    async def coordinate_agents(
        self, 
        state: AgentState, 
        active_agents: List[str]
    ) -> AgentState:
        """Coordinate communication between active agents."""
        # Send coordination messages
        for agent in active_agents:
            coord_message = AgentMessage(
                sender="coordinator",
                recipient=agent,
                message_type=MessageType.COORDINATION,
                content={
                    "session_id": state.session_id,
                    "active_agents": active_agents,
                    "execution_order": self.execution_order,
                    "shared_state": self._extract_shared_context(state)
                }
            )
            await self.send_message(coord_message)
            
        return state
        
    def _extract_shared_context(self, state: AgentState) -> Dict[str, Any]:
        """Extract shared context for agent coordination."""
        return {
            "query": state.query,
            "repository": state.repository,
            "progress": state.progress,
            "findings_summary": self._summarize_findings(state)
        }
        
    def _summarize_findings(self, state: AgentState) -> Dict[str, Any]:
        """Summarize findings for sharing between agents."""
        all_findings = state.get_all_findings()
        
        return {
            "total_findings": len(all_findings),
            "by_agent": {
                agent: len(findings) 
                for agent, findings in state.agent_results.items()
            },
            "avg_confidence": (
                sum(f.confidence for f in all_findings) / len(all_findings)
                if all_findings else 0.0
            ),
            "finding_types": list(set(f.finding_type for f in all_findings))
        }


class ConflictResolver:
    """Resolver for conflicts between agent findings."""
    
    def __init__(self):
        """Initialize the conflict resolver."""
        self.resolution_strategies = {
            "confidence_weighted": self._confidence_weighted_resolution,
            "evidence_based": self._evidence_based_resolution,
            "consensus": self._consensus_resolution,
            "verification_required": self._verification_required_resolution
        }
        
    async def detect_conflicts(
        self, 
        state: AgentState
    ) -> List[Tuple[str, List[AgentFinding]]]:
        """Detect conflicts between agent findings."""
        conflicts = []
        all_findings = state.get_all_findings()
        
        # Group findings by topic/entity
        finding_groups = self._group_findings_by_topic(all_findings)
        
        for topic, findings in finding_groups.items():
            if len(findings) > 1:
                # Check for contradictory findings
                if self._are_findings_contradictory(findings):
                    conflict_id = str(uuid4())
                    conflicts.append((conflict_id, findings))
                    
        logger.info(f"Detected {len(conflicts)} conflicts")
        return conflicts
        
    def _group_findings_by_topic(
        self, 
        findings: List[AgentFinding]
    ) -> Dict[str, List[AgentFinding]]:
        """Group findings by topic or entity."""
        groups = {}
        
        for finding in findings:
            # Group by finding type only for simpler conflict detection
            topic = finding.finding_type
            
            if topic not in groups:
                groups[topic] = []
            groups[topic].append(finding)
            
        return groups
        
    def _extract_key_entity(self, finding: AgentFinding) -> str:
        """Extract key entity from finding for grouping."""
        # Simple extraction - in practice, this would be more sophisticated
        content_words = finding.content.split()[:5]
        return "_".join(content_words).lower()
        
    def _are_findings_contradictory(self, findings: List[AgentFinding]) -> bool:
        """Check if findings are contradictory."""
        # Simple contradiction detection
        # In practice, this would use NLP and semantic analysis
        
        if len(findings) < 2:
            return False
            
        # Check for explicit contradiction keywords
        contradiction_keywords = [
            ("increased", "decreased"),
            ("added", "removed"),
            ("improved", "degraded"),
            ("fixed", "broken"),
            ("present", "absent"),
            ("performance increased", "performance decreased"),
            ("significantly", "decreased")  # More flexible matching
        ]
        
        contents = [f.content.lower() for f in findings]
        
        for i, content1 in enumerate(contents):
            for j, content2 in enumerate(contents):
                if i != j:  # Don't compare with self
                    for pos_word, neg_word in contradiction_keywords:
                        if pos_word in content1 and neg_word in content2:
                            return True
                        if neg_word in content1 and pos_word in content2:
                            return True
                            
        return False
        
    async def resolve_conflict(
        self, 
        conflict_id: str, 
        findings: List[AgentFinding],
        strategy: str = "confidence_weighted"
    ) -> ConflictResolution:
        """Resolve a conflict between findings."""
        if strategy not in self.resolution_strategies:
            strategy = "confidence_weighted"
            
        resolver = self.resolution_strategies[strategy]
        resolution = await resolver(conflict_id, findings)
        
        logger.info(
            f"Resolved conflict {conflict_id} using {strategy} strategy"
        )
        
        return resolution
        
    async def _confidence_weighted_resolution(
        self, 
        conflict_id: str, 
        findings: List[AgentFinding]
    ) -> ConflictResolution:
        """Resolve conflict by weighting findings by confidence."""
        # Select finding with highest confidence
        best_finding = max(findings, key=lambda f: f.confidence)
        
        # Adjust confidence based on conflict
        confidence_penalty = 0.1 * (len(findings) - 1)
        adjusted_confidence = max(0.0, best_finding.confidence - confidence_penalty)
        
        # Create adjusted finding
        resolved_finding = AgentFinding(
            agent_name="conflict_resolver",
            finding_type=best_finding.finding_type,
            content=f"[RESOLVED] {best_finding.content}",
            confidence=adjusted_confidence,
            citations=best_finding.citations,
            metadata={
                **best_finding.metadata,
                "resolution_method": "confidence_weighted",
                "original_confidence": best_finding.confidence,
                "conflicting_agents": [f.agent_name for f in findings]
            }
        )
        
        return ConflictResolution(
            conflict_id=conflict_id,
            agents_involved=[f.agent_name for f in findings],
            resolution_strategy="confidence_weighted",
            resolved_finding=resolved_finding,
            confidence_adjustment=-confidence_penalty,
            resolution_notes=f"Selected finding from {best_finding.agent_name} "
                           f"with highest confidence ({best_finding.confidence:.2f})"
        )
        
    async def _evidence_based_resolution(
        self, 
        conflict_id: str, 
        findings: List[AgentFinding]
    ) -> ConflictResolution:
        """Resolve conflict based on evidence strength."""
        # Score findings by evidence quality
        evidence_scores = []
        
        for finding in findings:
            score = len(finding.citations) * 0.3  # Citation count
            score += len(finding.metadata) * 0.1  # Metadata richness
            
            # Bonus for specific citations
            for citation in finding.citations:
                if citation.line_number is not None:
                    score += 0.2
                if citation.commit_sha is not None:
                    score += 0.2
                    
            evidence_scores.append((finding, score))
            
        # Select finding with best evidence
        best_finding, best_score = max(evidence_scores, key=lambda x: x[1])
        
        resolved_finding = AgentFinding(
            agent_name="conflict_resolver",
            finding_type=best_finding.finding_type,
            content=f"[EVIDENCE-BASED] {best_finding.content}",
            confidence=best_finding.confidence,
            citations=best_finding.citations,
            metadata={
                **best_finding.metadata,
                "resolution_method": "evidence_based",
                "evidence_score": best_score
            }
        )
        
        return ConflictResolution(
            conflict_id=conflict_id,
            agents_involved=[f.agent_name for f in findings],
            resolution_strategy="evidence_based",
            resolved_finding=resolved_finding,
            confidence_adjustment=0.0,
            resolution_notes=f"Selected finding from {best_finding.agent_name} "
                           f"with strongest evidence (score: {best_score:.2f})"
        )
        
    async def _consensus_resolution(
        self, 
        conflict_id: str, 
        findings: List[AgentFinding]
    ) -> ConflictResolution:
        """Resolve conflict by finding consensus elements."""
        # Extract common elements from findings
        common_citations = self._find_common_citations(findings)
        common_content = self._extract_common_content(findings)
        
        avg_confidence = sum(f.confidence for f in findings) / len(findings)
        
        resolved_finding = AgentFinding(
            agent_name="conflict_resolver",
            finding_type="consensus",
            content=f"[CONSENSUS] {common_content}",
            confidence=avg_confidence * 0.8,  # Reduce confidence for consensus
            citations=common_citations,
            metadata={
                "resolution_method": "consensus",
                "source_agents": [f.agent_name for f in findings],
                "original_findings_count": len(findings)
            }
        )
        
        return ConflictResolution(
            conflict_id=conflict_id,
            agents_involved=[f.agent_name for f in findings],
            resolution_strategy="consensus",
            resolved_finding=resolved_finding,
            confidence_adjustment=-0.2,
            resolution_notes=f"Synthesized consensus from {len(findings)} findings"
        )
        
    async def _verification_required_resolution(
        self, 
        conflict_id: str, 
        findings: List[AgentFinding]
    ) -> ConflictResolution:
        """Mark conflict as requiring verification."""
        return ConflictResolution(
            conflict_id=conflict_id,
            agents_involved=[f.agent_name for f in findings],
            resolution_strategy="verification_required",
            resolved_finding=None,
            confidence_adjustment=0.0,
            resolution_notes=f"Conflict requires manual verification. "
                           f"{len(findings)} contradictory findings detected."
        )
        
    def _find_common_citations(
        self, 
        findings: List[AgentFinding]
    ) -> List[Citation]:
        """Find citations that appear in multiple findings."""
        citation_counts = {}
        
        for finding in findings:
            for citation in finding.citations:
                key = (citation.file_path, citation.line_number)
                if key not in citation_counts:
                    citation_counts[key] = []
                citation_counts[key].append(citation)
                
        # Return citations that appear in multiple findings
        common_citations = []
        for citations in citation_counts.values():
            if len(citations) > 1:
                common_citations.append(citations[0])  # Take first instance
                
        return common_citations
        
    def _extract_common_content(self, findings: List[AgentFinding]) -> str:
        """Extract common content elements from findings."""
        # Simple implementation - find common words
        all_words = []
        for finding in findings:
            words = finding.content.lower().split()
            all_words.extend(words)
            
        # Find words that appear in multiple findings
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        common_words = [
            word for word, count in word_counts.items() 
            if count > 1 and len(word) > 3
        ]
        
        return f"Common elements: {', '.join(common_words[:10])}"


class StateValidator:
    """Validator for agent state consistency."""
    
    def __init__(self):
        """Initialize the state validator."""
        self.validation_rules = [
            self._validate_session_consistency,
            self._validate_finding_integrity,
            self._validate_citation_validity,
            self._validate_confidence_ranges,
            self._validate_temporal_consistency
        ]
        
    async def validate_state(self, state: AgentState) -> List[str]:
        """Validate state consistency and return any issues."""
        issues = []
        
        for rule in self.validation_rules:
            try:
                rule_issues = await rule(state)
                issues.extend(rule_issues)
            except Exception as e:
                issues.append(f"Validation rule failed: {str(e)}")
                
        return issues
        
    async def _validate_session_consistency(self, state: AgentState) -> List[str]:
        """Validate session-level consistency."""
        issues = []
        
        if not state.session_id:
            issues.append("Missing session ID")
            
        if not state.query:
            issues.append("Missing query information")
            
        return issues
        
    async def _validate_finding_integrity(self, state: AgentState) -> List[str]:
        """Validate finding integrity."""
        issues = []
        
        for agent_name, findings in state.agent_results.items():
            for i, finding in enumerate(findings):
                if finding.agent_name != agent_name:
                    issues.append(
                        f"Finding {i} from {agent_name} has incorrect agent_name: "
                        f"{finding.agent_name}"
                    )
                    
                if not finding.content.strip():
                    issues.append(f"Empty finding content from {agent_name}")
                    
        return issues
        
    async def _validate_citation_validity(self, state: AgentState) -> List[str]:
        """Validate citation validity."""
        issues = []
        
        all_findings = state.get_all_findings()
        for finding in all_findings:
            for citation in finding.citations:
                if not citation.file_path:
                    issues.append(
                        f"Citation missing file_path in finding from "
                        f"{finding.agent_name}"
                    )
                    
                if citation.line_number is not None and citation.line_number < 1:
                    issues.append(
                        f"Invalid line number {citation.line_number} in citation "
                        f"from {finding.agent_name}"
                    )
                    
        return issues
        
    async def _validate_confidence_ranges(self, state: AgentState) -> List[str]:
        """Validate confidence score ranges."""
        issues = []
        
        all_findings = state.get_all_findings()
        for finding in all_findings:
            if not (0.0 <= finding.confidence <= 1.0):
                issues.append(
                    f"Invalid confidence score {finding.confidence} from "
                    f"{finding.agent_name}"
                )
                
        return issues
        
    async def _validate_temporal_consistency(self, state: AgentState) -> List[str]:
        """Validate temporal consistency."""
        issues = []
        
        if state.created_at > state.updated_at:
            issues.append("Created timestamp is after updated timestamp")
            
        # Check finding timestamps
        all_findings = state.get_all_findings()
        for finding in all_findings:
            if finding.timestamp < state.created_at:
                issues.append(
                    f"Finding timestamp {finding.timestamp} is before state "
                    f"creation {state.created_at}"
                )
                
        return issues