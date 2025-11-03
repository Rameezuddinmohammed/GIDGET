"""Synthesizer Agent for result compilation and report generation."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, AgentFinding, Citation


logger = get_logger(__name__)


class SynthesizerAgent(BaseAgent):
    """Agent responsible for multi-source result aggregation and narrative generation."""
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize the Synthesizer Agent."""
        if config is None:
            config = AgentConfig(
                name="synthesizer",
                description="Aggregates findings from multiple agents and generates coherent analysis reports"
            )
        super().__init__(config, **kwargs)
        
        # Register synthesis templates
        self._register_templates()
        
    def _register_templates(self) -> None:
        """Register prompt templates for result synthesis."""
        synthesis_template = PromptTemplate(
            """Synthesize findings from multiple specialized agents into a coherent analysis report.

Original Query: {original_query}
Agent Findings:
{agent_findings}

Create a comprehensive report that:
1. Directly answers the original query
2. Integrates findings from all agents
3. Resolves any conflicts or contradictions
4. Provides a clear narrative flow
5. Includes specific citations and evidence
6. Highlights key insights and recommendations

Structure the response as:
## Executive Summary
[Brief overview of key findings]

## Detailed Analysis
[Comprehensive analysis integrating all agent findings]

## Key Insights
[Most important discoveries and patterns]

## Evidence and Citations
[Supporting evidence with specific references]

## Recommendations
[Actionable recommendations based on findings]

## Confidence Assessment
[Overall confidence in findings with uncertainty notes]

Ensure the report is coherent, well-structured, and directly addresses the user's query.""",
            variables=["original_query", "agent_findings"]
        )
        
        conflict_resolution_template = PromptTemplate(
            """Resolve conflicts between agent findings and determine the most accurate conclusion.

Conflicting Findings:
{conflicting_findings}

For each conflict:
1. Analyze the evidence strength from each agent
2. Consider the agent's specialization and reliability
3. Look for supporting or contradicting evidence
4. Determine the most likely accurate conclusion
5. Assign confidence scores

Respond in JSON format:
{{
    "resolved_conflicts": [
        {{
            "conflict_topic": "topic description",
            "conflicting_agents": ["agent1", "agent2"],
            "resolution": "resolved conclusion",
            "reasoning": "why this resolution was chosen",
            "confidence": 0.0-1.0,
            "remaining_uncertainty": "any remaining uncertainties"
        }}
    ],
    "overall_confidence": 0.0-1.0
}}""",
            variables=["conflicting_findings"]
        )
        
        self.synthesis_template = synthesis_template
        self.conflict_resolution_template = conflict_resolution_template
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute synthesizer logic."""
        self._log_execution_start(state)
        
        if not self._validate_state(state):
            state.add_error("Invalid state for synthesizer", self.config.name)
            return state
            
        try:
            # Collect all agent findings
            all_findings = state.get_all_findings()
            
            if not all_findings:
                state.add_warning("No findings to synthesize", self.config.name)
                return state
                
            # Organize findings by agent and type
            organized_findings = self._organize_findings(all_findings)
            
            # Detect and resolve conflicts
            conflicts = await self._detect_conflicts(organized_findings)
            resolved_conflicts = await self._resolve_conflicts(conflicts) if conflicts else []
            
            # Check if we have working code and integration analysis
            working_code_available = self._check_working_code_availability(organized_findings)
            integration_analysis = self._extract_integration_analysis(organized_findings)
            
            if working_code_available and integration_analysis:
                # Generate executable solution steps
                solution_steps = await self._generate_executable_solution(state, organized_findings, integration_analysis)
                
                # Generate solution-oriented synthesis
                synthesis_report = await self._generate_solution_synthesis(state, organized_findings, solution_steps)
            else:
                # Fallback to comprehensive analysis
                synthesis_report = await self._generate_synthesis_report(state, organized_findings, resolved_conflicts)
                solution_steps = []
            
            # Create citation index
            citation_index = self._create_citation_index(all_findings)
            
            # Generate final recommendations
            recommendations = await self._generate_recommendations(state, organized_findings)
            
            # Create synthesizer findings
            synthesis_finding = self._create_finding(
                finding_type="comprehensive_synthesis",
                content=synthesis_report,
                confidence=self._calculate_overall_confidence(all_findings, resolved_conflicts),
                citations=self._extract_key_citations(all_findings),
                metadata={
                    "agent_count": len(organized_findings),
                    "total_findings": len(all_findings),
                    "conflicts_resolved": len(resolved_conflicts),
                    "citation_count": len(citation_index)
                }
            )
            state.add_finding(self.config.name, synthesis_finding)
            
            # Add recommendations as separate finding
            if recommendations:
                recommendations_finding = self._create_finding(
                    finding_type="recommendations",
                    content=recommendations,
                    confidence=0.8,
                    citations=[],
                    metadata={"recommendation_count": len(recommendations.split("\n")) if recommendations else 0}
                )
                state.add_finding(self.config.name, recommendations_finding)
                
            # Update state with synthesis data
            state.analysis["synthesis"] = {
                "organized_findings": organized_findings,
                "resolved_conflicts": resolved_conflicts,
                "citation_index": citation_index,
                "overall_confidence": synthesis_finding.confidence
            }
            
            self._log_execution_end(state, True)
            return state
            
        except Exception as e:
            self.logger.error(f"Synthesizer execution failed: {str(e)}")
            state.add_error(f"Synthesizer failed: {str(e)}", self.config.name)
            self._log_execution_end(state, False)
            return state
            
    def _organize_findings(self, findings: List[AgentFinding]) -> Dict[str, Dict[str, List[AgentFinding]]]:
        """Organize findings by agent and finding type."""
        organized = {}
        
        for finding in findings:
            agent_name = finding.agent_name
            finding_type = finding.finding_type
            
            if agent_name not in organized:
                organized[agent_name] = {}
                
            if finding_type not in organized[agent_name]:
                organized[agent_name][finding_type] = []
                
            organized[agent_name][finding_type].append(finding)
            
        return organized
        
    async def _detect_conflicts(self, organized_findings: Dict[str, Dict[str, List[AgentFinding]]]) -> List[Dict[str, Any]]:
        """Detect conflicts between agent findings."""
        conflicts = []
        
        # Compare findings across agents for similar topics
        agent_names = list(organized_findings.keys())
        
        for i, agent1 in enumerate(agent_names):
            for j, agent2 in enumerate(agent_names[i+1:], i+1):
                agent1_findings = organized_findings[agent1]
                agent2_findings = organized_findings[agent2]
                
                # Look for conflicting findings on similar topics
                for type1, findings1 in agent1_findings.items():
                    for type2, findings2 in agent2_findings.items():
                        # Check if findings might be about the same topic
                        if self._are_findings_related(type1, type2):
                            potential_conflicts = self._find_conflicting_content(findings1, findings2)
                            for conflict in potential_conflicts:
                                conflicts.append({
                                    "agents": [agent1, agent2],
                                    "finding_types": [type1, type2],
                                    "conflict": conflict
                                })
                                
        return conflicts
        
    def _are_findings_related(self, type1: str, type2: str) -> bool:
        """Check if two finding types might be related."""
        # Simple heuristic - findings are related if they share keywords
        type1_words = set(type1.lower().split("_"))
        type2_words = set(type2.lower().split("_"))
        
        # Check for common words (excluding common terms)
        common_words = type1_words.intersection(type2_words)
        common_words.discard("analysis")
        common_words.discard("finding")
        common_words.discard("result")
        
        return len(common_words) > 0
        
    def _find_conflicting_content(self, findings1: List[AgentFinding], findings2: List[AgentFinding]) -> List[Dict[str, Any]]:
        """Find conflicting content between two sets of findings."""
        conflicts = []
        
        for f1 in findings1:
            for f2 in findings2:
                # Simple conflict detection based on confidence and content similarity
                if abs(f1.confidence - f2.confidence) > 0.3:  # Significant confidence difference
                    # Check if they're talking about similar things
                    if self._have_content_overlap(f1.content, f2.content):
                        conflicts.append({
                            "finding1": f1,
                            "finding2": f2,
                            "conflict_type": "confidence_mismatch",
                            "description": f"Confidence mismatch: {f1.confidence:.2f} vs {f2.confidence:.2f}"
                        })
                        
        return conflicts
        
    def _have_content_overlap(self, content1: str, content2: str) -> bool:
        """Check if two content strings have significant overlap."""
        # Simple word-based overlap check
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        min_length = min(len(words1), len(words2))
        
        return overlap / min_length > 0.3  # 30% overlap threshold
        
    async def _resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between agent findings."""
        if not conflicts:
            return []
            
        # Prepare conflict data for LLM resolution
        conflict_descriptions = []
        for conflict in conflicts:
            agents = conflict["agents"]
            conflict_info = conflict["conflict"]
            
            description = (
                f"Conflict between {agents[0]} and {agents[1]}:\n"
                f"Finding 1: {conflict_info['finding1'].content[:200]}... (confidence: {conflict_info['finding1'].confidence})\n"
                f"Finding 2: {conflict_info['finding2'].content[:200]}... (confidence: {conflict_info['finding2'].confidence})\n"
                f"Conflict type: {conflict_info['conflict_type']}\n"
            )
            conflict_descriptions.append(description)
            
        prompt = self.conflict_resolution_template.format(
            conflicting_findings="\n---\n".join(conflict_descriptions)
        )
        
        system_prompt = (
            "You are an expert at resolving conflicts between different analysis results. "
            "Use evidence strength, agent specialization, and logical reasoning to determine the most accurate conclusions."
        )
        
        response = await self._call_llm(prompt, system_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group()).get("resolved_conflicts", [])
            else:
                return self._fallback_conflict_resolution(conflicts)
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse conflict resolution, using fallback: {str(e)}")
            return self._fallback_conflict_resolution(conflicts)
            
    def _fallback_conflict_resolution(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback conflict resolution using simple heuristics."""
        resolved = []
        
        for conflict in conflicts:
            conflict_info = conflict["conflict"]
            finding1 = conflict_info["finding1"]
            finding2 = conflict_info["finding2"]
            
            # Simple resolution: choose the finding with higher confidence
            if finding1.confidence > finding2.confidence:
                chosen_finding = finding1
                chosen_agent = conflict["agents"][0]
            else:
                chosen_finding = finding2
                chosen_agent = conflict["agents"][1]
                
            resolved.append({
                "conflict_topic": conflict_info["conflict_type"],
                "conflicting_agents": conflict["agents"],
                "resolution": chosen_finding.content,
                "reasoning": f"Chose finding from {chosen_agent} due to higher confidence ({chosen_finding.confidence:.2f})",
                "confidence": chosen_finding.confidence,
                "remaining_uncertainty": "Automated resolution - manual review recommended"
            })
            
        return resolved
        
    async def _generate_synthesis_report(
        self, 
        state: AgentState, 
        organized_findings: Dict[str, Dict[str, List[AgentFinding]]],
        resolved_conflicts: List[Dict[str, Any]]
    ) -> str:
        """Generate comprehensive synthesis report."""
        original_query = state.query.get("original", "")
        
        # Prepare agent findings summary
        findings_summary = []
        for agent_name, finding_types in organized_findings.items():
            agent_summary = f"\n{agent_name.upper()} FINDINGS:\n"
            for finding_type, findings in finding_types.items():
                agent_summary += f"- {finding_type}: {len(findings)} findings\n"
                for finding in findings[:2]:  # Top 2 findings per type
                    agent_summary += f"  * {finding.content[:150]}...\n"
            findings_summary.append(agent_summary)
            
        prompt = self.synthesis_template.format(
            original_query=original_query,
            agent_findings="\n".join(findings_summary)
        )
        
        system_prompt = (
            "You are an expert at synthesizing complex technical analysis from multiple sources. "
            "Create a coherent, comprehensive report that directly addresses the user's query."
        )
        
        response = await self._call_llm(prompt, system_prompt)
        
        # Add conflict resolution section if there were conflicts
        if resolved_conflicts:
            conflict_section = "\n\n## Conflict Resolution\n"
            for resolution in resolved_conflicts:
                conflict_section += f"- **{resolution['conflict_topic']}**: {resolution['resolution']}\n"
                conflict_section += f"  - Reasoning: {resolution['reasoning']}\n"
                if resolution.get('remaining_uncertainty'):
                    conflict_section += f"  - Uncertainty: {resolution['remaining_uncertainty']}\n"
            response += conflict_section
            
        return response
        
    def _create_citation_index(self, findings: List[AgentFinding]) -> Dict[str, List[Citation]]:
        """Create an index of all citations."""
        citation_index = {}
        
        for finding in findings:
            for citation in finding.citations:
                file_path = citation.file_path
                if file_path not in citation_index:
                    citation_index[file_path] = []
                citation_index[file_path].append(citation)
                
        return citation_index
        
    def _extract_key_citations(self, findings: List[AgentFinding]) -> List[Citation]:
        """Extract the most important citations."""
        all_citations = []
        for finding in findings:
            all_citations.extend(finding.citations)
            
        # Sort by relevance (simple heuristic: prefer citations with line numbers and commit SHAs)
        def citation_score(citation: Citation) -> float:
            score = 0.0
            if citation.line_number:
                score += 0.5
            if citation.commit_sha:
                score += 0.3
            if citation.url:
                score += 0.2
            return score
            
        sorted_citations = sorted(all_citations, key=citation_score, reverse=True)
        return sorted_citations[:10]  # Top 10 citations
        
    async def _generate_recommendations(
        self, 
        state: AgentState, 
        organized_findings: Dict[str, Dict[str, List[AgentFinding]]]
    ) -> str:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        # Extract recommendations from findings
        for agent_name, finding_types in organized_findings.items():
            for finding_type, findings in finding_types.items():
                for finding in findings:
                    # Look for recommendation-related content
                    if any(word in finding.content.lower() for word in ["recommend", "suggest", "should", "improve"]):
                        recommendations.append(f"- {finding.content}")
                        
                    # Extract specific recommendations from metadata
                    if "recommendations" in finding.metadata:
                        recs = finding.metadata["recommendations"]
                        if isinstance(recs, list):
                            for rec in recs:
                                if isinstance(rec, dict):
                                    recommendations.append(f"- {rec.get('description', str(rec))}")
                                else:
                                    recommendations.append(f"- {str(rec)}")
                                    
        # Generate additional recommendations based on patterns
        if "circular_dependencies" in str(organized_findings):
            recommendations.append("- Consider refactoring to eliminate circular dependencies")
            
        if "complexity" in str(organized_findings):
            recommendations.append("- Review high-complexity components for potential simplification")
            
        return "\n".join(recommendations[:10]) if recommendations else "No specific recommendations generated."
        
    def _calculate_overall_confidence(
        self, 
        findings: List[AgentFinding], 
        resolved_conflicts: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score for the synthesis."""
        if not findings:
            return 0.0
            
        # Base confidence is average of all findings
        base_confidence = sum(f.confidence for f in findings) / len(findings)
        
        # Reduce confidence based on conflicts
        conflict_penalty = len(resolved_conflicts) * 0.05  # 5% penalty per conflict
        
        # Reduce confidence if we have very few findings
        if len(findings) < 3:
            base_confidence *= 0.8
            
        final_confidence = max(base_confidence - conflict_penalty, 0.1)  # Minimum 10% confidence
        return min(final_confidence, 1.0)  # Maximum 100% confidence
        
    def format_structured_report(
        self, 
        synthesis_report: str, 
        organized_findings: Dict[str, Dict[str, List[AgentFinding]]],
        citation_index: Dict[str, List[Citation]]
    ) -> Dict[str, Any]:
        """Format synthesis into structured report format."""
        # Extract sections from the synthesis report
        sections = {}
        current_section = "summary"
        current_content = []
        
        for line in synthesis_report.split("\n"):
            if line.startswith("##"):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                    
                # Start new section
                current_section = line.replace("##", "").strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)
                
        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()
            
        # Create structured report
        structured_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "sections": sections,
            "agent_summary": {
                agent: len(findings) for agent, findings in organized_findings.items()
            },
            "citation_summary": {
                file_path: len(citations) for file_path, citations in citation_index.items()
            },
            "metadata": {
                "total_findings": sum(len(findings) for findings in organized_findings.values()),
                "total_citations": sum(len(citations) for citations in citation_index.values()),
                "agents_involved": list(organized_findings.keys())
            }
        }
        
        return structured_report
        
    def create_executive_summary(self, organized_findings: Dict[str, Dict[str, List[AgentFinding]]]) -> str:
        """Create executive summary of key findings."""
        summary_points = []
        
        # Count findings by type
        finding_counts = {}
        for agent_findings in organized_findings.values():
            for finding_type, findings in agent_findings.items():
                finding_counts[finding_type] = finding_counts.get(finding_type, 0) + len(findings)
                
        # Generate summary points
        total_findings = sum(finding_counts.values())
        summary_points.append(f"Analyzed {total_findings} findings from {len(organized_findings)} specialized agents")
        
        # Highlight top finding types
        top_types = sorted(finding_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for finding_type, count in top_types:
            summary_points.append(f"- {finding_type.replace('_', ' ').title()}: {count} findings")
            
        return "\n".join(summary_points)
        
    def _check_working_code_availability(self, organized_findings: Dict[str, Dict[str, List[AgentFinding]]]) -> bool:
        """Check if working code was extracted by historian."""
        
        historian_findings = organized_findings.get("historian", {})
        
        for finding_type, findings in historian_findings.items():
            if finding_type == "working_code_extraction":
                return len(findings) > 0
                
        return False
        
    def _extract_integration_analysis(self, organized_findings: Dict[str, Dict[str, List[AgentFinding]]]) -> Optional[Dict[str, Any]]:
        """Extract integration analysis from analyst findings."""
        
        analyst_findings = organized_findings.get("analyst", {})
        
        for finding_type, findings in analyst_findings.items():
            if finding_type == "integration_analysis":
                if findings:
                    return findings[0].metadata  # Return first integration analysis
                    
        return None
        
    async def _generate_executable_solution(
        self, 
        state: AgentState, 
        organized_findings: Dict[str, Dict[str, List[AgentFinding]]],
        integration_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate executable solution steps for the developer."""
        
        solution_steps = []
        
        try:
            # Step 1: Extract working code information
            working_code_info = self._get_working_code_info(organized_findings)
            
            if working_code_info:
                solution_steps.append({
                    "step": 1,
                    "title": "Extract Working Code",
                    "description": f"Copy working implementation from commit {working_code_info['commit_sha'][:8]}",
                    "action": "code_extraction",
                    "details": {
                        "file_path": working_code_info["file_path"],
                        "commit_sha": working_code_info["commit_sha"],
                        "code_length": working_code_info.get("code_length", 0)
                    },
                    "confidence": working_code_info.get("confidence", 0.7)
                })
                
            # Step 2: Handle dependencies
            dependencies = integration_analysis.get("dependencies", [])
            missing_deps = [d for d in integration_analysis.get("compatibility", []) if not d.get("compatible", False)]
            
            if missing_deps:
                solution_steps.append({
                    "step": 2,
                    "title": "Install Missing Dependencies",
                    "description": f"Add {len(missing_deps)} missing dependencies to project",
                    "action": "dependency_installation",
                    "details": {
                        "missing_dependencies": [d["dependency"] for d in missing_deps],
                        "installation_commands": self._generate_install_commands(missing_deps)
                    },
                    "confidence": 0.8
                })
            else:
                solution_steps.append({
                    "step": 2,
                    "title": "Dependencies Check",
                    "description": "All required dependencies are already available",
                    "action": "dependency_verification",
                    "details": {"status": "all_available"},
                    "confidence": 0.9
                })
                
            # Step 3: Integration steps
            integration_steps = integration_analysis.get("integration_steps", [])
            if integration_steps:
                solution_steps.append({
                    "step": 3,
                    "title": "Integrate Code",
                    "description": "Follow integration steps to add working code to codebase",
                    "action": "code_integration",
                    "details": {
                        "integration_steps": integration_steps,
                        "estimated_time": f"{len(integration_steps) * 5} minutes"
                    },
                    "confidence": integration_analysis.get("confidence", 0.7)
                })
                
            # Step 4: Testing and validation
            solution_steps.append({
                "step": 4,
                "title": "Test Integration",
                "description": "Verify the integrated code works correctly",
                "action": "testing",
                "details": {
                    "test_types": ["unit_tests", "integration_tests", "manual_verification"],
                    "validation_points": self._generate_validation_points(working_code_info, integration_analysis)
                },
                "confidence": 0.8
            })
            
        except Exception as e:
            self.logger.error(f"Failed to generate executable solution: {str(e)}")
            solution_steps.append({
                "step": 1,
                "title": "Solution Generation Failed",
                "description": f"Could not generate executable steps: {str(e)}",
                "action": "error",
                "details": {"error": str(e)},
                "confidence": 0.1
            })
            
        return solution_steps
        
    def _get_working_code_info(self, organized_findings: Dict[str, Dict[str, List[AgentFinding]]]) -> Optional[Dict[str, Any]]:
        """Extract working code information from historian findings."""
        
        historian_findings = organized_findings.get("historian", {})
        
        for finding_type, findings in historian_findings.items():
            if finding_type == "working_code_extraction" and findings:
                finding = findings[0]  # Take first working code finding
                return {
                    "commit_sha": finding.metadata.get("commit_sha", ""),
                    "file_path": finding.metadata.get("file_path", ""),
                    "code_length": finding.metadata.get("code_length", 0),
                    "confidence": finding.confidence,
                    "description": finding.content
                }
                
        return None
        
    def _generate_install_commands(self, missing_deps: List[Dict[str, Any]]) -> List[str]:
        """Generate installation commands for missing dependencies."""
        
        commands = []
        
        # Group by dependency type
        python_deps = []
        java_deps = []
        js_deps = []
        
        for dep in missing_deps:
            dep_name = dep["dependency"]
            if any(keyword in dep_name.lower() for keyword in ["python", "pip", "pypi"]):
                python_deps.append(dep_name)
            elif any(keyword in dep_name.lower() for keyword in ["java", "maven", "gradle"]):
                java_deps.append(dep_name)
            elif any(keyword in dep_name.lower() for keyword in ["npm", "node", "javascript"]):
                js_deps.append(dep_name)
            else:
                # Default to Python
                python_deps.append(dep_name)
                
        # Generate commands
        if python_deps:
            commands.append(f"pip install {' '.join(python_deps)}")
            
        if java_deps:
            commands.append("# Add to pom.xml or build.gradle:")
            for dep in java_deps:
                commands.append(f"# - {dep}")
                
        if js_deps:
            commands.append(f"npm install {' '.join(js_deps)}")
            
        return commands if commands else ["# No installation commands needed"]
        
    def _generate_validation_points(self, working_code_info: Optional[Dict[str, Any]], integration_analysis: Dict[str, Any]) -> List[str]:
        """Generate validation points for testing the integration."""
        
        validation_points = []
        
        if working_code_info:
            validation_points.append(f"Verify {working_code_info['file_path']} compiles without errors")
            validation_points.append("Test the main functionality described in the original query")
            
        dependencies = integration_analysis.get("dependencies", [])
        if dependencies:
            validation_points.append("Confirm all dependencies are properly imported")
            
        validation_points.extend([
            "Run existing tests to ensure no regressions",
            "Perform manual testing of the integrated feature",
            "Check for any performance impacts"
        ])
        
        return validation_points
        
    async def _generate_solution_synthesis(
        self, 
        state: AgentState, 
        organized_findings: Dict[str, Dict[str, List[AgentFinding]]],
        solution_steps: List[Dict[str, Any]]
    ) -> str:
        """Generate solution-oriented synthesis report."""
        
        original_query = state.query.get("original", "")
        working_code_info = self._get_working_code_info(organized_findings)
        
        # Calculate overall solution confidence
        step_confidences = [step.get("confidence", 0.5) for step in solution_steps]
        overall_confidence = sum(step_confidences) / len(step_confidences) if step_confidences else 0.5
        
        synthesis_parts = []
        
        # Executive Summary
        synthesis_parts.append("## Executive Summary")
        if working_code_info:
            synthesis_parts.append(f"✅ **SOLUTION FOUND**: Working implementation extracted from commit {working_code_info['commit_sha'][:8]}")
            synthesis_parts.append(f"📁 **File**: {working_code_info['file_path']}")
            synthesis_parts.append(f"🎯 **Confidence**: {overall_confidence:.1%}")
            synthesis_parts.append(f"⏱️ **Estimated Integration Time**: {len(solution_steps) * 10} minutes")
        else:
            synthesis_parts.append("❌ **No working implementation found in git history**")
            synthesis_parts.append("📋 **Analysis provided instead of executable solution**")
            
        synthesis_parts.append("")
        
        # Solution Steps
        synthesis_parts.append("## Executable Solution Steps")
        for step in solution_steps:
            synthesis_parts.append(f"### Step {step['step']}: {step['title']}")
            synthesis_parts.append(f"**Description**: {step['description']}")
            synthesis_parts.append(f"**Confidence**: {step['confidence']:.1%}")
            
            if step['action'] == 'code_extraction' and 'details' in step:
                details = step['details']
                synthesis_parts.append(f"- Extract from: `{details['file_path']}`")
                synthesis_parts.append(f"- Commit: `{details['commit_sha']}`")
                synthesis_parts.append(f"- Code size: {details['code_length']} characters")
                
            elif step['action'] == 'dependency_installation' and 'details' in step:
                details = step['details']
                synthesis_parts.append("**Installation Commands**:")
                for cmd in details['installation_commands']:
                    synthesis_parts.append(f"```bash\n{cmd}\n```")
                    
            elif step['action'] == 'code_integration' and 'details' in step:
                details = step['details']
                synthesis_parts.append("**Integration Steps**:")
                for i, integration_step in enumerate(details['integration_steps'], 1):
                    synthesis_parts.append(f"{i}. {integration_step}")
                    
            elif step['action'] == 'testing' and 'details' in step:
                details = step['details']
                synthesis_parts.append("**Validation Points**:")
                for point in details['validation_points']:
                    synthesis_parts.append(f"- {point}")
                    
            synthesis_parts.append("")
            
        # Technical Analysis
        synthesis_parts.append("## Technical Analysis")
        
        # Add findings from other agents
        for agent_name, agent_findings in organized_findings.items():
            if agent_name in ["historian", "analyst"] and agent_findings:
                synthesis_parts.append(f"### {agent_name.title()} Findings")
                for finding_type, findings in agent_findings.items():
                    if findings and finding_type != "working_code_extraction":  # Skip duplicate working code
                        finding = findings[0]  # Take first finding of each type
                        synthesis_parts.append(f"- **{finding_type.replace('_', ' ').title()}**: {finding.content[:200]}...")
                        
        synthesis_parts.append("")
        
        # Confidence Assessment
        synthesis_parts.append("## Confidence Assessment")
        synthesis_parts.append(f"**Overall Solution Confidence**: {overall_confidence:.1%}")
        
        if overall_confidence >= 0.8:
            synthesis_parts.append("🟢 **HIGH CONFIDENCE** - Solution should work with minimal issues")
        elif overall_confidence >= 0.6:
            synthesis_parts.append("🟡 **MEDIUM CONFIDENCE** - Solution likely to work with some adjustments")
        else:
            synthesis_parts.append("🔴 **LOW CONFIDENCE** - Solution may require significant debugging")
            
        synthesis_parts.append("")
        synthesis_parts.append("## Next Steps")
        synthesis_parts.append("1. Follow the executable solution steps above")
        synthesis_parts.append("2. Test thoroughly before deploying to production")
        synthesis_parts.append("3. Consider code review if confidence is below 80%")
        
        return "\n".join(synthesis_parts)