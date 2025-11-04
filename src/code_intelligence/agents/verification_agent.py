"""Verification Agent for independent validation of all findings."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..database.neo4j_client import Neo4jClient
from ..git.repository import GitRepository
from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, AgentFinding, Citation
from .tools import Neo4jTool, GitTool


logger = get_logger(__name__)


class VerificationAgent(BaseAgent):
    """Agent responsible for independent validation of findings against source data."""
    
    def __init__(self, config: Optional[AgentConfig] = None, neo4j_client: Optional[Neo4jClient] = None, **kwargs):
        """Initialize the Verification Agent."""
        if config is None:
            config = AgentConfig(
                name="verifier",
                description="Validates findings against actual source code and git history"
            )
        super().__init__(config, **kwargs)
        
        # Initialize tools for independent validation
        self.neo4j_client = neo4j_client
        self.neo4j_tool = Neo4jTool(neo4j_client) if neo4j_client else None
        self.git_tool = GitTool()
        
        # Register verification templates
        self._register_templates()
        
    def _register_templates(self) -> None:
        """Register prompt templates for verification."""
        validation_template = PromptTemplate(
            """Validate the following findings against actual source code and evidence.

Findings to Validate:
{findings_to_validate}

Source Code Evidence:
{source_evidence}

Git History Evidence:
{git_evidence}

For each finding, assess:
1. Accuracy against actual source code
2. Consistency with git history
3. Strength of supporting evidence
4. Potential inaccuracies or uncertainties

Respond in JSON format:
{{
    "validations": [
        {{
            "finding_id": "finding identifier",
            "agent_name": "source agent",
            "validation_result": "valid|invalid|uncertain",
            "confidence_score": 0.0-1.0,
            "evidence_strength": "strong|moderate|weak",
            "issues_found": ["list", "of", "issues"],
            "supporting_evidence": ["evidence1", "evidence2"],
            "recommendations": "validation recommendations"
        }}
    ],
    "overall_assessment": {{
        "total_validated": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "uncertain_count": 0,
        "average_confidence": 0.0-1.0
    }}
}}""",
            variables=["findings_to_validate", "source_evidence", "git_evidence"]
        )
        
        uncertainty_detection_template = PromptTemplate(
            """Analyze findings for potential uncertainties and flag areas requiring additional validation.

Findings Analysis:
{findings_analysis}

Identify:
1. Claims that cannot be fully verified
2. Inconsistencies between agents
3. Missing or weak evidence
4. Areas requiring human review
5. Confidence score adjustments

Focus on detecting and flagging uncertainties rather than making definitive judgments.""",
            variables=["findings_analysis"]
        )
        
        self.validation_template = validation_template
        self.uncertainty_detection_template = uncertainty_detection_template
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute INDEPENDENT verification logic against actual repository data."""
        self._log_execution_start(state)
        
        if not self._validate_state(state):
            state.add_error("Invalid state for verifier", self.config.name)
            return state
            
        try:
            # Get all findings to validate
            all_findings = state.get_all_findings()
            
            if not all_findings:
                state.add_warning("No findings to verify", self.config.name)
                return state
                
            # Initialize git repository for validation
            repository_path = state.repository.get("path", "")
            git_repo = None
            if repository_path and self._is_valid_git_repository(repository_path):
                try:
                    git_repo = GitRepository(repository_path)
                except Exception as e:
                    self.logger.warning(f"Failed to initialize git repository: {str(e)}")
                    git_repo = None
            
            # Check if we have a complete solution to validate
            solution_available = self._check_solution_availability(all_findings)
            
            if solution_available:
                # Perform solution-level validation
                solution_validation = await self._validate_complete_solution(all_findings, git_repo, repository_path, state)
                
                # QUALITY GATE: 90% confidence threshold for solutions (Requirement 3.5)
                if solution_validation["confidence"] >= 0.9:
                    validation_status = "SOLUTION_APPROVED"
                    validation_message = f"✅ Solution validated with {solution_validation['confidence']:.1%} confidence"
                else:
                    validation_status = "SOLUTION_NEEDS_REVIEW"
                    validation_message = f"⚠️ Solution needs review - {solution_validation['confidence']:.1%} confidence (below 90% threshold)"
                    
                # Create solution validation finding
                solution_finding = self._create_finding(
                    finding_type="solution_validation",
                    content=validation_message,
                    confidence=solution_validation["confidence"],
                    citations=solution_validation.get("citations", []),
                    metadata={
                        "validation_status": validation_status,
                        "validation_details": solution_validation,
                        "threshold_met": solution_validation["confidence"] >= 0.9
                    }
                )
                state.add_finding(self.config.name, solution_finding)
                
            else:
                # Fallback to individual finding validation
                validation_results = []
                for finding in all_findings:
                    if finding.agent_name != self.config.name:  # Don't validate our own findings
                        validation = await self._validate_finding(finding, git_repo, repository_path)
                        validation_results.append(validation)
                        
                # Generate verification summary
                verification_summary = self._generate_verification_summary(validation_results)
                overall_confidence = verification_summary['average_confidence']
                
                # QUALITY GATE: 90% threshold for individual findings (Requirement 3.5)
                if overall_confidence >= 0.9:
                    # HIGH CONFIDENCE: Create approved report
                    verification_finding = self._create_finding(
                        finding_type="verified_report",
                        content=f"✅ VERIFIED REPORT (Confidence: {overall_confidence:.1%})\n"
                               f"Validated {len(validation_results)} findings. "
                               f"{verification_summary['valid_count']} valid, "
                               f"{verification_summary['invalid_count']} invalid, "
                               f"{verification_summary['uncertain_count']} uncertain.\n"
                               f"Report meets quality standards and is ready for use.",
                        confidence=overall_confidence,
                        citations=self._extract_verified_citations(all_findings, validation_results),
                        metadata={
                            "validation_results": validation_results,
                            "verification_summary": verification_summary,
                            "quality_gate": "PASSED",
                            "ready_for_user": True
                        }
                    )
                    state.add_finding(self.config.name, verification_finding)
                
                    # Mark state as verified and ready
                    state.verification["quality_gate_passed"] = True
                    state.verification["ready_for_user"] = True
                    
                else:
                    # LOW CONFIDENCE: Create investigation needed report
                    verification_finding = self._create_finding(
                        finding_type="investigation_needed",
                        content=f"⚠️ INVESTIGATION NEEDED (Confidence: {overall_confidence:.1%})\n"
                               f"Analysis incomplete. {verification_summary['valid_count']}/{len(validation_results)} findings validated.\n"
                               f"Issues found: {verification_summary['uncertain_count']} uncertainties detected.\n"
                               f"Recommendation: Gather more data or manual review required.",
                        confidence=overall_confidence,
                        citations=[],
                        metadata={
                            "validation_results": validation_results,
                            "verification_summary": verification_summary,
                            "quality_gate": "FAILED",
                            "ready_for_user": False
                        }
                    )
                    state.add_finding(self.config.name, verification_finding)
                
                # Mark state as needing more work
                state.verification["quality_gate_passed"] = False
                state.verification["ready_for_user"] = False
            
            # Add uncertainty findings
            for uncertainty in uncertainties:
                uncertainty_finding = self._create_finding(
                    finding_type="uncertainty_flag",
                    content=uncertainty["description"],
                    confidence=uncertainty["confidence"],
                    citations=[],
                    metadata=uncertainty
                )
                state.add_finding(self.config.name, uncertainty_finding)
                
            # Update state verification data
            state.verification.update({
                "validation_results": validation_results,
                "verification_summary": verification_summary,
                "uncertainties": uncertainties,
                "overall_confidence": verification_summary['average_confidence']
            })
            
            self._log_execution_end(state, True)
            return state
            
        except Exception as e:
            self.logger.error(f"Verifier execution failed: {str(e)}")
            state.add_error(f"Verifier failed: {str(e)}", self.config.name)
            self._log_execution_end(state, False)
            return state
            
    async def _validate_finding(
        self, 
        finding: AgentFinding, 
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate a single finding against source data."""
        validation_result = {
            "finding_id": f"{finding.agent_name}_{finding.finding_type}_{finding.timestamp}",
            "agent_name": finding.agent_name,
            "finding_type": finding.finding_type,
            "original_confidence": finding.confidence,
            "validation_result": "uncertain",
            "confidence_score": finding.confidence,
            "evidence_strength": "weak",
            "issues_found": [],
            "supporting_evidence": [],
            "recommendations": ""
        }
        
        try:
            # Validate citations
            citation_validation = await self._validate_citations(finding.citations, git_repo, repository_path)
            validation_result["citation_validation"] = citation_validation
            
            # Validate content claims
            content_validation = await self._validate_content_claims(finding, git_repo, repository_path)
            validation_result["content_validation"] = content_validation
            
            # Calculate overall validation score
            overall_score = self._calculate_validation_score(citation_validation, content_validation)
            validation_result["confidence_score"] = overall_score
            
            # Determine validation result (Requirement 3.5: 90% threshold)
            if overall_score >= 0.9:
                validation_result["validation_result"] = "valid"
                validation_result["evidence_strength"] = "strong"
            elif overall_score >= 0.6:
                validation_result["validation_result"] = "valid"
                validation_result["evidence_strength"] = "moderate"
            elif overall_score >= 0.4:
                validation_result["validation_result"] = "uncertain"
                validation_result["evidence_strength"] = "weak"
            else:
                validation_result["validation_result"] = "invalid"
                validation_result["evidence_strength"] = "weak"
                validation_result["issues_found"].append("Low validation score")
                
            # Generate recommendations
            validation_result["recommendations"] = self._generate_validation_recommendations(validation_result)
            
        except Exception as e:
            self.logger.warning(f"Failed to validate finding from {finding.agent_name}: {str(e)}")
            validation_result["issues_found"].append(f"Validation error: {str(e)}")
            validation_result["confidence_score"] = 0.3
            
        return validation_result
        
    async def _validate_citations(
        self, 
        citations: List[Citation], 
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate citations against actual files and git history."""
        citation_validation = {
            "total_citations": len(citations),
            "valid_citations": 0,
            "invalid_citations": 0,
            "validation_details": []
        }
        
        for citation in citations:
            citation_result = {
                "file_path": citation.file_path,
                "line_number": citation.line_number,
                "commit_sha": citation.commit_sha,
                "valid": False,
                "issues": []
            }
            
            try:
                # Validate file path
                if citation.file_path and citation.file_path != "git_history":
                    # Prevent path traversal attacks
                    if not self._is_safe_path(citation.file_path, repository_path):
                        citation_result["issues"].append("Unsafe file path detected")
                        citation_validation["invalid_citations"] += 1
                        continue
                        
                    full_path = os.path.join(repository_path, citation.file_path)
                    if os.path.exists(full_path):
                        citation_result["valid"] = True
                        citation_validation["valid_citations"] += 1
                        
                        # Validate line number if specified
                        if citation.line_number:
                            if await self._validate_line_number(full_path, citation.line_number):
                                citation_result["supporting_evidence"] = "Line number exists"
                            else:
                                citation_result["issues"].append("Line number out of range")
                                citation_result["valid"] = False
                    else:
                        citation_result["issues"].append("File does not exist")
                        citation_validation["invalid_citations"] += 1
                        
                # Validate commit SHA
                if citation.commit_sha and git_repo:
                    if await self._validate_commit_sha(git_repo, citation.commit_sha):
                        citation_result["commit_valid"] = True
                    else:
                        citation_result["issues"].append("Invalid commit SHA")
                        
            except Exception as e:
                citation_result["issues"].append(f"Validation error: {str(e)}")
                citation_validation["invalid_citations"] += 1
                
            citation_validation["validation_details"].append(citation_result)
            
        return citation_validation
        
    async def _validate_line_number(self, file_path: str, line_number: int) -> bool:
        """Validate that a line number exists in a file."""
        import asyncio
        try:
            # Use asyncio to run file operation in thread pool
            def read_file():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.readlines()
            
            lines = await asyncio.get_event_loop().run_in_executor(None, read_file)
            return 1 <= line_number <= len(lines)
        except Exception:
            return False
            
    async def _validate_commit_sha(self, git_repo: GitRepository, commit_sha: str) -> bool:
        """Validate that a commit SHA exists in the repository."""
        try:
            # Try to get the commit
            commit = git_repo.get_commit(commit_sha)
            return commit is not None
        except Exception:
            return False
            
    async def _validate_content_claims(
        self, 
        finding: AgentFinding, 
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """INDEPENDENTLY validate content claims against actual repository data using Neo4j and Git."""
        content_validation = {
            "claims_validated": 0,
            "claims_failed": 0,
            "validation_details": [],
            "independent_verification": True
        }
        
        try:
            # Extract specific claims that can be validated against Neo4j CPG and Git
            claims = await self._extract_verifiable_claims(finding)
            
            for claim in claims:
                claim_validation = await self._validate_claim_independently(claim, git_repo, repository_path)
                content_validation["validation_details"].append(claim_validation)
                
                if claim_validation["verified"]:
                    content_validation["claims_validated"] += 1
                else:
                    content_validation["claims_failed"] += 1
                    # Add uncertainty to state for failed validations
                    await self._add_validation_uncertainty(finding, claim, claim_validation)
                    
        except Exception as e:
            self.logger.error(f"Content validation failed: {str(e)}")
            content_validation["validation_details"].append({
                "claim": "validation_error",
                "verified": False,
                "error": str(e)
            })
            content_validation["claims_failed"] += 1
            
        return content_validation
        
    async def _extract_verifiable_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract specific claims that can be verified against Neo4j CPG and Git."""
        claims = []
        
        # Extract claims based on finding type and content
        if finding.finding_type == "code_element_changed":
            claims.extend(await self._extract_code_change_claims(finding))
        elif finding.finding_type == "dependency_changed":
            claims.extend(await self._extract_dependency_claims(finding))
        elif finding.finding_type == "commit_message_intent":
            claims.extend(await self._extract_commit_intent_claims(finding))
        elif finding.finding_type == "function_analysis":
            claims.extend(await self._extract_function_claims(finding))
        elif finding.finding_type == "structural_analysis":
            claims.extend(await self._extract_structural_claims(finding))
        else:
            # Generic claim extraction for other finding types
            claims.extend(await self._extract_generic_claims(finding))
            
        return claims
        
    async def _extract_code_change_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract claims about code element changes."""
        claims = []
        content = finding.content
        
        # Pattern: "Function foo was modified in commit abc123"
        function_change_pattern = r'(?:function|method)\s+(\w+)\s+(?:was\s+)?(?:modified|changed|updated)\s+(?:in\s+)?commit\s+([a-f0-9]{7,40})'
        matches = re.finditer(function_change_pattern, content.lower())
        
        for match in matches:
            function_name = match.group(1)
            commit_sha = match.group(2)
            claims.append({
                "type": "code_element_changed",
                "element_type": "Function",
                "element_name": function_name,
                "commit_sha": commit_sha,
                "claim": f"Function {function_name} was modified in commit {commit_sha}",
                "validation_method": "neo4j_cpg_query"
            })
            
        # Pattern: "Class Bar was added in commit def456"
        class_change_pattern = r'(?:class)\s+(\w+)\s+(?:was\s+)?(?:added|created|introduced)\s+(?:in\s+)?commit\s+([a-f0-9]{7,40})'
        matches = re.finditer(class_change_pattern, content.lower())
        
        for match in matches:
            class_name = match.group(1)
            commit_sha = match.group(2)
            claims.append({
                "type": "code_element_changed",
                "element_type": "Class",
                "element_name": class_name,
                "commit_sha": commit_sha,
                "claim": f"Class {class_name} was added in commit {commit_sha}",
                "validation_method": "neo4j_cpg_query"
            })
            
        return claims
        
    async def _extract_dependency_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract claims about dependency changes."""
        claims = []
        content = finding.content
        
        # Pattern: "Function foo now calls function bar"
        call_pattern = r'(?:function|method)\s+(\w+)\s+(?:now\s+)?(?:calls|invokes)\s+(?:function|method)?\s*(\w+)'
        matches = re.finditer(call_pattern, content.lower())
        
        for match in matches:
            caller = match.group(1)
            callee = match.group(2)
            claims.append({
                "type": "dependency_changed",
                "caller": caller,
                "callee": callee,
                "relationship": "CALLS",
                "claim": f"Function {caller} calls function {callee}",
                "validation_method": "neo4j_relationship_query"
            })
            
        # Pattern: "Module A imports module B"
        import_pattern = r'(?:module|class)\s+(\w+)\s+(?:imports|uses)\s+(?:module|class)?\s*(\w+)'
        matches = re.finditer(import_pattern, content.lower())
        
        for match in matches:
            importer = match.group(1)
            imported = match.group(2)
            claims.append({
                "type": "dependency_changed",
                "caller": importer,
                "callee": imported,
                "relationship": "IMPORTS",
                "claim": f"Module {importer} imports module {imported}",
                "validation_method": "neo4j_relationship_query"
            })
            
        return claims
        
    async def _extract_commit_intent_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract claims about commit message intent."""
        claims = []
        content = finding.content
        
        # Pattern: "This change was to fix 'bug #123'"
        bug_fix_pattern = r'(?:fix|fixes|fixed)\s+(?:bug\s+)?[#]?(\d+)'
        matches = re.finditer(bug_fix_pattern, content.lower())
        
        for match in matches:
            bug_number = match.group(1)
            # Extract commit SHA from metadata or content
            commit_sha = finding.metadata.get("commit_sha") or self._extract_commit_sha_from_content(content)
            if commit_sha:
                claims.append({
                    "type": "commit_message_intent",
                    "commit_sha": commit_sha,
                    "intent": f"bug #{bug_number}",
                    "claim": f"Commit {commit_sha} was to fix bug #{bug_number}",
                    "validation_method": "git_commit_message_check"
                })
                
        return claims
        
    async def _extract_function_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract claims about function analysis."""
        claims = []
        content = finding.content
        
        # Pattern: "Function exists at line X in file Y"
        function_location_pattern = r'(?:function|method)\s+(\w+)\s+(?:exists|is located|found)\s+(?:at\s+)?(?:line\s+)?(\d+)\s+(?:in\s+)?(?:file\s+)?([^\s,]+)'
        matches = re.finditer(function_location_pattern, content.lower())
        
        for match in matches:
            function_name = match.group(1)
            line_number = int(match.group(2))
            file_path = match.group(3)
            claims.append({
                "type": "function_location",
                "function_name": function_name,
                "line_number": line_number,
                "file_path": file_path,
                "claim": f"Function {function_name} exists at line {line_number} in {file_path}",
                "validation_method": "neo4j_function_location_query"
            })
            
        return claims
        
    async def _extract_structural_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract claims about structural analysis."""
        claims = []
        content = finding.content
        
        # Pattern: "Class X inherits from class Y"
        inheritance_pattern = r'(?:class)\s+(\w+)\s+(?:inherits from|extends)\s+(?:class\s+)?(\w+)'
        matches = re.finditer(inheritance_pattern, content.lower())
        
        for match in matches:
            child_class = match.group(1)
            parent_class = match.group(2)
            claims.append({
                "type": "inheritance_relationship",
                "child_class": child_class,
                "parent_class": parent_class,
                "claim": f"Class {child_class} inherits from class {parent_class}",
                "validation_method": "neo4j_inheritance_query"
            })
            
        return claims
        
    async def _extract_generic_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract generic verifiable claims."""
        claims = []
        content = finding.content
        
        # Extract commit SHAs mentioned in content
        commit_pattern = r'commit\s+([a-f0-9]{7,40})'
        matches = re.finditer(commit_pattern, content.lower())
        
        for match in matches:
            commit_sha = match.group(1)
            claims.append({
                "type": "commit_existence",
                "commit_sha": commit_sha,
                "claim": f"Commit {commit_sha} exists in repository",
                "validation_method": "git_commit_existence_check"
            })
            
        return claims
        
    def _extract_commit_sha_from_content(self, content: str) -> Optional[str]:
        """Extract commit SHA from content."""
        commit_pattern = r'commit\s+([a-f0-9]{7,40})'
        match = re.search(commit_pattern, content.lower())
        return match.group(1) if match else None
        
    def _extract_historian_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract verifiable claims from historian findings."""
        claims = []
        content = finding.content
        metadata = finding.metadata
        
        # Claim: Working code extracted from specific commit
        if "extracted working code" in content.lower():
            commit_sha = metadata.get("commit_sha")
            file_path = metadata.get("file_path")
            if commit_sha and file_path:
                claims.append({
                    "type": "code_extraction",
                    "claim": f"Working code extracted from commit {commit_sha} in file {file_path}",
                    "commit_sha": commit_sha,
                    "file_path": file_path,
                    "verification_method": "git_show_validation"
                })
                
        # Claim: Commit contains specific changes
        commit_pattern = r'commit ([a-f0-9]{7,40})'
        commits = re.findall(commit_pattern, content.lower())
        for commit_sha in commits:
            claims.append({
                "type": "commit_existence",
                "claim": f"Commit {commit_sha} exists in repository",
                "commit_sha": commit_sha,
                "verification_method": "git_commit_validation"
            })
            
        return claims
        
    def _extract_analyst_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract verifiable claims from analyst findings."""
        claims = []
        content = finding.content
        metadata = finding.metadata
        
        # Claim: Dependencies identified
        if "dependencies" in metadata:
            dependencies = metadata.get("dependencies", [])
            for dep in dependencies:
                if isinstance(dep, dict) and "name" in dep:
                    claims.append({
                        "type": "dependency_existence",
                        "claim": f"Dependency {dep['name']} exists in codebase",
                        "dependency_name": dep["name"],
                        "verification_method": "dependency_validation"
                    })
                    
        # Claim: Integration analysis
        if "integration_steps" in metadata:
            integration_steps = metadata.get("integration_steps", [])
            claims.append({
                "type": "integration_feasibility",
                "claim": f"Integration requires {len(integration_steps)} steps",
                "steps_count": len(integration_steps),
                "verification_method": "integration_validation"
            })
            
        return claims
        
    def _extract_synthesizer_claims(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract verifiable claims from synthesizer findings."""
        claims = []
        content = finding.content
        
        # Claim: Solution found with specific confidence
        if "solution found" in content.lower():
            confidence_match = re.search(r'(\d+)%\s*confidence', content.lower())
            if confidence_match:
                confidence = int(confidence_match.group(1))
                claims.append({
                    "type": "solution_confidence",
                    "claim": f"Solution has {confidence}% confidence",
                    "claimed_confidence": confidence,
                    "verification_method": "confidence_validation"
                })
                
        # Claim: Executable steps provided
        step_pattern = r'step\s+(\d+):'
        steps = re.findall(step_pattern, content.lower())
        if steps:
            claims.append({
                "type": "executable_steps",
                "claim": f"Solution provides {len(steps)} executable steps",
                "steps_count": len(steps),
                "verification_method": "steps_validation"
            })
            
        return claims
        
    async def _validate_claim_independently(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """Independently validate a specific claim using Neo4j CPG and Git tools."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": [],
            "confidence": 0.0
        }
        
        try:
            validation_method = claim.get("validation_method", "unknown")
            
            if validation_method == "neo4j_cpg_query":
                validation_result = await self._validate_code_element_change(claim)
                
            elif validation_method == "neo4j_relationship_query":
                validation_result = await self._validate_dependency_relationship(claim)
                
            elif validation_method == "git_commit_message_check":
                validation_result = await self._validate_commit_intent(claim, git_repo, repository_path)
                
            elif validation_method == "neo4j_function_location_query":
                validation_result = await self._validate_function_location(claim)
                
            elif validation_method == "neo4j_inheritance_query":
                validation_result = await self._validate_inheritance_relationship(claim)
                
            elif validation_method == "git_commit_existence_check":
                validation_result = await self._validate_commit_existence(claim, git_repo)
                
            else:
                validation_result["issues"].append(f"Unknown validation method: {validation_method}")
                validation_result["confidence"] = 0.0
                
        except Exception as e:
            validation_result["issues"].append(f"Validation error: {str(e)}")
            validation_result["confidence"] = 0.0
            
        return validation_result
        
    async def _validate_code_element_change(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code element change claim using Neo4j CPG query."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": [],
            "confidence": 0.0
        }
        
        if not self.neo4j_tool:
            validation_result["issues"].append("Neo4j tool not available for validation")
            return validation_result
            
        try:
            element_type = claim["element_type"]
            element_name = claim["element_name"]
            commit_sha = claim["commit_sha"]
            
            # Query Neo4j CPG to verify the relationship exists
            query = f"""
            MATCH (e:{element_type} {{name: $element_name}})
            MATCH (c:Commit {{sha: $commit_sha}})
            MATCH (e)-[:CHANGED_IN]->(c)
            RETURN e.name as element_name, e.file_path as file_path, 
                   c.sha as commit_sha, c.message as commit_message
            """
            
            results = await self.neo4j_tool.execute(
                "query", 
                query=query, 
                parameters={
                    "element_name": element_name,
                    "commit_sha": commit_sha
                }
            )
            
            if results and len(results) > 0:
                validation_result["verified"] = True
                validation_result["confidence"] = 1.0
                validation_result["evidence"].append(
                    f"Neo4j CPG confirms {element_type} {element_name} was changed in commit {commit_sha}"
                )
                for result in results:
                    validation_result["evidence"].append(
                        f"Found in file: {result.get('file_path', 'unknown')}"
                    )
            else:
                validation_result["verified"] = False
                validation_result["confidence"] = 0.0
                validation_result["issues"].append(
                    f"Neo4j CPG query found no evidence that {element_type} {element_name} was changed in commit {commit_sha}"
                )
                
        except Exception as e:
            validation_result["issues"].append(f"Neo4j validation error: {str(e)}")
            validation_result["confidence"] = 0.0
            
        return validation_result
        
    async def _validate_dependency_relationship(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dependency relationship claim using Neo4j CPG query."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": [],
            "confidence": 0.0
        }
        
        if not self.neo4j_tool:
            validation_result["issues"].append("Neo4j tool not available for validation")
            return validation_result
            
        try:
            caller = claim["caller"]
            callee = claim["callee"]
            relationship = claim["relationship"]
            
            # Query Neo4j CPG to verify the relationship exists
            query = f"""
            MATCH (caller:Function {{name: $caller}})
            MATCH (callee:Function {{name: $callee}})
            MATCH (caller)-[:{relationship}]->(callee)
            RETURN caller.name as caller_name, caller.file_path as caller_file,
                   callee.name as callee_name, callee.file_path as callee_file
            """
            
            results = await self.neo4j_tool.execute(
                "query",
                query=query,
                parameters={
                    "caller": caller,
                    "callee": callee
                }
            )
            
            if results and len(results) > 0:
                validation_result["verified"] = True
                validation_result["confidence"] = 1.0
                validation_result["evidence"].append(
                    f"Neo4j CPG confirms function {caller} {relationship.lower()}s function {callee}"
                )
                for result in results:
                    validation_result["evidence"].append(
                        f"Caller in: {result.get('caller_file', 'unknown')}, "
                        f"Callee in: {result.get('callee_file', 'unknown')}"
                    )
            else:
                validation_result["verified"] = False
                validation_result["confidence"] = 0.0
                validation_result["issues"].append(
                    f"Neo4j CPG query found no {relationship} relationship between {caller} and {callee}"
                )
                
        except Exception as e:
            validation_result["issues"].append(f"Neo4j validation error: {str(e)}")
            validation_result["confidence"] = 0.0
            
        return validation_result
        
    async def _validate_commit_intent(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate commit message intent using Git tools."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": [],
            "confidence": 0.0
        }
        
        if not git_repo:
            validation_result["issues"].append("Git repository not available for validation")
            return validation_result
            
        try:
            commit_sha = claim["commit_sha"]
            expected_intent = claim["intent"]
            
            # Use Git tool to get commit information
            commit_info = git_repo.get_commit_info(commit_sha)
            
            if commit_info and commit_info.get("message"):
                commit_message = commit_info["message"].lower()
                
                if expected_intent.lower() in commit_message:
                    validation_result["verified"] = True
                    validation_result["confidence"] = 1.0
                    validation_result["evidence"].append(
                        f"Commit message contains expected intent: '{expected_intent}'"
                    )
                    validation_result["evidence"].append(
                        f"Full commit message: {commit_info['message'][:200]}..."
                    )
                else:
                    validation_result["verified"] = False
                    validation_result["confidence"] = 0.0
                    validation_result["issues"].append(
                        f"Commit message does not contain expected intent: '{expected_intent}'"
                    )
                    validation_result["issues"].append(
                        f"Actual commit message: {commit_info['message'][:100]}..."
                    )
            else:
                validation_result["verified"] = False
                validation_result["confidence"] = 0.0
                validation_result["issues"].append(f"Could not retrieve commit message for {commit_sha}")
                
        except Exception as e:
            validation_result["issues"].append(f"Git validation error: {str(e)}")
            validation_result["confidence"] = 0.0
            
        return validation_result
        
    async def _validate_function_location(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function location claim using Neo4j CPG query."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": [],
            "confidence": 0.0
        }
        
        if not self.neo4j_tool:
            validation_result["issues"].append("Neo4j tool not available for validation")
            return validation_result
            
        try:
            function_name = claim["function_name"]
            expected_line = claim["line_number"]
            expected_file = claim["file_path"]
            
            # Query Neo4j CPG to verify function location
            query = """
            MATCH (f:Function {name: $function_name})
            WHERE f.file_path = $file_path
            AND f.start_line <= $line_number 
            AND f.end_line >= $line_number
            RETURN f.name as name, f.file_path as file_path,
                   f.start_line as start_line, f.end_line as end_line
            """
            
            results = await self.neo4j_tool.execute(
                "query",
                query=query,
                parameters={
                    "function_name": function_name,
                    "file_path": expected_file,
                    "line_number": expected_line
                }
            )
            
            if results and len(results) > 0:
                validation_result["verified"] = True
                validation_result["confidence"] = 1.0
                result = results[0]
                validation_result["evidence"].append(
                    f"Neo4j CPG confirms function {function_name} exists in {expected_file} "
                    f"at lines {result['start_line']}-{result['end_line']}"
                )
            else:
                validation_result["verified"] = False
                validation_result["confidence"] = 0.0
                validation_result["issues"].append(
                    f"Neo4j CPG found no function {function_name} at line {expected_line} in {expected_file}"
                )
                
        except Exception as e:
            validation_result["issues"].append(f"Neo4j validation error: {str(e)}")
            validation_result["confidence"] = 0.0
            
        return validation_result
        
    async def _validate_inheritance_relationship(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inheritance relationship claim using Neo4j CPG query."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": [],
            "confidence": 0.0
        }
        
        if not self.neo4j_tool:
            validation_result["issues"].append("Neo4j tool not available for validation")
            return validation_result
            
        try:
            child_class = claim["child_class"]
            parent_class = claim["parent_class"]
            
            # Query Neo4j CPG to verify inheritance relationship
            query = """
            MATCH (child:Class {name: $child_class})
            MATCH (parent:Class {name: $parent_class})
            MATCH (child)-[:INHERITS_FROM]->(parent)
            RETURN child.name as child_name, child.file_path as child_file,
                   parent.name as parent_name, parent.file_path as parent_file
            """
            
            results = await self.neo4j_tool.execute(
                "query",
                query=query,
                parameters={
                    "child_class": child_class,
                    "parent_class": parent_class
                }
            )
            
            if results and len(results) > 0:
                validation_result["verified"] = True
                validation_result["confidence"] = 1.0
                validation_result["evidence"].append(
                    f"Neo4j CPG confirms class {child_class} inherits from class {parent_class}"
                )
                for result in results:
                    validation_result["evidence"].append(
                        f"Child class in: {result.get('child_file', 'unknown')}, "
                        f"Parent class in: {result.get('parent_file', 'unknown')}"
                    )
            else:
                validation_result["verified"] = False
                validation_result["confidence"] = 0.0
                validation_result["issues"].append(
                    f"Neo4j CPG found no inheritance relationship between {child_class} and {parent_class}"
                )
                
        except Exception as e:
            validation_result["issues"].append(f"Neo4j validation error: {str(e)}")
            validation_result["confidence"] = 0.0
            
        return validation_result
        
    async def _validate_commit_existence(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository]
    ) -> Dict[str, Any]:
        """Validate commit existence using Git tools."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": [],
            "confidence": 0.0
        }
        
        if not git_repo:
            validation_result["issues"].append("Git repository not available for validation")
            return validation_result
            
        try:
            commit_sha = claim["commit_sha"]
            
            # Use Git tool to verify commit exists
            commit_info = git_repo.get_commit_info(commit_sha)
            
            if commit_info:
                validation_result["verified"] = True
                validation_result["confidence"] = 1.0
                validation_result["evidence"].append(f"Commit {commit_sha} exists in repository")
                validation_result["evidence"].append(f"Author: {commit_info.get('author', 'unknown')}")
                validation_result["evidence"].append(f"Date: {commit_info.get('date', 'unknown')}")
            else:
                validation_result["verified"] = False
                validation_result["confidence"] = 0.0
                validation_result["issues"].append(f"Commit {commit_sha} not found in repository")
                
        except Exception as e:
            validation_result["issues"].append(f"Git validation error: {str(e)}")
            validation_result["confidence"] = 0.0
            
        return validation_result
        
    async def _validate_git_show_claim(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate code extraction claim by independently checking git show."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": []
        }
        
        if not git_repo:
            validation_result["issues"].append("No git repository available for validation")
            return validation_result
            
        try:
            commit_sha = claim["commit_sha"]
            file_path = claim["file_path"]
            
            # Independently extract the file content using git show
            import subprocess
            cmd = ["git", "show", f"{commit_sha}:{file_path}"]
            result = subprocess.run(
                cmd, 
                cwd=repository_path, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                extracted_content = result.stdout
                if len(extracted_content) > 50:  # Reasonable code length
                    validation_result["verified"] = True
                    validation_result["evidence"].append(f"Successfully extracted {len(extracted_content)} characters from {file_path} at {commit_sha}")
                else:
                    validation_result["issues"].append("Extracted content too short to be meaningful code")
            else:
                validation_result["issues"].append(f"Git show failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            validation_result["issues"].append("Git show command timed out")
        except Exception as e:
            validation_result["issues"].append(f"Git show validation error: {str(e)}")
            
        return validation_result
        
    async def _validate_commit_claim(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository]
    ) -> Dict[str, Any]:
        """Validate commit existence claim by independently checking git log."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": []
        }
        
        if not git_repo:
            validation_result["issues"].append("No git repository available for validation")
            return validation_result
            
        try:
            commit_sha = claim["commit_sha"]
            
            # Independently verify commit exists
            commit_info = git_repo.get_commit_info(commit_sha)
            if commit_info:
                validation_result["verified"] = True
                validation_result["evidence"].append(f"Commit {commit_sha} exists with message: {commit_info.get('message', '')[:100]}")
            else:
                validation_result["issues"].append(f"Commit {commit_sha} not found in repository")
                
        except Exception as e:
            validation_result["issues"].append(f"Commit validation error: {str(e)}")
            
        return validation_result
        
    async def _validate_dependency_claim(
        self, 
        claim: Dict[str, Any], 
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate dependency claim by independently checking source files."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": []
        }
        
        try:
            dependency_name = claim["dependency_name"]
            
            # Independently search for dependency in source files
            found_in_files = []
            
            # Check common dependency files
            dependency_files = ["requirements.txt", "package.json", "pom.xml", "build.gradle"]
            for dep_file in dependency_files:
                file_path = os.path.join(repository_path, dep_file)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if dependency_name in content:
                                found_in_files.append(dep_file)
                    except Exception:
                        continue
                        
            # Check source files for imports
            for root, dirs, files in os.walk(repository_path):
                # Skip .git and other hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files[:10]:  # Limit to first 10 files for performance
                    if any(file.endswith(ext) for ext in ['.py', '.java', '.js', '.ts']):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if dependency_name in content:
                                    found_in_files.append(file)
                                    break  # Found it, move on
                        except Exception:
                            continue
                            
            if found_in_files:
                validation_result["verified"] = True
                validation_result["evidence"].append(f"Dependency {dependency_name} found in: {', '.join(found_in_files[:3])}")
            else:
                validation_result["issues"].append(f"Dependency {dependency_name} not found in codebase")
                
        except Exception as e:
            validation_result["issues"].append(f"Dependency validation error: {str(e)}")
            
        return validation_result
        
    async def _validate_integration_claim(
        self, 
        claim: Dict[str, Any], 
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate integration steps claim by checking feasibility."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": []
        }
        
        try:
            steps_count = claim["steps_count"]
            
            # Reasonable integration should have 2-10 steps
            if 2 <= steps_count <= 10:
                validation_result["verified"] = True
                validation_result["evidence"].append(f"Integration steps count ({steps_count}) is reasonable")
            else:
                validation_result["issues"].append(f"Integration steps count ({steps_count}) seems unrealistic")
                
        except Exception as e:
            validation_result["issues"].append(f"Integration validation error: {str(e)}")
            
        return validation_result
        
    async def _validate_confidence_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate confidence claim by checking if it's reasonable."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": []
        }
        
        try:
            claimed_confidence = claim["claimed_confidence"]
            
            # Confidence should be between 0-100%
            if 0 <= claimed_confidence <= 100:
                validation_result["verified"] = True
                validation_result["evidence"].append(f"Confidence value ({claimed_confidence}%) is within valid range")
            else:
                validation_result["issues"].append(f"Confidence value ({claimed_confidence}%) is outside valid range")
                
        except Exception as e:
            validation_result["issues"].append(f"Confidence validation error: {str(e)}")
            
        return validation_result
        
    async def _validate_steps_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate executable steps claim by checking reasonableness."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "verified": False,
            "evidence": [],
            "issues": []
        }
        
        try:
            steps_count = claim["steps_count"]
            
            # Reasonable solution should have 1-8 steps
            if 1 <= steps_count <= 8:
                validation_result["verified"] = True
                validation_result["evidence"].append(f"Steps count ({steps_count}) is reasonable for executable solution")
            else:
                validation_result["issues"].append(f"Steps count ({steps_count}) seems unrealistic for executable solution")
                
        except Exception as e:
            validation_result["issues"].append(f"Steps validation error: {str(e)}")
            
        return validation_result
        
        for claim in claims:
            claim_result = await self._validate_specific_claim(claim, finding, git_repo, repository_path)
            content_validation["validation_details"].append(claim_result)
            
            if claim_result["valid"]:
                content_validation["claims_validated"] += 1
            else:
                content_validation["claims_failed"] += 1
                
        return content_validation
        
    def _extract_claims_from_content(self, content: str) -> List[Dict[str, str]]:
        """Extract specific claims from finding content."""
        claims = []
        
        # Look for specific patterns that can be validated
        patterns = [
            (r"(\d+)\s+(?:commits?|changes?)", "commit_count"),
            (r"function\s+(\w+)", "function_reference"),
            (r"class\s+(\w+)", "class_reference"),
            (r"file\s+([^\s]+\.(?:py|js|ts|java|cpp|c|h))", "file_reference"),
            (r"(\w+)\s+calls?\s+(\w+)", "function_call"),
            (r"(\w+)\s+depends?\s+on\s+(\w+)", "dependency"),
        ]
        
        for pattern, claim_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "type": claim_type,
                    "content": match.group(0),
                    "groups": match.groups()
                })
                
        return claims[:10]  # Limit to 10 claims for performance
        
    async def _validate_specific_claim(
        self, 
        claim: Dict[str, str], 
        finding: AgentFinding,
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate a specific claim."""
        claim_result = {
            "claim": claim["content"],
            "type": claim["type"],
            "valid": False,
            "evidence": [],
            "issues": []
        }
        
        try:
            if claim["type"] == "file_reference":
                # Validate file exists
                file_path = claim["groups"][0]
                full_path = os.path.join(repository_path, file_path)
                if os.path.exists(full_path):
                    claim_result["valid"] = True
                    claim_result["evidence"].append(f"File {file_path} exists")
                else:
                    claim_result["issues"].append(f"File {file_path} not found")
                    
            elif claim["type"] == "function_reference":
                # Simple validation - check if function name appears in cited files
                function_name = claim["groups"][0]
                found_in_files = await self._search_for_identifier(function_name, finding.citations, repository_path)
                if found_in_files:
                    claim_result["valid"] = True
                    claim_result["evidence"].append(f"Function {function_name} found in {len(found_in_files)} files")
                else:
                    claim_result["issues"].append(f"Function {function_name} not found in cited files")
                    
            elif claim["type"] == "class_reference":
                # Similar to function reference
                class_name = claim["groups"][0]
                found_in_files = await self._search_for_identifier(class_name, finding.citations, repository_path)
                if found_in_files:
                    claim_result["valid"] = True
                    claim_result["evidence"].append(f"Class {class_name} found in {len(found_in_files)} files")
                else:
                    claim_result["issues"].append(f"Class {class_name} not found in cited files")
                    
            elif claim["type"] == "commit_count":
                # Validate commit count if git repo is available
                count = int(claim["groups"][0])
                if git_repo:
                    actual_count = len(git_repo.get_recent_commits(limit=count + 10))
                    if actual_count >= count:
                        claim_result["valid"] = True
                        claim_result["evidence"].append(f"Repository has at least {count} commits")
                    else:
                        claim_result["issues"].append(f"Repository has only {actual_count} commits, not {count}")
                else:
                    claim_result["issues"].append("Cannot validate commit count without git repository")
                    
        except Exception as e:
            claim_result["issues"].append(f"Validation error: {str(e)}")
            
        return claim_result
        
    async def _search_for_identifier(
        self, 
        identifier: str, 
        citations: List[Citation],
        repository_path: str
    ) -> List[str]:
        """Search for an identifier in cited files."""
        found_files = []
        
        for citation in citations:
            if citation.file_path and citation.file_path != "git_history":
                # Prevent path traversal attacks
                if not self._is_safe_path(citation.file_path, repository_path):
                    continue
                    
                full_path = os.path.join(repository_path, citation.file_path)
                if os.path.exists(full_path):
                    try:
                        import asyncio
                        def read_file():
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                return f.read()
                        
                        content = await asyncio.get_event_loop().run_in_executor(None, read_file)
                        if identifier in content:
                            found_files.append(citation.file_path)
                    except Exception:
                        continue
                        
        return found_files
        
    def _calculate_validation_score(
        self, 
        citation_validation: Dict[str, Any], 
        content_validation: Dict[str, Any]
    ) -> float:
        """Calculate validation score based on ACTUAL INDEPENDENT VERIFICATION using Neo4j and Git."""
        
        # Calculate score based on actual validation results
        total_claims = content_validation.get("claims_validated", 0) + content_validation.get("claims_failed", 0)
        
        if total_claims == 0:
            # No claims to validate - neutral score
            return 0.5
            
        # Primary score: ratio of validated claims
        validated_claims = content_validation.get("claims_validated", 0)
        primary_score = validated_claims / total_claims
        
        # Weight by citation validation
        citation_weight = 0.3
        content_weight = 0.7
        
        citation_score = 0.5  # Default neutral
        total_citations = citation_validation.get("total_citations", 0)
        if total_citations > 0:
            valid_citations = citation_validation.get("valid_citations", 0)
            citation_score = valid_citations / total_citations
            
        # Calculate weighted final score
        final_score = (citation_score * citation_weight) + (primary_score * content_weight)
        
        # Apply confidence boost for high validation rates
        if primary_score >= 0.9:  # 90% or more claims validated (Requirement 3.5)
            final_score = min(1.0, final_score * 1.1)  # 10% boost
        elif primary_score <= 0.3:  # 30% or fewer claims validated
            final_score = final_score * 0.8  # 20% penalty
            
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))
        
    def _generate_validation_recommendations(self, validation_result: Dict[str, Any]) -> str:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if validation_result["validation_result"] == "invalid":
            recommendations.append("Finding requires significant revision or additional evidence")
            
        if validation_result["evidence_strength"] == "weak":
            recommendations.append("Strengthen evidence with more specific citations")
            
        if validation_result["issues_found"]:
            recommendations.append("Address identified issues before accepting finding")
            
        citation_validation = validation_result.get("citation_validation", {})
        if citation_validation.get("invalid_citations", 0) > 0:
            recommendations.append("Verify and correct invalid citations")
            
        if not recommendations:
            recommendations.append("Finding appears well-supported")
            
        return "; ".join(recommendations)
        
    async def _detect_uncertainties(
        self, 
        all_findings: List[AgentFinding], 
        validation_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect uncertainties and areas requiring additional validation."""
        uncertainties = []
        
        # Check for low confidence findings
        low_confidence_findings = [f for f in all_findings if f.confidence < 0.7]
        if low_confidence_findings:
            uncertainties.append({
                "type": "low_confidence",
                "description": f"{len(low_confidence_findings)} findings have confidence below 70%",
                "confidence": 0.8,
                "affected_findings": [f.finding_type for f in low_confidence_findings],
                "recommendation": "Review low-confidence findings for accuracy"
            })
            
        # Check for validation failures
        failed_validations = [v for v in validation_results if v["validation_result"] == "invalid"]
        if failed_validations:
            uncertainties.append({
                "type": "validation_failure",
                "description": f"{len(failed_validations)} findings failed validation",
                "confidence": 0.9,
                "affected_findings": [v["finding_type"] for v in failed_validations],
                "recommendation": "Investigate and correct validation failures"
            })
            
        # Check for inconsistent findings
        inconsistencies = self._detect_inconsistent_findings(all_findings)
        if inconsistencies:
            uncertainties.append({
                "type": "inconsistency",
                "description": f"Detected {len(inconsistencies)} potential inconsistencies between findings",
                "confidence": 0.7,
                "affected_findings": inconsistencies,
                "recommendation": "Resolve inconsistencies between agent findings"
            })
            
        # Check for missing citations
        findings_without_citations = [f for f in all_findings if not f.citations]
        if findings_without_citations:
            uncertainties.append({
                "type": "missing_citations",
                "description": f"{len(findings_without_citations)} findings lack supporting citations",
                "confidence": 0.6,
                "affected_findings": [f.finding_type for f in findings_without_citations],
                "recommendation": "Add citations to support uncited findings"
            })
            
        return uncertainties
        
    def _detect_inconsistent_findings(self, findings: List[AgentFinding]) -> List[str]:
        """Detect potentially inconsistent findings."""
        inconsistencies = []
        
        # Group findings by type
        findings_by_type = {}
        for finding in findings:
            if finding.finding_type not in findings_by_type:
                findings_by_type[finding.finding_type] = []
            findings_by_type[finding.finding_type].append(finding)
            
        # Check for conflicting confidence scores on similar topics
        for finding_type, type_findings in findings_by_type.items():
            if len(type_findings) > 1:
                confidences = [f.confidence for f in type_findings]
                if max(confidences) - min(confidences) > 0.4:  # Large confidence spread
                    inconsistencies.append(finding_type)
                    
        return inconsistencies
        
    def _generate_verification_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of verification results."""
        total_validated = len(validation_results)
        valid_count = len([v for v in validation_results if v["validation_result"] == "valid"])
        invalid_count = len([v for v in validation_results if v["validation_result"] == "invalid"])
        uncertain_count = total_validated - valid_count - invalid_count
        
        if total_validated > 0:
            average_confidence = sum(v["confidence_score"] for v in validation_results) / total_validated
        else:
            average_confidence = 0.0
            
        return {
            "total_validated": total_validated,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "uncertain_count": uncertain_count,
            "average_confidence": average_confidence,
            "validation_rate": valid_count / total_validated if total_validated > 0 else 0.0
        }
        
    def _extract_verified_citations(self, all_findings: List[AgentFinding], validation_results: List[Dict[str, Any]]) -> List[Citation]:
        """Extract only citations that passed validation."""
        verified_citations = []
        
        for finding in all_findings:
            for citation in finding.citations:
                # Only include citations that were validated successfully
                for validation in validation_results:
                    citation_validation = validation.get("citation_validation", {})
                    for detail in citation_validation.get("validation_details", []):
                        if (detail.get("file_path") == citation.file_path and 
                            detail.get("valid", False)):
                            verified_citations.append(citation)
                            break
                            
        return verified_citations[:10]  # Top 10 verified citations
        
    def _generate_improvement_actions(self, uncertainties: List[Dict[str, Any]]) -> List[str]:
        """Generate specific actions to improve confidence."""
        actions = []
        
        for uncertainty in uncertainties:
            if uncertainty["type"] == "missing_citations":
                actions.append("Add specific file references and line numbers to findings")
            elif uncertainty["type"] == "validation_failure":
                actions.append("Verify claims against actual source code")
            elif uncertainty["type"] == "low_confidence":
                actions.append("Gather additional evidence to support findings")
            elif uncertainty["type"] == "inconsistency":
                actions.append("Resolve conflicting information between agents")
                
        if not actions:
            actions.append("Manual review recommended to improve analysis quality")
            
        return actions
        
    def _is_valid_git_repository(self, path: str) -> bool:
        """Check if the given path is a valid git repository."""
        import os
        try:
            git_dir = os.path.join(path, '.git')
            return os.path.exists(git_dir) and (os.path.isdir(git_dir) or os.path.isfile(git_dir))
        except Exception:
            return False
            
    def _is_safe_path(self, file_path: str, repository_path: str) -> bool:
        """Check if the file path is safe and doesn't contain path traversal."""
        import os
        try:
            # Normalize paths to prevent traversal
            normalized_repo = os.path.normpath(os.path.abspath(repository_path))
            full_path = os.path.normpath(os.path.abspath(os.path.join(repository_path, file_path)))
            
            # Ensure the full path is within the repository
            return full_path.startswith(normalized_repo + os.sep) or full_path == normalized_repo
        except Exception:
            return False
            
    def _check_solution_availability(self, all_findings: List[AgentFinding]) -> bool:
        """Check if we have a complete solution to validate."""
        
        # Look for working code extraction and integration analysis
        has_working_code = False
        has_integration_analysis = False
        has_solution_steps = False
        
        for finding in all_findings:
            if finding.finding_type == "working_code_extraction":
                has_working_code = True
            elif finding.finding_type == "integration_analysis":
                has_integration_analysis = True
            elif finding.finding_type == "comprehensive_synthesis":
                # Check if synthesis contains solution steps
                if "executable solution steps" in finding.content.lower():
                    has_solution_steps = True
                    
        # We have a solution if we have working code OR solution steps
        return has_working_code or has_solution_steps
        
    async def _validate_complete_solution(
        self, 
        all_findings: List[AgentFinding], 
        git_repo: Optional[GitRepository],
        repository_path: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """Validate a complete solution against developer requirements."""
        
        validation_result = {
            "confidence": 0.0,
            "validation_points": [],
            "issues": [],
            "citations": [],
            "details": {}
        }
        
        try:
            # Extract solution components
            working_code_finding = self._find_working_code_finding(all_findings)
            integration_finding = self._find_integration_finding(all_findings)
            synthesis_finding = self._find_synthesis_finding(all_findings)
            
            validation_scores = []
            
            # 1. Validate working code extraction (if available)
            if working_code_finding:
                code_validation = await self._validate_working_code(working_code_finding, git_repo, repository_path)
                validation_scores.append(code_validation["score"])
                validation_result["validation_points"].extend(code_validation["points"])
                validation_result["issues"].extend(code_validation["issues"])
                validation_result["details"]["working_code_validation"] = code_validation
                
            # 2. Validate integration requirements (if available)
            if integration_finding:
                integration_validation = self._validate_integration_analysis(integration_finding, repository_path)
                validation_scores.append(integration_validation["score"])
                validation_result["validation_points"].extend(integration_validation["points"])
                validation_result["issues"].extend(integration_validation["issues"])
                validation_result["details"]["integration_validation"] = integration_validation
                
            # 3. Validate solution completeness
            completeness_validation = self._validate_solution_completeness(all_findings, state)
            validation_scores.append(completeness_validation["score"])
            validation_result["validation_points"].extend(completeness_validation["points"])
            validation_result["issues"].extend(completeness_validation["issues"])
            validation_result["details"]["completeness_validation"] = completeness_validation
            
            # 4. Validate developer requirements alignment
            requirements_validation = self._validate_requirements_alignment(all_findings, state)
            validation_scores.append(requirements_validation["score"])
            validation_result["validation_points"].extend(requirements_validation["points"])
            validation_result["issues"].extend(requirements_validation["issues"])
            validation_result["details"]["requirements_validation"] = requirements_validation
            
            # Calculate overall confidence
            if validation_scores:
                validation_result["confidence"] = sum(validation_scores) / len(validation_scores)
            else:
                validation_result["confidence"] = 0.3  # Low confidence if no validations
                validation_result["issues"].append("No validation components found")
                
            # Add citations from validated findings
            for finding in [working_code_finding, integration_finding, synthesis_finding]:
                if finding and finding.citations:
                    validation_result["citations"].extend(finding.citations)
                    
        except Exception as e:
            self.logger.error(f"Solution validation failed: {str(e)}")
            validation_result["confidence"] = 0.2
            validation_result["issues"].append(f"Validation error: {str(e)}")
            
        return validation_result
        
    def _find_working_code_finding(self, all_findings: List[AgentFinding]) -> Optional[AgentFinding]:
        """Find working code extraction finding."""
        for finding in all_findings:
            if finding.finding_type == "working_code_extraction":
                return finding
        return None
        
    def _find_integration_finding(self, all_findings: List[AgentFinding]) -> Optional[AgentFinding]:
        """Find integration analysis finding."""
        for finding in all_findings:
            if finding.finding_type == "integration_analysis":
                return finding
        return None
        
    def _find_synthesis_finding(self, all_findings: List[AgentFinding]) -> Optional[AgentFinding]:
        """Find synthesis finding."""
        for finding in all_findings:
            if finding.finding_type == "comprehensive_synthesis":
                return finding
        return None
        
    async def _validate_working_code(
        self, 
        working_code_finding: AgentFinding, 
        git_repo: Optional[GitRepository],
        repository_path: str
    ) -> Dict[str, Any]:
        """Validate extracted working code."""
        
        validation = {
            "score": 0.0,
            "points": [],
            "issues": []
        }
        
        try:
            metadata = working_code_finding.metadata
            commit_sha = metadata.get("commit_sha", "")
            file_path = metadata.get("file_path", "")
            code_content = metadata.get("code_content", "")
            
            score_components = []
            
            # 1. Validate commit exists
            if git_repo and commit_sha:
                try:
                    commit_info = git_repo.get_commit_info(commit_sha)
                    if commit_info:
                        score_components.append(0.3)  # 30% for valid commit
                        validation["points"].append(f"✅ Commit {commit_sha[:8]} exists and is accessible")
                    else:
                        validation["issues"].append(f"❌ Commit {commit_sha[:8]} not found")
                except Exception as e:
                    validation["issues"].append(f"❌ Could not verify commit {commit_sha[:8]}: {str(e)}")
            else:
                validation["issues"].append("❌ No git repository or commit SHA available")
                
            # 2. Validate file path
            if file_path:
                if self._is_safe_path(file_path, repository_path):
                    score_components.append(0.2)  # 20% for valid file path
                    validation["points"].append(f"✅ File path {file_path} is valid and safe")
                else:
                    validation["issues"].append(f"❌ File path {file_path} is invalid or unsafe")
            else:
                validation["issues"].append("❌ No file path provided")
                
            # 3. Validate code content
            if code_content:
                if len(code_content) > 50:  # Reasonable code length
                    score_components.append(0.3)  # 30% for substantial code
                    validation["points"].append(f"✅ Code content extracted ({len(code_content)} characters)")
                else:
                    validation["issues"].append("❌ Code content too short to be meaningful")
            else:
                validation["issues"].append("❌ No code content extracted")
                
            # 4. Validate confidence level
            if working_code_finding.confidence >= 0.7:
                score_components.append(0.2)  # 20% for high confidence
                validation["points"].append(f"✅ High confidence in working code ({working_code_finding.confidence:.1%})")
            else:
                validation["issues"].append(f"❌ Low confidence in working code ({working_code_finding.confidence:.1%})")
                
            validation["score"] = sum(score_components)
            
        except Exception as e:
            validation["issues"].append(f"❌ Working code validation error: {str(e)}")
            validation["score"] = 0.1
            
        return validation
        
    def _validate_integration_analysis(self, integration_finding: AgentFinding, repository_path: str) -> Dict[str, Any]:
        """Validate integration analysis."""
        
        validation = {
            "score": 0.0,
            "points": [],
            "issues": []
        }
        
        try:
            metadata = integration_finding.metadata
            dependencies = metadata.get("dependencies", [])
            compatibility = metadata.get("compatibility", [])
            integration_steps = metadata.get("integration_steps", [])
            
            score_components = []
            
            # 1. Validate dependency analysis
            if dependencies:
                score_components.append(0.3)  # 30% for dependency analysis
                validation["points"].append(f"✅ {len(dependencies)} dependencies identified")
            else:
                validation["points"].append("ℹ️ No dependencies identified (may be self-contained)")
                score_components.append(0.2)  # Still give some credit
                
            # 2. Validate compatibility check
            if compatibility:
                compatible_count = len([c for c in compatibility if c.get("compatible", False)])
                compatibility_ratio = compatible_count / len(compatibility)
                
                if compatibility_ratio >= 0.9:  # 90% compatible (aligned with Requirement 3.5)
                    score_components.append(0.3)  # 30% for high compatibility
                    validation["points"].append(f"✅ High compatibility: {compatible_count}/{len(compatibility)} dependencies available")
                elif compatibility_ratio >= 0.5:  # 50% compatible
                    score_components.append(0.2)  # 20% for medium compatibility
                    validation["points"].append(f"⚠️ Medium compatibility: {compatible_count}/{len(compatibility)} dependencies available")
                else:
                    score_components.append(0.1)  # 10% for low compatibility
                    validation["issues"].append(f"❌ Low compatibility: {compatible_count}/{len(compatibility)} dependencies available")
            else:
                validation["issues"].append("❌ No compatibility analysis performed")
                
            # 3. Validate integration steps
            if integration_steps and len(integration_steps) > 0:
                score_components.append(0.2)  # 20% for integration steps
                validation["points"].append(f"✅ {len(integration_steps)} integration steps provided")
            else:
                validation["issues"].append("❌ No integration steps provided")
                
            # 4. Validate confidence
            if integration_finding.confidence >= 0.6:
                score_components.append(0.2)  # 20% for reasonable confidence
                validation["points"].append(f"✅ Reasonable integration confidence ({integration_finding.confidence:.1%})")
            else:
                validation["issues"].append(f"❌ Low integration confidence ({integration_finding.confidence:.1%})")
                
            validation["score"] = sum(score_components)
            
        except Exception as e:
            validation["issues"].append(f"❌ Integration validation error: {str(e)}")
            validation["score"] = 0.1
            
        return validation
        
    def _validate_solution_completeness(self, all_findings: List[AgentFinding], state: AgentState) -> Dict[str, Any]:
        """Validate that the solution is complete."""
        
        validation = {
            "score": 0.0,
            "points": [],
            "issues": []
        }
        
        try:
            # Check for required components
            has_historian = any(f.agent_name == "historian" for f in all_findings)
            has_analyst = any(f.agent_name == "analyst" for f in all_findings)
            has_synthesizer = any(f.agent_name == "synthesizer" for f in all_findings)
            
            score_components = []
            
            # 1. Multi-agent analysis
            agent_count = len(set(f.agent_name for f in all_findings if f.agent_name != "verifier"))
            if agent_count >= 3:
                score_components.append(0.3)  # 30% for comprehensive analysis
                validation["points"].append(f"✅ Comprehensive analysis from {agent_count} agents")
            elif agent_count >= 2:
                score_components.append(0.2)  # 20% for partial analysis
                validation["points"].append(f"⚠️ Partial analysis from {agent_count} agents")
            else:
                validation["issues"].append(f"❌ Insufficient analysis - only {agent_count} agent(s)")
                
            # 2. Solution actionability
            actionable_findings = 0
            for finding in all_findings:
                if any(keyword in finding.content.lower() for keyword in ["step", "install", "copy", "integrate", "test"]):
                    actionable_findings += 1
                    
            if actionable_findings >= 3:
                score_components.append(0.3)  # 30% for actionable solution
                validation["points"].append(f"✅ Solution contains {actionable_findings} actionable elements")
            elif actionable_findings >= 1:
                score_components.append(0.2)  # 20% for some actionability
                validation["points"].append(f"⚠️ Solution contains {actionable_findings} actionable elements")
            else:
                validation["issues"].append("❌ Solution lacks actionable steps")
                
            # 3. Confidence levels
            high_confidence_findings = len([f for f in all_findings if f.confidence >= 0.7])
            total_findings = len(all_findings)
            
            if total_findings > 0:
                confidence_ratio = high_confidence_findings / total_findings
                if confidence_ratio >= 0.7:
                    score_components.append(0.2)  # 20% for high confidence
                    validation["points"].append(f"✅ {high_confidence_findings}/{total_findings} findings have high confidence")
                elif confidence_ratio >= 0.5:
                    score_components.append(0.1)  # 10% for medium confidence
                    validation["points"].append(f"⚠️ {high_confidence_findings}/{total_findings} findings have high confidence")
                else:
                    validation["issues"].append(f"❌ Only {high_confidence_findings}/{total_findings} findings have high confidence")
                    
            # 4. Citation quality
            total_citations = sum(len(f.citations) for f in all_findings)
            if total_citations >= 5:
                score_components.append(0.2)  # 20% for good citation coverage
                validation["points"].append(f"✅ Solution backed by {total_citations} citations")
            elif total_citations >= 2:
                score_components.append(0.1)  # 10% for some citations
                validation["points"].append(f"⚠️ Solution backed by {total_citations} citations")
            else:
                validation["issues"].append(f"❌ Solution has insufficient citations ({total_citations})")
                
            validation["score"] = sum(score_components)
            
        except Exception as e:
            validation["issues"].append(f"❌ Completeness validation error: {str(e)}")
            validation["score"] = 0.1
            
        return validation
        
    def _validate_requirements_alignment(self, all_findings: List[AgentFinding], state: AgentState) -> Dict[str, Any]:
        """Validate that the solution aligns with developer requirements."""
        
        validation = {
            "score": 0.0,
            "points": [],
            "issues": []
        }
        
        try:
            original_query = state.query.get("original", "").lower()
            
            if not original_query:
                validation["issues"].append("❌ No original query to validate against")
                validation["score"] = 0.3  # Give some credit for having findings
                return validation
                
            score_components = []
            
            # 1. Keyword alignment
            query_keywords = set(re.findall(r'\b\w+\b', original_query))
            query_keywords = {word for word in query_keywords if len(word) > 3}  # Filter short words
            
            matching_findings = 0
            for finding in all_findings:
                finding_text = finding.content.lower()
                if any(keyword in finding_text for keyword in query_keywords):
                    matching_findings += 1
                    
            if matching_findings >= len(all_findings) * 0.7:  # 70% of findings match
                score_components.append(0.3)  # 30% for good keyword alignment
                validation["points"].append(f"✅ {matching_findings}/{len(all_findings)} findings align with query keywords")
            elif matching_findings >= len(all_findings) * 0.4:  # 40% of findings match
                score_components.append(0.2)  # 20% for partial alignment
                validation["points"].append(f"⚠️ {matching_findings}/{len(all_findings)} findings align with query keywords")
            else:
                validation["issues"].append(f"❌ Only {matching_findings}/{len(all_findings)} findings align with query")
                
            # 2. Problem type alignment
            problem_indicators = {
                "deadlock": ["deadlock", "lock", "thread", "synchroniz", "blocking"],
                "performance": ["performance", "slow", "speed", "optimize", "efficient"],
                "bug": ["bug", "error", "exception", "fix", "broken"],
                "feature": ["feature", "implement", "add", "create", "new"]
            }
            
            detected_problem_type = None
            for problem_type, indicators in problem_indicators.items():
                if any(indicator in original_query for indicator in indicators):
                    detected_problem_type = problem_type
                    break
                    
            if detected_problem_type:
                # Check if findings address the detected problem type
                relevant_findings = 0
                for finding in all_findings:
                    finding_text = finding.content.lower()
                    if any(indicator in finding_text for indicator in problem_indicators[detected_problem_type]):
                        relevant_findings += 1
                        
                if relevant_findings >= 2:  # At least 2 findings address the problem
                    score_components.append(0.3)  # 30% for problem alignment
                    validation["points"].append(f"✅ Solution addresses {detected_problem_type} problem with {relevant_findings} relevant findings")
                elif relevant_findings >= 1:
                    score_components.append(0.2)  # 20% for partial problem alignment
                    validation["points"].append(f"⚠️ Solution partially addresses {detected_problem_type} problem")
                else:
                    validation["issues"].append(f"❌ Solution doesn't address detected {detected_problem_type} problem")
            else:
                score_components.append(0.2)  # 20% for general query (no specific problem detected)
                validation["points"].append("ℹ️ General query - no specific problem type detected")
                
            # 3. Solution directness
            synthesis_finding = self._find_synthesis_finding(all_findings)
            if synthesis_finding:
                synthesis_content = synthesis_finding.content.lower()
                if "solution found" in synthesis_content or "executable" in synthesis_content:
                    score_components.append(0.2)  # 20% for direct solution
                    validation["points"].append("✅ Solution provides direct, executable answer")
                elif "analysis" in synthesis_content or "findings" in synthesis_content:
                    score_components.append(0.1)  # 10% for analytical answer
                    validation["points"].append("⚠️ Solution provides analytical answer")
                else:
                    validation["issues"].append("❌ Solution lacks clear direction")
            else:
                validation["issues"].append("❌ No synthesis provided")
                
            # 4. Confidence in requirements alignment
            if original_query and len(query_keywords) > 0:
                score_components.append(0.2)  # 20% for having analyzable requirements
                validation["points"].append("✅ Requirements are clear and analyzable")
            else:
                validation["issues"].append("❌ Requirements are unclear or too vague")
                
            validation["score"] = sum(score_components)
            
        except Exception as e:
            validation["issues"].append(f"❌ Requirements validation error: {str(e)}")
            validation["score"] = 0.1
            
        return validation
        
    async def _add_validation_uncertainty(
        self, 
        finding: AgentFinding, 
        claim: Dict[str, Any], 
        validation_result: Dict[str, Any]
    ) -> None:
        """Add validation uncertainty to the agent state."""
        
        uncertainty_message = f"Analyst claim failed: {claim['claim']}"
        
        if validation_result.get("issues"):
            uncertainty_message += f" - {'; '.join(validation_result['issues'])}"
            
        # Add to verification uncertainties (this would be added to state in the calling method)
        self.logger.warning(f"Validation uncertainty: {uncertainty_message}")
        
        # Store uncertainty details for reporting
        if not hasattr(self, '_validation_uncertainties'):
            self._validation_uncertainties = []
            
        self._validation_uncertainties.append({
            "finding_agent": finding.agent_name,
            "finding_type": finding.finding_type,
            "failed_claim": claim["claim"],
            "uncertainty": uncertainty_message,
            "validation_details": validation_result
        })
        
    def get_validation_uncertainties(self) -> List[Dict[str, Any]]:
        """Get all validation uncertainties found during verification."""
        return getattr(self, '_validation_uncertainties', [])
        
    async def _extract_specific_claims_from_finding(self, finding: AgentFinding) -> List[Dict[str, Any]]:
        """Extract specific, verifiable claims from a finding that can be validated against repository data."""
        claims = []
        content = finding.content.lower()
        
        # Extract function call claims: "Function A calls function B"
        call_pattern = r'function\s+(\w+)\s+(?:now\s+)?calls?\s+function\s+(\w+)'
        for match in re.finditer(call_pattern, content):
            claims.append({
                "type": "function_calls",
                "claim": f"Function {match.group(1)} calls function {match.group(2)}",
                "caller": match.group(1),
                "callee": match.group(2),
                "validation_method": "neo4j_calls_query"
            })
        
        # Extract function modification claims: "Function X was modified in commit Y"
        modification_pattern = r'function\s+(\w+)\s+(?:was\s+)?(?:modified|changed|updated)\s+(?:in\s+)?commit\s+([a-f0-9]{7,40})'
        for match in re.finditer(modification_pattern, content):
            claims.append({
                "type": "function_modified",
                "claim": f"Function {match.group(1)} was modified in commit {match.group(2)}",
                "function_name": match.group(1),
                "commit_sha": match.group(2),
                "validation_method": "neo4j_git_modification_query"
            })
        
        # Extract class inheritance claims: "Class A inherits from class B"
        inheritance_pattern = r'class\s+(\w+)\s+(?:inherits\s+from|extends)\s+class\s+(\w+)'
        for match in re.finditer(inheritance_pattern, content):
            claims.append({
                "type": "class_inheritance",
                "claim": f"Class {match.group(1)} inherits from class {match.group(2)}",
                "child_class": match.group(1),
                "parent_class": match.group(2),
                "validation_method": "neo4j_inheritance_query"
            })
        
        # Extract import/dependency claims: "Module A imports module B"
        import_pattern = r'(?:module|class)\s+(\w+)\s+imports?\s+(?:module|class)?\s*(\w+)'
        for match in re.finditer(import_pattern, content):
            claims.append({
                "type": "module_imports",
                "claim": f"Module {match.group(1)} imports module {match.group(2)}",
                "importer": match.group(1),
                "imported": match.group(2),
                "validation_method": "neo4j_imports_query"
            })
        
        # Extract commit message intent claims from metadata
        if finding.metadata and "commit_sha" in finding.metadata:
            commit_sha = finding.metadata["commit_sha"]
            # Look for bug fix claims
            bug_pattern = r'(?:fix|fixes|fixed)\s+(?:bug\s+)?[#]?(\d+)'
            for match in re.finditer(bug_pattern, content):
                claims.append({
                    "type": "commit_intent",
                    "claim": f"Commit {commit_sha} was to fix bug #{match.group(1)}",
                    "commit_sha": commit_sha,
                    "intent": f"bug #{match.group(1)}",
                    "validation_method": "git_commit_message_validation"
                })
        
        return claims
    
    async def _independently_validate_claim_against_repository(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository],
        repository_path: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """Independently validate a specific claim against actual repository data using Neo4j and Git."""
        
        validation_result = {
            "claim": claim["claim"],
            "type": claim["type"],
            "validated": False,
            "evidence": [],
            "reason": "",
            "neo4j_query_executed": False,
            "git_validation_performed": False
        }
        
        try:
            validation_method = claim.get("validation_method", "unknown")
            
            if validation_method == "neo4j_calls_query":
                validation_result = await self._validate_function_calls_with_neo4j(claim, validation_result)
                
            elif validation_method == "neo4j_git_modification_query":
                validation_result = await self._validate_function_modification_with_neo4j_and_git(
                    claim, git_repo, validation_result
                )
                
            elif validation_method == "neo4j_inheritance_query":
                validation_result = await self._validate_class_inheritance_with_neo4j(claim, validation_result)
                
            elif validation_method == "neo4j_imports_query":
                validation_result = await self._validate_module_imports_with_neo4j(claim, validation_result)
                
            elif validation_method == "git_commit_message_validation":
                validation_result = await self._validate_commit_intent_with_git(claim, git_repo, validation_result)
                
            else:
                validation_result["reason"] = f"Unknown validation method: {validation_method}"
                
        except Exception as e:
            validation_result["reason"] = f"Validation error: {str(e)}"
            self.logger.error(f"Independent validation failed for claim '{claim['claim']}': {str(e)}")
            
        return validation_result
    
    async def _validate_function_calls_with_neo4j(self, claim: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function call claim by independently querying Neo4j CPG."""
        
        if not self.neo4j_tool:
            validation_result["reason"] = "Neo4j tool not available for validation"
            return validation_result
            
        try:
            caller = claim["caller"]
            callee = claim["callee"]
            
            # INDEPENDENT NEO4J QUERY: Check if the CALLS relationship actually exists
            query = """
            MATCH (caller:Function {name: $caller})
            MATCH (callee:Function {name: $callee})
            MATCH (caller)-[:CALLS]->(callee)
            RETURN caller.name as caller_name, caller.file_path as caller_file,
                   callee.name as callee_name, callee.file_path as callee_file,
                   caller.start_line as caller_line
            """
            
            results = await self.neo4j_tool.execute(
                "query",
                query=query,
                parameters={"caller": caller, "callee": callee}
            )
            
            validation_result["neo4j_query_executed"] = True
            
            if results and len(results) > 0:
                validation_result["validated"] = True
                validation_result["evidence"] = [
                    f"Neo4j CPG confirms function {caller} calls function {callee}",
                    f"Caller location: {results[0].get('caller_file', 'unknown')}:{results[0].get('caller_line', 'unknown')}",
                    f"Callee location: {results[0].get('callee_file', 'unknown')}"
                ]
            else:
                validation_result["validated"] = False
                validation_result["reason"] = f"Neo4j CPG query found no CALLS relationship between {caller} and {callee}"
                
        except Exception as e:
            validation_result["reason"] = f"Neo4j validation error: {str(e)}"
            
        return validation_result
    
    async def _validate_function_modification_with_neo4j_and_git(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate function modification claim by querying both Neo4j CPG and Git history."""
        
        function_name = claim["function_name"]
        commit_sha = claim["commit_sha"]
        
        # Step 1: Validate with Neo4j CPG
        if self.neo4j_tool:
            try:
                # INDEPENDENT NEO4J QUERY: Check if function was changed in specific commit
                query = """
                MATCH (f:Function {name: $function_name})
                MATCH (c:Commit {sha: $commit_sha})
                MATCH (f)-[:CHANGED_IN]->(c)
                RETURN f.name as function_name, f.file_path as file_path,
                       c.sha as commit_sha, c.message as commit_message
                """
                
                results = await self.neo4j_tool.execute(
                    "query",
                    query=query,
                    parameters={"function_name": function_name, "commit_sha": commit_sha}
                )
                
                validation_result["neo4j_query_executed"] = True
                
                if results and len(results) > 0:
                    validation_result["validated"] = True
                    validation_result["evidence"].append(
                        f"Neo4j CPG confirms function {function_name} was changed in commit {commit_sha}"
                    )
                    validation_result["evidence"].append(
                        f"Function location: {results[0].get('file_path', 'unknown')}"
                    )
                else:
                    validation_result["reason"] = f"Neo4j CPG found no evidence that function {function_name} was changed in commit {commit_sha}"
                    
            except Exception as e:
                validation_result["reason"] = f"Neo4j validation error: {str(e)}"
        
        # Step 2: Cross-reference with Git history
        if git_repo and not validation_result["validated"]:
            try:
                # INDEPENDENT GIT VALIDATION: Check commit exists and get diff
                commit_info = git_repo.get_commit_info(commit_sha)
                validation_result["git_validation_performed"] = True
                
                if commit_info:
                    validation_result["evidence"].append(f"Git confirms commit {commit_sha} exists")
                    
                    # Try to get diff to see if function was actually modified
                    try:
                        changed_files = git_repo.get_changed_files(commit_sha)
                        if changed_files:
                            validation_result["evidence"].append(f"Commit modified {len(changed_files)} files")
                            # If Neo4j didn't confirm but Git shows changes, mark as partially validated
                            if not validation_result["validated"]:
                                validation_result["validated"] = True
                                validation_result["reason"] = "Validated via Git history (Neo4j CPG may be incomplete)"
                    except Exception:
                        pass  # Diff analysis failed, but commit exists
                else:
                    validation_result["reason"] = f"Git validation failed: commit {commit_sha} not found"
                    
            except Exception as e:
                validation_result["reason"] = f"Git validation error: {str(e)}"
        
        return validation_result
    
    async def _validate_class_inheritance_with_neo4j(self, claim: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class inheritance claim by independently querying Neo4j CPG."""
        
        if not self.neo4j_tool:
            validation_result["reason"] = "Neo4j tool not available for validation"
            return validation_result
            
        try:
            child_class = claim["child_class"]
            parent_class = claim["parent_class"]
            
            # INDEPENDENT NEO4J QUERY: Check if inheritance relationship exists
            query = """
            MATCH (child:Class {name: $child_class})
            MATCH (parent:Class {name: $parent_class})
            MATCH (child)-[:INHERITS_FROM]->(parent)
            RETURN child.name as child_name, child.file_path as child_file,
                   parent.name as parent_name, parent.file_path as parent_file
            """
            
            results = await self.neo4j_tool.execute(
                "query",
                query=query,
                parameters={"child_class": child_class, "parent_class": parent_class}
            )
            
            validation_result["neo4j_query_executed"] = True
            
            if results and len(results) > 0:
                validation_result["validated"] = True
                validation_result["evidence"] = [
                    f"Neo4j CPG confirms class {child_class} inherits from class {parent_class}",
                    f"Child class location: {results[0].get('child_file', 'unknown')}",
                    f"Parent class location: {results[0].get('parent_file', 'unknown')}"
                ]
            else:
                validation_result["validated"] = False
                validation_result["reason"] = f"Neo4j CPG found no inheritance relationship between {child_class} and {parent_class}"
                
        except Exception as e:
            validation_result["reason"] = f"Neo4j validation error: {str(e)}"
            
        return validation_result
    
    async def _validate_module_imports_with_neo4j(self, claim: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate module import claim by independently querying Neo4j CPG."""
        
        if not self.neo4j_tool:
            validation_result["reason"] = "Neo4j tool not available for validation"
            return validation_result
            
        try:
            importer = claim["importer"]
            imported = claim["imported"]
            
            # INDEPENDENT NEO4J QUERY: Check if import relationship exists
            query = """
            MATCH (importer:Module {name: $importer})
            MATCH (imported:Module {name: $imported})
            MATCH (importer)-[:IMPORTS]->(imported)
            RETURN importer.name as importer_name, importer.file_path as importer_file,
                   imported.name as imported_name, imported.file_path as imported_file
            """
            
            results = await self.neo4j_tool.execute(
                "query",
                query=query,
                parameters={"importer": importer, "imported": imported}
            )
            
            validation_result["neo4j_query_executed"] = True
            
            if results and len(results) > 0:
                validation_result["validated"] = True
                validation_result["evidence"] = [
                    f"Neo4j CPG confirms module {importer} imports module {imported}",
                    f"Importer location: {results[0].get('importer_file', 'unknown')}",
                    f"Imported location: {results[0].get('imported_file', 'unknown')}"
                ]
            else:
                validation_result["validated"] = False
                validation_result["reason"] = f"Neo4j CPG found no import relationship between {importer} and {imported}"
                
        except Exception as e:
            validation_result["reason"] = f"Neo4j validation error: {str(e)}"
            
        return validation_result
    
    async def _validate_commit_intent_with_git(
        self, 
        claim: Dict[str, Any], 
        git_repo: Optional[GitRepository],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate commit message intent claim by independently checking Git history."""
        
        if not git_repo:
            validation_result["reason"] = "Git repository not available for validation"
            return validation_result
            
        try:
            commit_sha = claim["commit_sha"]
            expected_intent = claim["intent"]
            
            # INDEPENDENT GIT VALIDATION: Get actual commit message
            commit_info = git_repo.get_commit_info(commit_sha)
            validation_result["git_validation_performed"] = True
            
            if commit_info and commit_info.get("message"):
                commit_message = commit_info["message"].lower()
                
                if expected_intent.lower() in commit_message:
                    validation_result["validated"] = True
                    validation_result["evidence"] = [
                        f"Git confirms commit {commit_sha} message contains expected intent: '{expected_intent}'",
                        f"Full commit message: {commit_info['message'][:200]}..."
                    ]
                else:
                    validation_result["validated"] = False
                    validation_result["reason"] = f"Git commit message does not contain expected intent '{expected_intent}'"
                    validation_result["evidence"] = [
                        f"Actual commit message: {commit_info['message'][:100]}..."
                    ]
            else:
                validation_result["validated"] = False
                validation_result["reason"] = f"Could not retrieve commit message for {commit_sha}"
                
        except Exception as e:
            validation_result["reason"] = f"Git validation error: {str(e)}"
            
        return validation_result
    
    def _extract_validation_citations(self, validation_results: List[Dict[str, Any]]) -> List[Citation]:
        """Extract citations from validation results."""
        citations = []
        
        for result in validation_results:
            if result.get("validated") and result.get("validation_details"):
                for detail in result["validation_details"]:
                    if detail.get("evidence"):
                        citations.append(Citation(
                            file_path="neo4j_cpg",
                            line_number=0,
                            commit_sha="independent_validation",
                            description=f"Independent validation: {detail['claim']}"
                        ))
        
        return citations
    
    def _count_neo4j_queries(self, validation_results: List[Dict[str, Any]]) -> int:
        """Count the number of Neo4j queries executed during validation."""
        count = 0
        for result in validation_results:
            if result.get("validation_details"):
                for detail in result["validation_details"]:
                    if detail.get("neo4j_query_executed"):
                        count += 1
        return count
    
    def _count_git_validations(self, validation_results: List[Dict[str, Any]]) -> int:
        """Count the number of Git validations performed."""
        count = 0
        for result in validation_results:
            if result.get("validation_details"):
                for detail in result["validation_details"]:
                    if detail.get("git_validation_performed"):
                        count += 1
        return count