"""Verification Agent for independent validation of all findings."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..git.repository import GitRepository
from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, AgentFinding, Citation


logger = get_logger(__name__)


class VerificationAgent(BaseAgent):
    """Agent responsible for independent validation of findings against source data."""
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize the Verification Agent."""
        if config is None:
            config = AgentConfig(
                name="verifier",
                description="Validates findings against actual source code and git history"
            )
        super().__init__(config, **kwargs)
        
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
        """Execute verification logic."""
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
                
                # QUALITY GATE: 80-90% confidence threshold for solutions
                if solution_validation["confidence"] >= 0.8:
                    validation_status = "SOLUTION_APPROVED"
                    validation_message = f"✅ Solution validated with {solution_validation['confidence']:.1%} confidence"
                else:
                    validation_status = "SOLUTION_NEEDS_REVIEW"
                    validation_message = f"⚠️ Solution needs review - {solution_validation['confidence']:.1%} confidence (below 80% threshold)"
                    
                # Create solution validation finding
                solution_finding = self._create_finding(
                    finding_type="solution_validation",
                    content=validation_message,
                    confidence=solution_validation["confidence"],
                    citations=solution_validation.get("citations", []),
                    metadata={
                        "validation_status": validation_status,
                        "validation_details": solution_validation,
                        "threshold_met": solution_validation["confidence"] >= 0.8
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
                
                # QUALITY GATE: 80% threshold for individual findings
                if overall_confidence >= 0.8:
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
                           f"Issues found: {len(uncertainties)} uncertainties detected.\n"
                           f"Recommendation: Gather more data or manual review required.",
                    confidence=overall_confidence,
                    citations=[],
                    metadata={
                        "validation_results": validation_results,
                        "verification_summary": verification_summary,
                        "quality_gate": "FAILED",
                        "ready_for_user": False,
                        "required_actions": self._generate_improvement_actions(uncertainties)
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
            
            # Determine validation result
            if overall_score >= 0.8:
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
        """Validate content claims against actual source code."""
        content_validation = {
            "claims_validated": 0,
            "claims_failed": 0,
            "validation_details": []
        }
        
        # Extract specific claims from the finding content
        claims = self._extract_claims_from_content(finding.content)
        
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
        """Calculate overall validation score."""
        citation_score = 0.0
        content_score = 0.0
        
        # Citation validation score
        total_citations = citation_validation["total_citations"]
        if total_citations > 0:
            valid_citations = citation_validation["valid_citations"]
            citation_score = valid_citations / total_citations
        else:
            citation_score = 0.5  # Neutral score if no citations
            
        # Content validation score
        total_claims = content_validation.get("claims_validated", 0) + content_validation.get("claims_failed", 0)
        if total_claims > 0:
            valid_claims = content_validation.get("claims_validated", 0)
            content_score = valid_claims / total_claims
        else:
            content_score = 0.5  # Neutral score if no claims
            
        # Weighted average (citations are more important)
        overall_score = (citation_score * 0.6) + (content_score * 0.4)
        return overall_score
        
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
                
                if compatibility_ratio >= 0.8:  # 80% compatible
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