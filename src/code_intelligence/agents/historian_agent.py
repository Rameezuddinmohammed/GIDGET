"""Historian Agent for git history analysis and temporal queries."""

import json
import re
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..git.repository import GitRepository
from ..logging import get_logger
from .base import BaseAgent, AgentConfig, PromptTemplate
from .state import AgentState, TimeRange, Citation
from .config import get_agent_config


logger = get_logger(__name__)


class HistorianAgent(BaseAgent):
    """Agent responsible for git history analysis and temporal data extraction."""
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """Initialize the Historian Agent."""
        if config is None:
            config = AgentConfig(
                name="historian",
                description="Analyzes git history and performs temporal code evolution tracking"
            )
        super().__init__(config, **kwargs)
        
        # Register temporal analysis templates
        self._register_templates()
        
    def _register_templates(self) -> None:
        """Register prompt templates for temporal analysis."""
        commit_analysis_template = PromptTemplate(
            """Analyze the following git commits for code evolution patterns and developer intent.

Target Elements: {target_elements}
Commits to Analyze:
{commit_data}

For each commit, extract:
1. Changes related to target elements
2. Developer intent from commit message
3. Impact assessment (low/medium/high)
4. Relationships to other changes

Respond in JSON format:
{{
    "timeline": [
        {{
            "commit_sha": "abc123",
            "timestamp": "2024-01-01T12:00:00Z",
            "author": "developer@example.com",
            "message": "commit message",
            "intent": "extracted developer intent",
            "changes": [
                {{
                    "element": "element_name",
                    "change_type": "added|modified|deleted|renamed",
                    "impact": "low|medium|high",
                    "description": "what changed"
                }}
            ],
            "related_commits": ["sha1", "sha2"]
        }}
    ],
    "patterns": [
        {{
            "pattern_type": "refactoring|bug_fix|feature_addition|optimization",
            "description": "pattern description",
            "commits": ["sha1", "sha2"],
            "confidence": 0.0-1.0
        }}
    ],
    "evolution_summary": "overall evolution narrative"
}}""",
            variables=["target_elements", "commit_data"]
        )
        
        temporal_query_template = PromptTemplate(
            """Answer the temporal query based on git history analysis.

Query: {original_query}
Time Range: {time_range}
Historical Data: {historical_data}

Provide a comprehensive answer that includes:
1. Timeline of relevant changes
2. Key evolution points
3. Developer intent and rationale
4. Impact on codebase

Format as a narrative with specific citations to commits and files.""",
            variables=["original_query", "time_range", "historical_data"]
        )
        
        self.commit_analysis_template = commit_analysis_template
        self.temporal_query_template = temporal_query_template
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute historian analysis."""
        self._log_execution_start(state)
        
        if not self._validate_state(state):
            state.add_error("Invalid state for historian", self.config.name)
            return state
            
        try:
            repository_path = state.repository.get("path", "")
            if not repository_path:
                state.add_error("No repository path provided", self.config.name)
                return state
                
            # Validate and initialize git repository
            if not self._is_valid_git_repository(repository_path):
                state.add_error(f"Invalid git repository: {repository_path}", self.config.name)
                return state
                
            git_repo = GitRepository(repository_path)
            
            # Determine time range for analysis
            time_range = await self._determine_time_range(state)
            
            # Get commit history
            commits = await self._get_relevant_commits(git_repo, state, time_range)
            
            # Extract actual working code if developer query requires it
            working_code_result = await self._find_and_extract_working_code(
                git_repo, repository_path, state, commits
            )
            
            # Analyze commit patterns and evolution
            analysis_result = await self._analyze_commit_history(state, commits)
            
            # Generate temporal insights
            insights = await self._generate_temporal_insights(state, analysis_result)
            
            # Add working code finding if found
            if working_code_result["found"]:
                insights.append({
                    "type": "working_code_extraction",
                    "content": working_code_result["summary"],
                    "confidence": working_code_result["confidence"],
                    "citations": working_code_result["citations"],
                    "metadata": working_code_result["metadata"]
                })
            
            # Create historian findings
            for insight in insights:
                finding = self._create_finding(
                    finding_type=insight["type"],
                    content=insight["content"],
                    confidence=insight["confidence"],
                    citations=insight.get("citations", []),
                    metadata=insight.get("metadata", {})
                )
                state.add_finding(self.config.name, finding)
                
            # Update state with temporal data
            state.analysis["temporal_data"] = {
                "time_range": time_range.model_dump() if time_range else None,
                "commit_count": len(commits),
                "analysis_result": analysis_result
            }
            
            self._log_execution_end(state, True)
            return state
            
        except Exception as e:
            error_context = {
                "session_id": state.session_id,
                "repository_path": state.repository.get("path", ""),
                "target_elements_count": len(state.analysis.get("target_elements", [])),
                "error_type": type(e).__name__
            }
            self.logger.error(f"Historian execution failed: {str(e)}", extra=error_context)
            state.add_error(f"Historian failed: {str(e)}", self.config.name)
            self._log_execution_end(state, False)
            return state
            
    async def _determine_time_range(self, state: AgentState) -> Optional[TimeRange]:
        """Determine the time range for temporal analysis."""
        parsed_query = state.query.get("parsed", {})
        time_range_str = parsed_query.get("time_range")
        
        if not time_range_str:
            # Default to last 30 days if no time range specified
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            return TimeRange(start_date=start_date, end_date=end_date)
            
        # Parse time range string
        time_range_str = time_range_str.lower()
        end_date = datetime.utcnow()
        
        if "week" in time_range_str:
            if "last week" in time_range_str:
                start_date = end_date - timedelta(weeks=1)
            else:
                # Extract number of weeks
                weeks_match = re.search(r'(\d+)\s*weeks?', time_range_str)
                weeks = int(weeks_match.group(1)) if weeks_match else 1
                start_date = end_date - timedelta(weeks=weeks)
        elif "month" in time_range_str:
            months_match = re.search(r'(\d+)\s*months?', time_range_str)
            months = int(months_match.group(1)) if months_match else 1
            start_date = end_date - timedelta(days=30 * months)
        elif "day" in time_range_str:
            days_match = re.search(r'(\d+)\s*days?', time_range_str)
            days = int(days_match.group(1)) if days_match else 7
            start_date = end_date - timedelta(days=days)
        elif "since" in time_range_str:
            # Look for commit SHA
            commit_match = re.search(r'[a-f0-9]{7,40}', time_range_str)
            if commit_match:
                return TimeRange(start_commit=commit_match.group())
            else:
                start_date = end_date - timedelta(days=30)
        else:
            # Default fallback
            start_date = end_date - timedelta(days=30)
            
        return TimeRange(start_date=start_date, end_date=end_date)
        
    async def _get_relevant_commits(
        self, 
        git_repo: GitRepository, 
        state: AgentState, 
        time_range: Optional[TimeRange]
    ) -> List[Dict[str, Any]]:
        """Get commits relevant to the analysis."""
        try:
            # Get target elements from analysis state
            target_elements = state.analysis.get("target_elements", [])
            
            # Build file paths to filter commits
            file_paths = []
            for element in target_elements:
                if element.get("file_path"):
                    file_paths.append(element["file_path"])
                    
            # Get commits with filters
            commits = []
            
            if time_range:
                if time_range.start_commit:
                    # Get commits since a specific commit
                    raw_commits = git_repo.get_commits_since(time_range.start_commit, limit=100)
                else:
                    # Get commits in date range
                    raw_commits = git_repo.get_commits_in_range(
                        since=time_range.start_date,
                        until=time_range.end_date,
                        limit=100
                    )
            else:
                # Get recent commits
                raw_commits = git_repo.get_recent_commits(limit=50)
                
            # Filter commits by file paths if specified
            for commit in raw_commits:
                commit_data = {
                    "sha": commit.sha,
                    "message": commit.message,
                    "author": commit.author.name,
                    "author_email": commit.author.email,
                    "timestamp": commit.committed_datetime.isoformat(),
                    "files_changed": []
                }
                
                # Get files changed in this commit
                try:
                    changed_files = git_repo.get_changed_files(commit.sha)
                    commit_data["files_changed"] = changed_files
                    
                    # Include commit if it affects target files or if no specific files
                    if not file_paths or any(fp in changed_files for fp in file_paths):
                        commits.append(commit_data)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get changed files for commit {commit.sha}: {str(e)}")
                    commits.append(commit_data)  # Include anyway
                    
            from .config import get_agent_config
            config = get_agent_config()
            return commits[:config.limits.max_commits]  # Configurable limit
            
        except Exception as e:
            self.logger.error(f"Failed to get commits: {str(e)}")
            return []
            
    async def _analyze_commit_history(
        self, 
        state: AgentState, 
        commits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze commit history for patterns and evolution."""
        if not commits:
            return {"timeline": [], "patterns": [], "evolution_summary": "No commits found in specified range"}
            
        target_elements = state.analysis.get("target_elements", [])
        target_names = [elem.get("name", "") for elem in target_elements]
        
        # Prepare commit data for LLM analysis
        commit_data = []
        for commit in commits[:20]:  # Limit to 20 commits for LLM analysis
            commit_summary = (
                f"Commit: {commit['sha'][:8]}\n"
                f"Author: {commit['author']} <{commit['author_email']}>\n"
                f"Date: {commit['timestamp']}\n"
                f"Message: {commit['message']}\n"
                f"Files: {', '.join(commit['files_changed'][:5])}\n"
            )
            commit_data.append(commit_summary)
            
        prompt = self.commit_analysis_template.format(
            target_elements=", ".join(target_names) if target_names else "all code elements",
            commit_data="\n---\n".join(commit_data)
        )
        
        system_prompt = (
            "You are an expert at analyzing git commit history for code evolution patterns. "
            "Extract meaningful insights about how code has changed over time."
        )
        
        response = await self._call_llm(prompt, system_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_commit_analysis(commits)
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse commit analysis, using fallback: {str(e)}")
            return self._fallback_commit_analysis(commits)
            
    def _fallback_commit_analysis(self, commits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback commit analysis using rule-based approach."""
        timeline = []
        patterns = []
        
        # Create basic timeline
        for commit in commits:
            timeline.append({
                "commit_sha": commit["sha"],
                "timestamp": commit["timestamp"],
                "author": commit["author_email"],
                "message": commit["message"],
                "intent": self._extract_intent_from_message(commit["message"]),
                "changes": [
                    {
                        "element": "unknown",
                        "change_type": "modified",
                        "impact": "medium",
                        "description": f"Modified {len(commit['files_changed'])} files"
                    }
                ],
                "related_commits": []
            })
            
        # Identify basic patterns
        bug_fix_commits = [c for c in commits if any(word in c["message"].lower() 
                          for word in ["fix", "bug", "error", "issue"])]
        if bug_fix_commits:
            patterns.append({
                "pattern_type": "bug_fix",
                "description": f"Found {len(bug_fix_commits)} bug fix commits",
                "commits": [c["sha"] for c in bug_fix_commits],
                "confidence": 0.8
            })
            
        feature_commits = [c for c in commits if any(word in c["message"].lower() 
                          for word in ["add", "feature", "implement", "new"])]
        if feature_commits:
            patterns.append({
                "pattern_type": "feature_addition",
                "description": f"Found {len(feature_commits)} feature addition commits",
                "commits": [c["sha"] for c in feature_commits],
                "confidence": 0.7
            })
            
        return {
            "timeline": timeline,
            "patterns": patterns,
            "evolution_summary": f"Analyzed {len(commits)} commits with {len(patterns)} patterns identified"
        }
        
    def _extract_intent_from_message(self, message: str) -> str:
        """Extract developer intent from commit message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["fix", "bug", "error", "issue"]):
            return "bug_fix"
        elif any(word in message_lower for word in ["add", "feature", "implement", "new"]):
            return "feature_addition"
        elif any(word in message_lower for word in ["refactor", "cleanup", "reorganize"]):
            return "refactoring"
        elif any(word in message_lower for word in ["update", "upgrade", "improve"]):
            return "improvement"
        elif any(word in message_lower for word in ["remove", "delete", "deprecate"]):
            return "removal"
        else:
            return "maintenance"
            
    async def _generate_temporal_insights(
        self, 
        state: AgentState, 
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate temporal insights from analysis results."""
        insights = []
        
        # Timeline insight
        timeline = analysis_result.get("timeline", [])
        if timeline:
            citations = []
            for entry in timeline[:5]:  # Top 5 commits
                citations.append(self._create_citation(
                    file_path="git_history",
                    description=f"Commit {entry['commit_sha'][:8]}: {entry['message'][:50]}...",
                    commit_sha=entry["commit_sha"]
                ))
                
            insights.append({
                "type": "temporal_timeline",
                "content": f"Identified {len(timeline)} relevant commits in the analysis period. "
                          f"Key evolution points include {len(analysis_result.get('patterns', []))} distinct patterns.",
                "confidence": 0.9,
                "citations": citations,
                "metadata": {"timeline_length": len(timeline)}
            })
            
        # Pattern insights
        patterns = analysis_result.get("patterns", [])
        for pattern in patterns:
            pattern_citations = []
            for commit_sha in pattern.get("commits", [])[:3]:  # Top 3 commits per pattern
                pattern_citations.append(self._create_citation(
                    file_path="git_history",
                    description=f"Pattern evidence in commit {commit_sha[:8]}",
                    commit_sha=commit_sha
                ))
                
            insights.append({
                "type": f"evolution_pattern_{pattern['pattern_type']}",
                "content": pattern["description"],
                "confidence": pattern.get("confidence", 0.7),
                "citations": pattern_citations,
                "metadata": {"pattern_type": pattern["pattern_type"]}
            })
            
        # Evolution summary insight
        evolution_summary = analysis_result.get("evolution_summary", "")
        if evolution_summary:
            insights.append({
                "type": "evolution_summary",
                "content": evolution_summary,
                "confidence": 0.8,
                "citations": [],
                "metadata": {"summary_type": "temporal_evolution"}
            })
            
        return insights
        
    def create_timeline_visualization(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create timeline visualization data."""
        timeline = analysis_result.get("timeline", [])
        
        # Group commits by time periods
        periods = {}
        for entry in timeline:
            timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
            period_key = timestamp.strftime("%Y-%m")
            
            if period_key not in periods:
                periods[period_key] = {
                    "period": period_key,
                    "commits": [],
                    "change_types": {},
                    "authors": set()
                }
                
            periods[period_key]["commits"].append(entry)
            periods[period_key]["authors"].add(entry["author"])
            
            # Count change types
            for change in entry.get("changes", []):
                change_type = change.get("change_type", "unknown")
                periods[period_key]["change_types"][change_type] = (
                    periods[period_key]["change_types"].get(change_type, 0) + 1
                )
                
        # Convert to visualization format
        visualization_data = {
            "periods": [
                {
                    **period_data,
                    "authors": list(period_data["authors"]),
                    "commit_count": len(period_data["commits"])
                }
                for period_data in periods.values()
            ],
            "total_commits": len(timeline),
            "date_range": {
                "start": min(entry["timestamp"] for entry in timeline) if timeline else None,
                "end": max(entry["timestamp"] for entry in timeline) if timeline else None
            }
        }
        
        return visualization_data
        
    def extract_developer_intent(self, commit_messages: List[str]) -> Dict[str, Any]:
        """Extract developer intent patterns from commit messages."""
        intent_patterns = {
            "bug_fixes": [],
            "features": [],
            "refactoring": [],
            "documentation": [],
            "testing": [],
            "maintenance": []
        }
        
        for message in commit_messages:
            message_lower = message.lower()
            
            if any(word in message_lower for word in ["fix", "bug", "error", "issue", "resolve"]):
                intent_patterns["bug_fixes"].append(message)
            elif any(word in message_lower for word in ["add", "feature", "implement", "new"]):
                intent_patterns["features"].append(message)
            elif any(word in message_lower for word in ["refactor", "cleanup", "reorganize", "restructure"]):
                intent_patterns["refactoring"].append(message)
            elif any(word in message_lower for word in ["doc", "readme", "comment", "documentation"]):
                intent_patterns["documentation"].append(message)
            elif any(word in message_lower for word in ["test", "spec", "coverage"]):
                intent_patterns["testing"].append(message)
            else:
                intent_patterns["maintenance"].append(message)
                
        # Calculate intent distribution
        total_commits = len(commit_messages)
        intent_distribution = {
            intent: len(messages) / total_commits if total_commits > 0 else 0
            for intent, messages in intent_patterns.items()
        }
        
        return {
            "patterns": intent_patterns,
            "distribution": intent_distribution,
            "dominant_intent": max(intent_distribution.items(), key=lambda x: x[1])[0] if intent_distribution else "unknown"
        }
        
    def _is_valid_git_repository(self, path: str) -> bool:
        """Check if the given path is a valid git repository."""
        import os
        try:
            git_dir = os.path.join(path, '.git')
            return os.path.exists(git_dir) and (os.path.isdir(git_dir) or os.path.isfile(git_dir))
        except Exception:
            return False
            
    async def _find_and_extract_working_code(
        self, 
        git_repo: GitRepository, 
        repository_path: str,
        state: AgentState, 
        commits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find and extract actual working code from git history."""
        
        result = {
            "found": False,
            "commit_sha": None,
            "file_path": None,
            "code_content": None,
            "summary": "",
            "confidence": 0.0,
            "citations": [],
            "metadata": {}
        }
        
        try:
            # Extract requirements from developer query
            query = state.query.get("original", "").lower()
            
            # Look for specific feature mentions
            feature_keywords = self._extract_feature_keywords(query)
            problem_type = self._identify_problem_type(query)
            
            # Score and filter commits for relevance
            relevant_commits = []
            for commit in commits:
                score = self._score_commit_for_working_code(commit, feature_keywords, problem_type)
                if score > 0.3:  # Only consider relevant commits
                    relevant_commits.append((commit, score))
                    
            # Sort by relevance score
            relevant_commits.sort(key=lambda x: x[1], reverse=True)
            
            # Try to extract working code from top commits
            for commit_data, score in relevant_commits[:5]:  # Check top 5 commits
                extraction_result = await self._extract_code_from_commit(
                    git_repo, repository_path, commit_data, feature_keywords
                )
                
                if extraction_result["success"]:
                    result.update({
                        "found": True,
                        "commit_sha": commit_data["sha"],
                        "file_path": extraction_result["file_path"],
                        "code_content": extraction_result["code_content"],
                        "summary": f"Extracted working code from commit {commit_data['sha'][:8]}: {extraction_result['description']}",
                        "confidence": min(0.9, 0.7 + score * 0.2),  # High confidence for relevant commits
                        "citations": [self._create_citation(
                            file_path=extraction_result["file_path"],
                            commit_sha=commit_data["sha"],
                            description=f"Working implementation extracted from commit {commit_data['sha'][:8]}"
                        )],
                        "metadata": {
                            "commit_message": commit_data["message"],
                            "commit_date": commit_data["timestamp"],
                            "relevance_score": score,
                            "extraction_method": "git_show",
                            "code_length": len(extraction_result["code_content"]),
                            "code_content": extraction_result["code_content"]  # Store for other agents
                        }
                    })
                    break  # Found working code, stop searching
                    
            if not result["found"]:
                result["summary"] = f"Analyzed {len(relevant_commits)} relevant commits but could not extract working code"
                result["confidence"] = 0.2
                
        except Exception as e:
            self.logger.error(f"Failed to extract working code: {str(e)}")
            result["summary"] = f"Code extraction failed: {str(e)}"
            result["confidence"] = 0.1
            
        return result
        
    def _extract_feature_keywords(self, query: str) -> List[str]:
        """Extract feature-related keywords from developer query."""
        keywords = []
        
        # Look for feature names
        feature_patterns = [
            r"(?:feature|functionality)\s+(\w+)",
            r"(\w+)\s+(?:feature|functionality)",
            r"the\s+(\w+)\s+(?:working|broken|system)"
        ]
        
        for pattern in feature_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                keywords.append(match.group(1).lower())
                
        # Add common related terms
        if "auth" in query:
            keywords.extend(["auth", "login", "security", "user"])
        if "user" in query:
            keywords.extend(["user", "account", "profile"])
            
        return list(set(keywords))  # Remove duplicates
        
    def _identify_problem_type(self, query: str) -> str:
        """Identify the type of problem from the query."""
        if "deadlock" in query:
            return "deadlock"
        elif "performance" in query or "slow" in query:
            return "performance"
        elif "broken" in query or "not working" in query:
            return "functionality_broken"
        elif "error" in query or "exception" in query:
            return "error"
        else:
            return "general"
            
    def _score_commit_for_working_code(self, commit: Dict[str, Any], keywords: List[str], problem_type: str) -> float:
        """Score how likely a commit contains working code."""
        score = 0.0
        message_lower = commit["message"].lower()
        
        # Positive indicators (working implementation)
        if any(term in message_lower for term in ["implement", "add", "create", "working"]):
            score += 0.3
            
        if any(term in message_lower for term in ["feature", "functionality"]):
            score += 0.2
            
        # Feature keyword matches
        for keyword in keywords:
            if keyword in message_lower:
                score += 0.2
                
        # Problem-specific indicators
        if problem_type == "deadlock":
            if any(term in message_lower for term in ["fix deadlock", "remove lock", "non-blocking"]):
                score += 0.4  # High score for deadlock fixes
            elif any(term in message_lower for term in ["synchronized", "lock", "thread"]):
                score -= 0.2  # Negative score for potential deadlock introduction
                
        # Negative indicators (likely broken or incomplete)
        if any(term in message_lower for term in ["wip", "todo", "fixme", "broken", "temp"]):
            score -= 0.3
            
        if any(term in message_lower for term in ["revert", "rollback"]):
            score -= 0.4
            
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        
    async def _extract_code_from_commit(
        self, 
        git_repo: GitRepository, 
        repository_path: str,
        commit: Dict[str, Any], 
        keywords: List[str]
    ) -> Dict[str, Any]:
        """Extract actual code content from a specific commit."""
        
        result = {
            "success": False,
            "file_path": None,
            "code_content": None,
            "description": ""
        }
        
        try:
            # Get files changed in this commit
            changed_files = git_repo.get_changed_files(commit["sha"])
            
            # Find the most relevant file
            target_file = self._find_most_relevant_file(changed_files, keywords)
            
            if not target_file:
                result["description"] = "No relevant files found in commit"
                return result
                
            # Extract file content at this commit using git show
            code_content = self._extract_file_at_commit(repository_path, commit["sha"], target_file)
            
            if code_content:
                result.update({
                    "success": True,
                    "file_path": target_file,
                    "code_content": code_content,
                    "description": f"Extracted {len(code_content)} characters from {target_file}"
                })
            else:
                result["description"] = f"Could not extract content from {target_file}"
                
        except Exception as e:
            result["description"] = f"Code extraction failed: {str(e)}"
            
        return result
        
    def _find_most_relevant_file(self, changed_files: List[str], keywords: List[str]) -> Optional[str]:
        """Find the most relevant file from changed files."""
        
        scored_files = []
        
        for file_path in changed_files:
            score = 0.0
            file_lower = file_path.lower()
            
            # Score based on keywords
            for keyword in keywords:
                if keyword in file_lower:
                    score += 0.3
                    
            # Score based on file type
            if any(file_path.endswith(ext) for ext in ['.java', '.py', '.js', '.ts', '.cpp', '.c', '.cs']):
                score += 0.2
                
            # Score based on common patterns
            if any(pattern in file_lower for pattern in ['service', 'manager', 'controller', 'handler']):
                score += 0.1
                
            if score > 0:
                scored_files.append((file_path, score))
                
        if scored_files:
            # Return file with highest score
            scored_files.sort(key=lambda x: x[1], reverse=True)
            return scored_files[0][0]
            
        # Fallback: return first code file
        for file_path in changed_files:
            if any(file_path.endswith(ext) for ext in ['.java', '.py', '.js', '.ts', '.cpp', '.c', '.cs']):
                return file_path
                
        return None
        
    def _extract_file_at_commit(self, repository_path: str, commit_sha: str, file_path: str) -> Optional[str]:
        """Extract file content at a specific commit using git show."""
        
        try:
            cmd = ["git", "show", f"{commit_sha}:{file_path}"]
            result = subprocess.run(
                cmd, 
                cwd=repository_path, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.warning(f"Git show failed for {file_path} at {commit_sha}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Git show timed out for {file_path} at {commit_sha}")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting file content: {str(e)}")
            return None