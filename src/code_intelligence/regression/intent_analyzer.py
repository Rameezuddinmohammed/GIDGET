"""Change correlation and intent analysis system."""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict

from ..git.repository import GitRepository
from .models import CommitIntent, ChangeCorrelation, StructuralDiff, BehavioralChange
from .config import get_config

logger = logging.getLogger(__name__)


class IntentAnalyzer:
    """Analyzes commit messages and correlates with actual code changes."""
    
    def __init__(self, git_repo: GitRepository, llm_client: Optional[Any] = None):
        """Initialize with git repository and optional LLM client."""
        self.git_repo = git_repo
        self.llm_client = llm_client
        self.config = get_config()
        self.intent_keywords = self._initialize_intent_keywords()
    
    def analyze_commit_intent(self, commit_sha: str) -> CommitIntent:
        """Analyze the intent of a specific commit."""
        commit_info = self.git_repo.get_commit_info(commit_sha)
        
        # Extract intent using keyword matching
        keyword_intent = self._extract_intent_keywords(commit_info.message)
        
        # Extract intent using LLM if available
        llm_intent = None
        if self.llm_client:
            llm_intent = self._extract_intent_llm(commit_info.message)
        
        # Combine and score intent
        extracted_intent = llm_intent or keyword_intent['primary_intent']
        intent_categories = keyword_intent['categories']
        confidence = keyword_intent['confidence']
        
        if llm_intent and self.llm_client:
            confidence = min(confidence + 0.2, 1.0)  # Boost confidence with LLM
        
        return CommitIntent(
            commit_sha=commit_sha,
            message=commit_info.message,
            extracted_intent=extracted_intent,
            intent_categories=intent_categories,
            confidence=confidence,
            keywords=keyword_intent['keywords']
        )
    
    def correlate_changes_with_intent(
        self, 
        commit_sha: str, 
        version_comparison
    ) -> ChangeCorrelation:
        """Correlate actual code changes with stated commit intent."""
        commit_intent = self.analyze_commit_intent(commit_sha)
        
        # Analyze actual changes
        actual_changes = self._summarize_actual_changes(version_comparison)
        
        # Calculate correlation score
        correlation_score = self._calculate_correlation_score(
            commit_intent, actual_changes
        )
        
        # Identify discrepancies
        discrepancies = self._identify_discrepancies(
            commit_intent, actual_changes
        )
        
        # Generate alignment analysis
        alignment_analysis = self._generate_alignment_analysis(
            commit_intent, actual_changes, correlation_score
        )
        
        return ChangeCorrelation(
            commit_sha=commit_sha,
            stated_intent=commit_intent.extracted_intent,
            actual_changes=actual_changes,
            correlation_score=correlation_score,
            discrepancies=discrepancies,
            alignment_analysis=alignment_analysis
        )    

    def _initialize_intent_keywords(self) -> Dict[str, Dict[str, Any]]:
        """Initialize keyword patterns for intent extraction."""
        return {
            'fix': {
                'keywords': ['fix', 'bug', 'error', 'issue', 'problem', 'resolve', 'correct'],
                'patterns': [r'fix\s+#?\d+', r'bug\s+fix', r'resolve\s+issue'],
                'weight': self.config.intent_weights['fix']
            },
            'feature': {
                'keywords': ['add', 'new', 'feature', 'implement', 'create', 'introduce'],
                'patterns': [r'add\s+\w+', r'new\s+feature', r'implement\s+\w+'],
                'weight': self.config.intent_weights['feature']
            },
            'refactor': {
                'keywords': ['refactor', 'cleanup', 'reorganize', 'restructure', 'improve'],
                'patterns': [r'refactor\s+\w+', r'clean\s+up', r'improve\s+\w+'],
                'weight': self.config.intent_weights['refactor']
            },
            'performance': {
                'keywords': ['optimize', 'performance', 'speed', 'faster', 'efficient'],
                'patterns': [r'optimize\s+\w+', r'improve\s+performance', r'speed\s+up'],
                'weight': self.config.intent_weights['performance']
            },
            'documentation': {
                'keywords': ['doc', 'comment', 'readme', 'documentation', 'explain'],
                'patterns': [r'update\s+doc', r'add\s+comment', r'improve\s+doc'],
                'weight': self.config.intent_weights['documentation']
            },
            'test': {
                'keywords': ['test', 'spec', 'coverage', 'unittest', 'integration'],
                'patterns': [r'add\s+test', r'test\s+\w+', r'improve\s+coverage'],
                'weight': self.config.intent_weights['test']
            }
        }
    
    def _extract_intent_keywords(self, commit_message: str) -> Dict[str, Any]:
        """Extract intent using keyword matching."""
        message_lower = commit_message.lower()
        
        # Score each intent category
        category_scores = {}
        matched_keywords = []
        
        for category, config in self.intent_keywords.items():
            score = 0.0
            
            # Check keywords
            for keyword in config['keywords']:
                if keyword in message_lower:
                    score += config['weight'] * 0.5
                    matched_keywords.append(keyword)
            
            # Check patterns
            for pattern in config['patterns']:
                if re.search(pattern, message_lower):
                    score += config['weight'] * 0.8
            
            if score > 0:
                category_scores[category] = score
        
        # Determine primary intent
        if category_scores:
            primary_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            primary_intent = self._generate_intent_description(primary_category, commit_message)
            confidence = min(category_scores[primary_category], 1.0)
        else:
            primary_intent = "General code change"
            confidence = 0.3
        
        return {
            'primary_intent': primary_intent,
            'categories': list(category_scores.keys()),
            'confidence': confidence,
            'keywords': matched_keywords
        }
    
    def _extract_intent_llm(self, commit_message: str) -> Optional[str]:
        """Extract intent using LLM analysis."""
        if not self.llm_client:
            return None
        
        prompt = f"""
        Analyze this commit message and extract the developer's intent in one clear sentence:
        
        Commit message: "{commit_message}"
        
        Focus on:
        - What the developer was trying to accomplish
        - The business or technical goal
        - The type of change (fix, feature, refactor, etc.)
        
        Respond with a single sentence describing the intent.
        """
        
        try:
            # Check if LLM client has async or sync interface
            if hasattr(self.llm_client, 'chat_completion'):
                # Async interface - would need to be called from async context
                logger.info("LLM client available but requires async context")
                return None
            elif hasattr(self.llm_client, 'generate_response'):
                # Sync interface
                response = self.llm_client.generate_response(prompt)
                return response.strip()
            else:
                logger.warning("LLM client interface not recognized")
                return None
        except Exception as e:
            logger.warning(f"LLM intent extraction failed: {e}")
            return None
    
    def _generate_intent_description(self, category: str, commit_message: str) -> str:
        """Generate a human-readable intent description."""
        descriptions = {
            'fix': "Fix a bug or resolve an issue",
            'feature': "Add new functionality or feature",
            'refactor': "Refactor or improve code structure",
            'performance': "Optimize performance or efficiency",
            'documentation': "Update documentation or comments",
            'test': "Add or improve tests"
        }
        
        base_description = descriptions.get(category, "Make a code change")
        
        # Try to extract specific details from commit message
        first_line = commit_message.split('\n')[0].strip()
        if len(first_line) > 10 and len(first_line) < 100:
            return f"{base_description}: {first_line}"
        
        return base_description
    
    def _summarize_actual_changes(self, version_comparison) -> List[str]:
        """Summarize actual code changes from version comparison."""
        changes = []
        
        # Summarize structural changes
        for diff in version_comparison.structural_diffs:
            change_desc = f"{diff.change_type.value} {diff.element_type} '{diff.element_name}'"
            if diff.file_path:
                change_desc += f" in {diff.file_path}"
            changes.append(change_desc)
        
        # Summarize behavioral changes
        for behavior_change in version_comparison.behavioral_changes:
            changes.append(f"Modified behavior of {behavior_change.element_name}: {behavior_change.change_description}")
        
        # Summarize dependency impacts
        for impact in version_comparison.dependency_impacts:
            changes.append(f"Dependency impact: {impact.source_element} affects {len(impact.affected_elements)} elements")
        
        return changes
    
    def _calculate_correlation_score(
        self, 
        commit_intent: CommitIntent, 
        actual_changes: List[str]
    ) -> float:
        """Calculate correlation score between intent and actual changes."""
        if not actual_changes:
            return 0.0
        
        intent_lower = commit_intent.extracted_intent.lower()
        changes_text = ' '.join(actual_changes).lower()
        
        # Check for direct keyword matches
        keyword_matches = 0
        for keyword in commit_intent.keywords:
            if keyword in changes_text:
                keyword_matches += 1
        
        keyword_score = min(keyword_matches / max(len(commit_intent.keywords), 1), 1.0)
        
        # Check for category alignment
        category_score = 0.0
        
        for category in commit_intent.intent_categories:
            match_score = self.config.correlation_weights['category_match_score']
            if category == 'fix' and any(word in changes_text for word in ['modified', 'removed', 'error']):
                category_score += match_score
            elif category == 'feature' and any(word in changes_text for word in ['added', 'new', 'create']):
                category_score += match_score
            elif category == 'refactor' and any(word in changes_text for word in ['modified', 'restructure']):
                category_score += match_score
            elif category == 'performance' and any(word in changes_text for word in ['optimize', 'efficient']):
                category_score += match_score
            elif category == 'test' and any(word in changes_text for word in ['test', 'spec']):
                category_score += match_score
        
        category_score = min(category_score, 1.0)
        
        # Combine scores using configuration weights
        correlation_score = (
            keyword_score * self.config.correlation_weights['keyword_weight'] + 
            category_score * self.config.correlation_weights['category_weight']
        )
        
        # Adjust based on intent confidence
        correlation_score *= commit_intent.confidence
        
        return correlation_score
    
    def _identify_discrepancies(
        self, 
        commit_intent: CommitIntent, 
        actual_changes: List[str]
    ) -> List[str]:
        """Identify discrepancies between intent and actual changes."""
        discrepancies = []
        
        intent_lower = commit_intent.extracted_intent.lower()
        changes_text = ' '.join(actual_changes).lower()
        
        # Check for intent-change mismatches
        if 'fix' in intent_lower and 'added' in changes_text and 'removed' not in changes_text:
            discrepancies.append("Intent suggests bug fix, but only additions were made")
        
        if 'add' in intent_lower and 'removed' in changes_text and 'added' not in changes_text:
            discrepancies.append("Intent suggests addition, but only removals were made")
        
        if 'refactor' in intent_lower and ('added' in changes_text or 'removed' in changes_text):
            if not ('modified' in changes_text):
                discrepancies.append("Intent suggests refactoring, but structural changes were made")
        
        return discrepancies
    
    def _generate_alignment_analysis(
        self, 
        commit_intent: CommitIntent, 
        actual_changes: List[str], 
        correlation_score: float
    ) -> str:
        """Generate alignment analysis between intent and changes."""
        if correlation_score >= 0.8:
            alignment = "Strong alignment"
        elif correlation_score >= 0.6:
            alignment = "Good alignment"
        elif correlation_score >= 0.4:
            alignment = "Moderate alignment"
        elif correlation_score >= 0.2:
            alignment = "Weak alignment"
        else:
            alignment = "Poor alignment"
        
        analysis = f"{alignment} between stated intent and actual changes (score: {correlation_score:.2f}). "
        
        if correlation_score >= 0.6:
            analysis += "The code changes appear to match the developer's stated intent well."
        elif correlation_score >= 0.4:
            analysis += "The code changes partially match the stated intent, with some discrepancies."
        else:
            analysis += "The code changes do not clearly match the stated intent."
        
        return analysis 
   
    def analyze_developer_communication(
        self, 
        commit_range: List[str]
    ) -> Dict[str, List[str]]:
        """Analyze developer communication patterns across commits."""
        communication_patterns = defaultdict(list)
        
        for commit_sha in commit_range:
            commit_info = self.git_repo.get_commit_info(commit_sha)
            
            # Analyze commit message patterns
            patterns = self._extract_communication_patterns(commit_info.message)
            
            for pattern_type, pattern_data in patterns.items():
                communication_patterns[pattern_type].extend(pattern_data)
        
        return dict(communication_patterns)
    
    def reconstruct_change_rationale(
        self, 
        element_name: str, 
        commit_range: List[str]
    ) -> Dict[str, str]:
        """Reconstruct the rationale for changes to a specific element."""
        rationale = {}
        
        for commit_sha in commit_range:
            commit_info = self.git_repo.get_commit_info(commit_sha)
            
            # Check if this commit affects the element
            if self._commit_affects_element(commit_info, element_name):
                intent = self.analyze_commit_intent(commit_sha)
                rationale[commit_sha] = {
                    'intent': intent.extracted_intent,
                    'message': commit_info.message,
                    'confidence': intent.confidence,
                    'timestamp': commit_info.committed_date.isoformat()
                }
        
        return rationale
    
    def _extract_communication_patterns(self, commit_message: str) -> Dict[str, List[str]]:
        """Extract communication patterns from commit message."""
        patterns = defaultdict(list)
        
        # Check for issue references
        issue_refs = re.findall(r'#(\d+)', commit_message)
        if issue_refs:
            patterns['issue_references'].extend(issue_refs)
        
        # Check for urgency indicators
        urgency_words = ['urgent', 'critical', 'hotfix', 'emergency', 'asap']
        if any(word in commit_message.lower() for word in urgency_words):
            patterns['urgency'].append('high_priority')
        
        # Check for breaking change indicators
        if any(word in commit_message.lower() for word in ['breaking', 'breaking change', 'major']):
            patterns['breaking_changes'].append('potential_breaking')
        
        return dict(patterns)
    
    def _commit_affects_element(self, commit_info, element_name: str) -> bool:
        """Check if a commit affects a specific code element."""
        # Check commit message
        if element_name in commit_info.message:
            return True
        
        # Check file changes
        for file_change in commit_info.file_changes:
            if element_name in file_change.file_path:
                return True
        
        return False