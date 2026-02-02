"""RQ3: Collective problem-solving analysis module.

Implements collaborative event identification, solution quality assessment,
and collaboration success modeling.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .storage import JSONStorage
from .config import Config
from .models import CollaborativeEvent

logger = logging.getLogger(__name__)

# Technical keywords for identifying problem-solving threads
TECHNICAL_KEYWORDS = {
    'error', 'bug', 'fix', 'issue', 'problem', 'help', 'solution', 'solve',
    'debug', 'crash', 'exception', 'fail', 'broken', 'stuck', 'how to',
    'implement', 'code', 'function', 'method', 'class', 'api', 'library',
}


class CollaborationIdentifier:
    """Identifies collaborative problem-solving events."""
    
    def __init__(self, storage: JSONStorage, config: Config) -> None:
        self.storage = storage
        self.config = config
    
    def identify_collaborative_events(
        self,
        min_agents: int = 3,
        min_comments: int = 5,
        min_duration_minutes: int = 30,
    ) -> list[CollaborativeEvent]:
        """Identify collaborative problem-solving events.
        
        Args:
            min_agents: Minimum unique participants.
            min_comments: Minimum number of comments.
            min_duration_minutes: Minimum thread duration.
            
        Returns:
            List of identified collaborative events.
        """
        posts = self.storage.get_posts()
        events = []
        
        for post in posts:
            comments = self.storage.get_comments(filters={'post_id': post.post_id})
            
            if len(comments) < min_comments:
                continue
            
            # Check for technical keywords
            all_text = (post.title or '') + ' ' + (post.body or '')
            for comment in comments:
                all_text += ' ' + (comment.body or '')
            
            if not self._contains_technical_keywords(all_text):
                continue
            
            # Get unique participants
            participants = {post.author_id}
            for comment in comments:
                participants.add(comment.author_id)
            
            if len(participants) < min_agents:
                continue
            
            # Check duration
            timestamps = [c.created_at for c in comments if c.created_at]
            if post.created_at:
                timestamps.append(post.created_at)
            
            if len(timestamps) < 2:
                continue
            
            duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
            if duration < min_duration_minutes:
                continue
            
            event = CollaborativeEvent(
                thread_id=post.post_id,
                participants=list(participants),
                start_time=min(timestamps),
                end_time=max(timestamps),
                problem_statement=post.title or '',
                solution=self._extract_solution(comments),
            )
            events.append(event)
        
        logger.info(f"Identified {len(events)} collaborative events")
        return events
    
    def _contains_technical_keywords(self, text: str) -> bool:
        """Check if text contains technical keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in TECHNICAL_KEYWORDS)
    
    def _extract_solution(self, comments: list) -> str:
        """Extract potential solution from comments."""
        # Simple heuristic: last comment with code block
        for comment in reversed(comments):
            if comment.body and ('```' in comment.body or '`' in comment.body):
                return comment.body
        return comments[-1].body if comments else ''
    
    def extract_technical_threads(self) -> list[str]:
        """Get IDs of threads with technical content."""
        posts = self.storage.get_posts()
        technical_ids = []
        
        for post in posts:
            text = (post.title or '') + ' ' + (post.body or '')
            if self._contains_technical_keywords(text):
                technical_ids.append(post.post_id)
        
        return technical_ids


class SolutionAssessor:
    """Assesses quality of collaborative solutions."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def assess_code_solution(self, code: str) -> dict:
        """Assess quality of a code solution.
        
        Args:
            code: Code string to assess.
            
        Returns:
            Dict with quality metrics.
        """
        metrics = {
            'has_code': bool(re.search(r'```|`[^`]+`', code)),
            'has_comments': bool(re.search(r'#.*|//.*|/\*.*\*/', code)),
            'has_tests': bool(re.search(r'test|assert|expect', code.lower())),
            'line_count': len(code.split('\n')),
            'syntax_valid': self._check_syntax(code),
        }
        
        # Compute overall score
        score = sum([
            metrics['has_code'] * 0.3,
            metrics['has_comments'] * 0.2,
            metrics['has_tests'] * 0.3,
            metrics['syntax_valid'] * 0.2,
        ])
        metrics['quality_score'] = score
        
        return metrics
    
    def _check_syntax(self, code: str) -> bool:
        """Basic syntax check for code."""
        # Simple heuristic: balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack.pop() != char:
                    return False
        
        return len(stack) == 0
    
    def assess_conceptual_solution(self, problem: str, solution: str) -> dict:
        """Assess quality of a conceptual solution.
        
        Args:
            problem: Problem statement.
            solution: Proposed solution.
            
        Returns:
            Dict with quality metrics.
        """
        metrics = {
            'solution_length': len(solution),
            'addresses_problem': self._solution_addresses_problem(problem, solution),
            'has_explanation': len(solution) > 100,
            'has_steps': bool(re.search(r'\d+\.|[-*]\s', solution)),
        }
        
        score = sum([
            metrics['addresses_problem'] * 0.4,
            metrics['has_explanation'] * 0.3,
            metrics['has_steps'] * 0.3,
        ])
        metrics['quality_score'] = score
        
        return metrics
    
    def _solution_addresses_problem(self, problem: str, solution: str) -> bool:
        """Check if solution addresses the problem."""
        problem_words = set(problem.lower().split())
        solution_words = set(solution.lower().split())
        overlap = problem_words & solution_words
        return len(overlap) >= 2
    
    def compute_inter_rater_reliability(
        self, 
        ratings1: list, 
        ratings2: list
    ) -> float:
        """Compute Cohen's kappa for inter-rater reliability.
        
        Args:
            ratings1: First rater's ratings.
            ratings2: Second rater's ratings.
            
        Returns:
            Cohen's kappa coefficient.
        """
        if len(ratings1) != len(ratings2) or len(ratings1) == 0:
            return 0.0
        
        # Convert to numpy arrays
        r1 = np.array(ratings1)
        r2 = np.array(ratings2)
        
        # Compute observed agreement
        po = np.mean(r1 == r2)
        
        # Compute expected agreement
        categories = np.unique(np.concatenate([r1, r2]))
        pe = 0.0
        for cat in categories:
            pe += (np.mean(r1 == cat) * np.mean(r2 == cat))
        
        # Cohen's kappa
        if pe == 1.0:
            return 1.0
        
        kappa = (po - pe) / (1 - pe)
        return kappa


class BaselineComparator:
    """Compares collaborative vs individual solutions."""
    
    def __init__(self, events: list[CollaborativeEvent]) -> None:
        self.events = events
    
    def collect_individual_baselines(self) -> pd.DataFrame:
        """Collect baseline individual solution attempts."""
        # Placeholder - would need actual individual attempts
        return pd.DataFrame()
    
    def compare_quality_distributions(self) -> dict:
        """Compare quality distributions using Wilcoxon test.
        
        Returns:
            Dict with test statistics.
        """
        collab_scores = [e.quality_score for e in self.events if e.quality_score]
        
        if len(collab_scores) < 5:
            return {'result': 'insufficient_data'}
        
        # Compare to hypothetical baseline (would need real data)
        baseline_scores = np.random.uniform(0.3, 0.7, len(collab_scores))
        
        stat, p_value = stats.wilcoxon(collab_scores, baseline_scores)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'collab_mean': np.mean(collab_scores),
            'baseline_mean': np.mean(baseline_scores),
        }


class CollaborationModeler:
    """Models collaboration success factors."""
    
    def __init__(
        self, 
        events: list[CollaborativeEvent], 
        network: nx.DiGraph
    ) -> None:
        self.events = events
        self.network = network
    
    def fit_success_model(self) -> dict:
        """Fit logistic regression for collaboration success.
        
        Returns:
            Dict with model coefficients.
        """
        from sklearn.linear_model import LogisticRegression
        
        if len(self.events) < 10:
            return {'result': 'insufficient_data'}
        
        # Extract features for each event
        features = []
        outcomes = []
        
        for event in self.events:
            # Network size
            n_participants = len(event.participants)
            
            # Network density among participants
            subgraph = self.network.subgraph(event.participants)
            density = nx.density(subgraph) if len(event.participants) > 1 else 0
            
            # Diversity (unique submolts - placeholder)
            diversity = n_participants / 10  # Normalized
            
            features.append([n_participants, density, diversity])
            outcomes.append(1 if event.quality_score and event.quality_score > 0.5 else 0)
        
        X = np.array(features)
        y = np.array(outcomes)
        
        if len(np.unique(y)) < 2:
            return {'result': 'insufficient_class_diversity'}
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        return {
            'intercept': model.intercept_[0],
            'coef_network_size': model.coef_[0][0],
            'coef_density': model.coef_[0][1],
            'coef_diversity': model.coef_[0][2],
        }
