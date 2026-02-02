"""RQ2: Information diffusion analysis module.

Implements cascade identification, diffusion modeling, and contagion type
classification for memes, skills, and behavioral patterns.
"""

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .storage import JSONStorage
from .config import Config
from .models import Cascade

logger = logging.getLogger(__name__)


class CascadeIdentifier:
    """Identifies information cascades in MoltBook data."""
    
    def __init__(
        self,
        storage: JSONStorage,
        config: Config,
    ) -> None:
        """Initialize cascade identifier.
        
        Args:
            storage: JSONStorage instance.
            config: Configuration parameters.
        """
        self.storage = storage
        self.config = config
    
    def identify_meme_cascades(
        self,
        min_adopters: int = 5,
        ngram_range: tuple = (2, 5),
    ) -> list[Cascade]:
        """Identify meme cascades using n-gram analysis.
        
        Args:
            min_adopters: Minimum number of unique adopters.
            ngram_range: Range of n-gram sizes to analyze.
            
        Returns:
            List of identified meme cascades.
        """
        posts = self.storage.get_posts()
        comments = self.storage.get_comments()
        
        # Collect all content with timestamps and authors
        content_items = []
        for post in posts:
            if post.body and post.created_at:
                content_items.append({
                    'text': post.body,
                    'author': post.author_id,
                    'timestamp': post.created_at,
                    'type': 'post',
                })
        for comment in comments:
            if comment.body and comment.created_at:
                content_items.append({
                    'text': comment.body,
                    'author': comment.author_id,
                    'timestamp': comment.created_at,
                    'type': 'comment',
                })
        
        if not content_items:
            return []
        
        # Extract n-grams and track adoptions
        ngram_adoptions = defaultdict(list)  # ngram -> [(author, timestamp), ...]
        
        for item in content_items:
            text = item['text'].lower()
            words = re.findall(r'\b\w+\b', text)
            
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    ngram_adoptions[ngram].append((item['author'], item['timestamp']))
        
        # Filter to cascades with minimum adopters
        cascades = []
        for ngram, adoptions in ngram_adoptions.items():
            # Get unique adopters
            unique_adopters = set(a[0] for a in adoptions)
            
            if len(unique_adopters) >= min_adopters:
                # Sort by timestamp
                sorted_adoptions = sorted(adoptions, key=lambda x: x[1])
                
                # Create cascade
                cascade = Cascade(
                    cascade_id=hashlib.md5(ngram.encode()).hexdigest()[:16],
                    cascade_type='meme',
                    seed_agent=sorted_adoptions[0][0],
                    seed_time=sorted_adoptions[0][1],
                    adoptions=sorted_adoptions,
                    content_hash=ngram,
                )
                cascades.append(cascade)
        
        logger.info(f"Identified {len(cascades)} meme cascades with >= {min_adopters} adopters")
        return cascades

    
    def identify_skill_cascades(
        self,
        min_adopters: int = 5,
    ) -> list[Cascade]:
        """Identify skill cascades from code/module sharing.
        
        Args:
            min_adopters: Minimum number of unique adopters.
            
        Returns:
            List of identified skill cascades.
        """
        posts = self.storage.get_posts()
        comments = self.storage.get_comments()
        
        # Extract code blocks and track adoptions
        code_adoptions = defaultdict(list)
        
        code_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        
        for post in posts:
            if post.body and post.created_at:
                code_blocks = code_pattern.findall(post.body)
                for code in code_blocks:
                    # Hash the code content
                    code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
                    code_adoptions[code_hash].append((post.author_id, post.created_at))
        
        for comment in comments:
            if comment.body and comment.created_at:
                code_blocks = code_pattern.findall(comment.body)
                for code in code_blocks:
                    code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
                    code_adoptions[code_hash].append((comment.author_id, comment.created_at))
        
        # Filter to cascades with minimum adopters
        cascades = []
        for code_hash, adoptions in code_adoptions.items():
            unique_adopters = set(a[0] for a in adoptions)
            
            if len(unique_adopters) >= min_adopters:
                sorted_adoptions = sorted(adoptions, key=lambda x: x[1])
                
                cascade = Cascade(
                    cascade_id=code_hash,
                    cascade_type='skill',
                    seed_agent=sorted_adoptions[0][0],
                    seed_time=sorted_adoptions[0][1],
                    adoptions=sorted_adoptions,
                    content_hash=code_hash,
                )
                cascades.append(cascade)
        
        logger.info(f"Identified {len(cascades)} skill cascades")
        return cascades
    
    def identify_behavioral_cascades(
        self,
        min_adopters: int = 5,
    ) -> list[Cascade]:
        """Identify behavioral cascades from formatting styles.
        
        Args:
            min_adopters: Minimum number of unique adopters.
            
        Returns:
            List of identified behavioral cascades.
        """
        posts = self.storage.get_posts()
        comments = self.storage.get_comments()
        
        # Track formatting patterns
        pattern_adoptions = defaultdict(list)
        
        # Define behavioral patterns to track
        patterns = {
            'emoji_heavy': re.compile(r'[\U0001F600-\U0001F64F]{3,}'),
            'bullet_list': re.compile(r'^\s*[-*]\s+', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+', re.MULTILINE),
            'header_style': re.compile(r'^#+\s+', re.MULTILINE),
            'quote_style': re.compile(r'^>\s+', re.MULTILINE),
            'link_heavy': re.compile(r'\[.*?\]\(.*?\)'),
        }
        
        for post in posts:
            if post.body and post.created_at:
                for pattern_name, pattern in patterns.items():
                    if pattern.search(post.body):
                        pattern_adoptions[pattern_name].append(
                            (post.author_id, post.created_at)
                        )
        
        for comment in comments:
            if comment.body and comment.created_at:
                for pattern_name, pattern in patterns.items():
                    if pattern.search(comment.body):
                        pattern_adoptions[pattern_name].append(
                            (comment.author_id, comment.created_at)
                        )
        
        # Filter to cascades with minimum adopters
        cascades = []
        for pattern_name, adoptions in pattern_adoptions.items():
            unique_adopters = set(a[0] for a in adoptions)
            
            if len(unique_adopters) >= min_adopters:
                sorted_adoptions = sorted(adoptions, key=lambda x: x[1])
                
                cascade = Cascade(
                    cascade_id=f"behavioral_{pattern_name}",
                    cascade_type='behavioral',
                    seed_agent=sorted_adoptions[0][0],
                    seed_time=sorted_adoptions[0][1],
                    adoptions=sorted_adoptions,
                    content_hash=pattern_name,
                )
                cascades.append(cascade)
        
        logger.info(f"Identified {len(cascades)} behavioral cascades")
        return cascades


class DiffusionModeler:
    """Models information diffusion dynamics."""
    
    def __init__(
        self,
        cascades: list[Cascade],
        network: nx.DiGraph,
    ) -> None:
        """Initialize diffusion modeler.
        
        Args:
            cascades: List of identified cascades.
            network: Interaction network.
        """
        self.cascades = cascades
        self.network = network
    
    def compute_exposures(self, cascade: Cascade) -> pd.DataFrame:
        """Compute exposure counts for each agent in a cascade.
        
        Args:
            cascade: The cascade to analyze.
            
        Returns:
            DataFrame with agent, exposure_count, adopted, adoption_time.
        """
        # Track who has adopted and when
        adopted_agents = {}
        for agent, timestamp in cascade.adoptions:
            if agent not in adopted_agents:
                adopted_agents[agent] = timestamp
        
        # Compute exposures for each agent
        exposure_data = []
        
        for agent in self.network.nodes():
            # Count neighbors who adopted before this agent
            exposure_count = 0
            
            for neighbor in self.network.predecessors(agent):
                if neighbor in adopted_agents:
                    if agent in adopted_agents:
                        # Only count if neighbor adopted before this agent
                        if adopted_agents[neighbor] < adopted_agents[agent]:
                            exposure_count += 1
                    else:
                        exposure_count += 1
            
            exposure_data.append({
                'agent': agent,
                'exposure_count': exposure_count,
                'adopted': agent in adopted_agents,
                'adoption_time': adopted_agents.get(agent),
            })
        
        return pd.DataFrame(exposure_data)

    
    def fit_logistic_model(self) -> dict:
        """Fit logistic regression model for adoption probability.
        
        Returns:
            Dict with model coefficients and statistics.
        """
        from sklearn.linear_model import LogisticRegression
        
        # Aggregate exposure data across all cascades
        all_exposures = []
        
        for cascade in self.cascades:
            exposure_df = self.compute_exposures(cascade)
            exposure_df['cascade_id'] = cascade.cascade_id
            all_exposures.append(exposure_df)
        
        if not all_exposures:
            return {}
        
        df = pd.concat(all_exposures, ignore_index=True)
        
        # Prepare features: linear and quadratic exposure
        X = df[['exposure_count']].copy()
        X['exposure_squared'] = X['exposure_count'] ** 2
        y = df['adopted'].astype(int)
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            logger.warning("Insufficient class diversity for logistic regression")
            return {}
        
        # Fit model
        try:
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            return {
                'intercept': model.intercept_[0],
                'beta_linear': model.coef_[0][0],
                'beta_quadratic': model.coef_[0][1],
                'n_observations': len(df),
            }
        except Exception as e:
            logger.warning(f"Logistic regression failed: {e}")
            return {}
    
    def fit_cox_hazards(self) -> dict:
        """Fit Cox proportional hazards model for adoption timing.
        
        Returns:
            Dict with model coefficients and statistics.
        """
        try:
            from lifelines import CoxPHFitter
            
            # Prepare survival data
            survival_data = []
            
            for cascade in self.cascades:
                exposure_df = self.compute_exposures(cascade)
                
                for _, row in exposure_df.iterrows():
                    if row['adopted'] and row['adoption_time']:
                        # Time to adoption from cascade start
                        duration = (row['adoption_time'] - cascade.seed_time).total_seconds() / 3600
                        if duration > 0:
                            survival_data.append({
                                'duration': duration,
                                'event': 1,
                                'exposure': row['exposure_count'],
                            })
            
            if not survival_data:
                return {}
            
            df = pd.DataFrame(survival_data)
            
            cph = CoxPHFitter()
            cph.fit(df, duration_col='duration', event_col='event')
            
            return {
                'hazard_ratio': np.exp(cph.params_['exposure']),
                'coefficient': cph.params_['exposure'],
                'p_value': cph.summary['p']['exposure'],
            }
        except ImportError:
            logger.warning("lifelines not available for Cox model")
            return {}
    
    def test_contagion_type(self) -> str:
        """Classify contagion as simple or complex.
        
        Simple contagion: Linear exposure effect (β₂ ≈ 0)
        Complex contagion: Quadratic exposure effect (β₂ > 0, significant)
        
        Returns:
            'simple' or 'complex'
        """
        model_results = self.fit_logistic_model()
        
        if not model_results:
            return 'unknown'
        
        beta_quadratic = model_results.get('beta_quadratic', 0)
        
        # Simple heuristic: if quadratic term is positive and substantial
        # A more rigorous test would use statistical significance
        if beta_quadratic > 0.01:
            return 'complex'
        else:
            return 'simple'


class CascadeAnalyzer:
    """Analyzes cascade statistics and distributions."""
    
    def __init__(self, cascades: list[Cascade]) -> None:
        """Initialize cascade analyzer.
        
        Args:
            cascades: List of cascades to analyze.
        """
        self.cascades = cascades
    
    def compute_cascade_statistics(self) -> pd.DataFrame:
        """Compute statistics for each cascade.
        
        Returns:
            DataFrame with cascade metrics.
        """
        stats_list = []
        
        for cascade in self.cascades:
            unique_adopters = set(a[0] for a in cascade.adoptions)
            
            if cascade.adoptions:
                timestamps = [a[1] for a in cascade.adoptions]
                duration = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            else:
                duration = 0
            
            stats_list.append({
                'cascade_id': cascade.cascade_id,
                'cascade_type': cascade.cascade_type,
                'n_adoptions': len(cascade.adoptions),
                'n_unique_adopters': len(unique_adopters),
                'duration_hours': duration,
                'seed_agent': cascade.seed_agent,
            })
        
        return pd.DataFrame(stats_list)
    
    def test_power_law(self) -> dict:
        """Test if cascade sizes follow a power-law distribution.
        
        Uses Clauset methodology for power-law testing.
        
        Returns:
            Dict with test results.
        """
        sizes = [len(set(c.adoptions)) for c in self.cascades]
        
        if len(sizes) < 10:
            return {'result': 'insufficient_data'}
        
        sizes = np.array(sizes)
        
        # Estimate power-law exponent using MLE
        # α = 1 + n / Σ ln(x_i / x_min)
        x_min = sizes.min()
        n = len(sizes)
        
        if x_min <= 0:
            return {'result': 'invalid_data'}
        
        alpha = 1 + n / np.sum(np.log(sizes / x_min))
        
        # Kolmogorov-Smirnov test against fitted power-law
        # Generate theoretical CDF
        sorted_sizes = np.sort(sizes)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = 1 - (x_min / sorted_sizes) ** (alpha - 1)
        
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        return {
            'alpha': alpha,
            'x_min': x_min,
            'ks_statistic': ks_stat,
            'n_cascades': n,
        }
    
    def compare_distributions(self) -> dict:
        """Compare cascade size distributions across types.
        
        Returns:
            Dict with comparison statistics.
        """
        type_sizes = defaultdict(list)
        
        for cascade in self.cascades:
            size = len(set(a[0] for a in cascade.adoptions))
            type_sizes[cascade.cascade_type].append(size)
        
        results = {}
        
        types = list(type_sizes.keys())
        for i, type1 in enumerate(types):
            for type2 in types[i+1:]:
                sizes1 = type_sizes[type1]
                sizes2 = type_sizes[type2]
                
                if len(sizes1) >= 5 and len(sizes2) >= 5:
                    stat, p_value = stats.mannwhitneyu(sizes1, sizes2)
                    results[f'{type1}_vs_{type2}'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'mean_1': np.mean(sizes1),
                        'mean_2': np.mean(sizes2),
                    }
        
        return results


def verify_cascade_ordering(cascade: Cascade) -> bool:
    """Verify that cascade adoptions are in chronological order.
    
    Args:
        cascade: Cascade to verify.
        
    Returns:
        True if adoptions are monotonically ordered by timestamp.
    """
    if len(cascade.adoptions) <= 1:
        return True
    
    for i in range(len(cascade.adoptions) - 1):
        if cascade.adoptions[i][1] > cascade.adoptions[i + 1][1]:
            return False
    
    return True
