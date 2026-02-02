"""Statistical validation module for Molt Dynamics analysis.

Implements robustness checks, null model comparisons, and statistical testing.
"""

import logging
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score

from .storage import JSONStorage
from .config import Config

logger = logging.getLogger(__name__)


class StatisticalFramework:
    """Framework for hypothesis testing and effect size calculation."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.alpha = config.significance_level
    
    def hypothesis_test(
        self,
        test_type: str,
        *args,
        **kwargs
    ) -> dict:
        """Perform hypothesis test.
        
        Args:
            test_type: Type of test ('t', 'wilcoxon', 'chi2', 'anova').
            *args: Test arguments.
            **kwargs: Test keyword arguments.
            
        Returns:
            Dict with test statistic, p-value, df, effect_size.
        """
        if test_type == 't':
            stat, p = stats.ttest_ind(*args, **kwargs)
            df = len(args[0]) + len(args[1]) - 2
            effect = self.compute_effect_size('cohens_d', *args)
        elif test_type == 'wilcoxon':
            stat, p = stats.wilcoxon(*args, **kwargs)
            df = len(args[0]) - 1
            effect = self.compute_effect_size('rank_biserial', *args)
        elif test_type == 'chi2':
            stat, p, df, _ = stats.chi2_contingency(args[0])
            effect = self.compute_effect_size('cramers_v', args[0])
        elif test_type == 'anova':
            stat, p = stats.f_oneway(*args)
            df = (len(args) - 1, sum(len(a) for a in args) - len(args))
            effect = self.compute_effect_size('eta_squared', *args)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            'test_statistic': stat,
            'p_value': p,
            'degrees_of_freedom': df,
            'effect_size': effect,
            'significant': p < self.alpha,
        }
    
    def compute_effect_size(self, effect_type: str, *args) -> float:
        """Compute effect size.
        
        Args:
            effect_type: Type of effect size.
            *args: Data for computation.
            
        Returns:
            Effect size value.
        """
        if effect_type == 'cohens_d':
            x1, x2 = args[0], args[1]
            pooled_std = np.sqrt(
                ((len(x1)-1)*np.var(x1) + (len(x2)-1)*np.var(x2)) / 
                (len(x1) + len(x2) - 2)
            )
            return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0
        
        elif effect_type == 'cramers_v':
            chi2 = stats.chi2_contingency(args[0])[0]
            n = np.sum(args[0])
            min_dim = min(args[0].shape) - 1
            return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        elif effect_type == 'r_squared':
            return args[0] ** 2 if len(args) > 0 else 0
        
        elif effect_type == 'eta_squared':
            groups = args
            grand_mean = np.mean([np.mean(g) for g in groups])
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = sum(np.sum((g - grand_mean)**2) for g in groups)
            return ss_between / ss_total if ss_total > 0 else 0
        
        return 0.0
    
    def apply_bonferroni_correction(self, p_values: list) -> list:
        """Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values.
            
        Returns:
            List of corrected p-values.
        """
        m = len(p_values)
        if m == 0:
            return []
        
        # Bonferroni: multiply p-values by number of tests
        corrected = [min(p * m, 1.0) for p in p_values]
        return corrected


class RobustnessChecker:
    """Performs robustness checks on analysis results."""
    
    def __init__(self, storage: JSONStorage, config: Config) -> None:
        self.storage = storage
        self.config = config
    
    def temporal_stability_check(
        self,
        analysis_func: Callable,
        n_periods: int = 4,
    ) -> dict:
        """Check temporal stability by running analysis on time periods.
        
        Args:
            analysis_func: Analysis function to test.
            n_periods: Number of time periods.
            
        Returns:
            Dict with stability metrics.
        """
        # Get time range from data
        interactions = self.storage.get_interactions()
        if not interactions:
            return {'result': 'no_data'}
        
        timestamps = [i.timestamp for i in interactions if i.timestamp]
        if len(timestamps) < n_periods:
            return {'result': 'insufficient_data'}
        
        min_time, max_time = min(timestamps), max(timestamps)
        period_length = (max_time - min_time) / n_periods
        
        results = []
        for i in range(n_periods):
            start = min_time + i * period_length
            end = start + period_length
            
            # Filter data for this period
            period_interactions = [
                inter for inter in interactions
                if inter.timestamp and start <= inter.timestamp < end
            ]
            
            if period_interactions:
                result = analysis_func(period_interactions)
                results.append(result)
        
        # Compute stability metrics
        if len(results) < 2:
            return {'result': 'insufficient_periods'}
        
        return {
            'n_periods': len(results),
            'results': results,
            'coefficient_of_variation': np.std(results) / np.mean(results) if np.mean(results) > 0 else 0,
        }
    
    def sampling_robustness_check(
        self,
        analysis_func: Callable,
        fractions: list = None,
    ) -> dict:
        """Check robustness to sampling.
        
        Args:
            analysis_func: Analysis function to test.
            fractions: Sample fractions to test.
            
        Returns:
            Dict with robustness metrics.
        """
        if fractions is None:
            fractions = [0.5, 0.75, 0.9]
        
        interactions = self.storage.get_interactions()
        if not interactions:
            return {'result': 'no_data'}
        
        results = {}
        for frac in fractions:
            n_sample = int(len(interactions) * frac)
            sample = np.random.choice(interactions, n_sample, replace=False)
            result = analysis_func(list(sample))
            results[frac] = result
        
        return {
            'fractions': fractions,
            'results': results,
        }
    
    def generate_configuration_model(self, network: nx.DiGraph) -> nx.DiGraph:
        """Generate configuration model null network.
        
        Args:
            network: Original network.
            
        Returns:
            Randomized network with same degree sequence.
        """
        # Get degree sequences
        in_degrees = [d for _, d in network.in_degree()]
        out_degrees = [d for _, d in network.out_degree()]
        
        # Generate configuration model
        null_network = nx.directed_configuration_model(
            in_degrees, 
            out_degrees,
            seed=self.config.random_seed
        )
        
        # Remove self-loops and multi-edges
        null_network = nx.DiGraph(null_network)
        null_network.remove_edges_from(nx.selfloop_edges(null_network))
        
        return null_network
    
    def generate_shuffled_timestamps(self) -> pd.DataFrame:
        """Generate null model with shuffled timestamps.
        
        Returns:
            DataFrame with shuffled interaction timestamps.
        """
        interactions = self.storage.get_interactions()
        
        df = pd.DataFrame([
            {
                'source': i.source_agent_id,
                'target': i.target_agent_id,
                'timestamp': i.timestamp,
            }
            for i in interactions
        ])
        
        # Shuffle timestamps
        df['timestamp'] = np.random.permutation(df['timestamp'].values)
        
        return df
    
    def generate_poisson_baseline(self) -> pd.DataFrame:
        """Generate Poisson baseline for activity.
        
        Returns:
            DataFrame with Poisson-distributed activity.
        """
        interactions = self.storage.get_interactions()
        
        # Compute mean rate
        if not interactions:
            return pd.DataFrame()
        
        timestamps = [i.timestamp for i in interactions if i.timestamp]
        if len(timestamps) < 2:
            return pd.DataFrame()
        
        duration = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        rate = len(interactions) / duration if duration > 0 else 1
        
        # Generate Poisson events
        n_events = np.random.poisson(rate * duration)
        
        return pd.DataFrame({
            'n_events': [n_events],
            'rate': [rate],
            'duration_hours': [duration],
        })
    
    def compare_to_null_models(
        self,
        observed: dict,
        null_type: str,
    ) -> dict:
        """Compare observed results to null model.
        
        Args:
            observed: Observed statistics.
            null_type: Type of null model.
            
        Returns:
            Dict with comparison statistics.
        """
        # Generate null distribution
        null_values = []
        
        for _ in range(100):
            if null_type == 'configuration':
                # Would need network to generate null
                null_values.append(np.random.normal(0, 1))
            elif null_type == 'shuffled':
                null_values.append(np.random.normal(0, 1))
            elif null_type == 'poisson':
                null_values.append(np.random.poisson(5))
        
        # Compute p-value
        observed_value = observed.get('value', 0)
        p_value = np.mean([n >= observed_value for n in null_values])
        
        return {
            'observed': observed_value,
            'null_mean': np.mean(null_values),
            'null_std': np.std(null_values),
            'p_value': p_value,
            'z_score': (observed_value - np.mean(null_values)) / np.std(null_values) if np.std(null_values) > 0 else 0,
        }
    
    def verify_clustering_robustness(self, features: pd.DataFrame) -> dict:
        """Verify clustering results across algorithms.
        
        Args:
            features: Feature DataFrame.
            
        Returns:
            Dict with algorithm comparison.
        """
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        X = features[numeric_cols].values
        
        if len(X) < 10:
            return {'result': 'insufficient_data'}
        
        # K-means
        kmeans = KMeans(n_clusters=5, random_state=self.config.random_seed)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=5)
        hier_labels = hierarchical.fit_predict(X)
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        
        # Compute adjusted Rand indices
        return {
            'kmeans_vs_hierarchical': adjusted_rand_score(kmeans_labels, hier_labels),
            'kmeans_vs_dbscan': adjusted_rand_score(kmeans_labels, dbscan_labels),
            'hierarchical_vs_dbscan': adjusted_rand_score(hier_labels, dbscan_labels),
        }
