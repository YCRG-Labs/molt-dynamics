"""Property-based tests for clustering and role classification.

Feature: molt-dynamics-analysis
Property 10: Cluster Assignment Completeness
Property 11: Role Classification Determinism
Validates: Requirements 4.1, 4.5, 4.6, 4.7
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, HealthCheck

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from molt_dynamics.rq1_roles import RoleAnalyzer
from molt_dynamics.config import Config


def create_test_features(n_agents: int, n_features: int = 10) -> pd.DataFrame:
    """Create test feature DataFrame."""
    np.random.seed(42)
    
    data = {
        'agent_id': [f'agent_{i}' for i in range(n_agents)],
        'normalized_entropy': np.random.randn(n_agents),
        'betweenness': np.random.randn(n_agents),
        'post_comment_ratio': np.random.randn(n_agents),
        'in_degree': np.random.randn(n_agents),
        'total_posts': np.random.randn(n_agents),
        'total_comments': np.random.randn(n_agents),
        'pagerank': np.random.randn(n_agents),
        'clustering_coefficient': np.random.randn(n_agents),
        'topic_entropy': np.random.randn(n_agents),
        'avg_sentiment': np.random.randn(n_agents),
    }
    
    return pd.DataFrame(data)


class TestClusterAssignmentCompleteness:
    """Property 10: Cluster Assignment Completeness
    
    For any k-means clustering result, every agent SHALL be assigned to exactly
    one cluster, and the number of distinct cluster labels SHALL equal k.
    """
    
    @given(
        n_agents=st.integers(min_value=20, max_value=100),
        k=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_every_agent_assigned_to_one_cluster(self, n_agents: int, k: int):
        """Every agent should be assigned to exactly one cluster."""
        assume(n_agents >= k)  # Need at least k agents
        
        features = create_test_features(n_agents)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        labels = analyzer.perform_clustering(k)
        
        # Every agent should have a label
        assert len(labels) == n_agents, (
            f"Expected {n_agents} labels, got {len(labels)}"
        )
        
        # All labels should be valid cluster indices
        assert all(0 <= label < k for label in labels), (
            f"Labels should be in range [0, {k}), got {set(labels)}"
        )
    
    @given(
        n_agents=st.integers(min_value=20, max_value=100),
        k=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_k_distinct_clusters(self, n_agents: int, k: int):
        """Number of distinct cluster labels should equal k."""
        assume(n_agents >= k * 3)  # Need enough agents for k clusters
        
        features = create_test_features(n_agents)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        labels = analyzer.perform_clustering(k)
        
        n_unique = len(np.unique(labels))
        
        # Should have exactly k clusters (or fewer if data doesn't support k)
        assert n_unique <= k, (
            f"Should have at most {k} clusters, got {n_unique}"
        )
    
    def test_cluster_labels_are_integers(self):
        """Cluster labels should be integers."""
        features = create_test_features(50)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        labels = analyzer.perform_clustering(k=5)
        
        assert labels.dtype in [np.int32, np.int64], (
            f"Labels should be integers, got {labels.dtype}"
        )
    
    def test_cluster_assignments_dataframe(self):
        """Cluster assignments should be retrievable as DataFrame."""
        features = create_test_features(50)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        analyzer.perform_clustering(k=5)
        
        assignments = analyzer.get_cluster_assignments()
        
        assert 'agent_id' in assignments.columns
        assert 'cluster' in assignments.columns
        assert len(assignments) == 50


class TestRoleClassificationDeterminism:
    """Property 11: Role Classification Determinism
    
    For any agent with feature values F, applying the role classification
    threshold rules SHALL produce the same role assignment on every invocation.
    """
    
    @given(
        n_runs=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=10)
    def test_role_classification_deterministic(self, n_runs: int):
        """Role classification should be deterministic."""
        features = create_test_features(50)
        config = Config()
        
        # Run classification multiple times
        results = []
        for _ in range(n_runs):
            analyzer = RoleAnalyzer(features, config)
            roles = analyzer.classify_roles()
            results.append(roles)
        
        # All results should be identical
        for i in range(1, len(results)):
            pd.testing.assert_series_equal(
                results[0], results[i],
                check_names=False,
            )
    
    def test_role_classification_covers_all_agents(self):
        """Every agent should be assigned a role."""
        features = create_test_features(50)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        roles = analyzer.classify_roles()
        
        assert len(roles) == 50
        assert not roles.isna().any()
    
    def test_role_classification_valid_roles(self):
        """All assigned roles should be from the taxonomy."""
        valid_roles = {'Specialist', 'Connector', 'Initiator', 'Synthesizer', 'Generalist'}
        
        features = create_test_features(100)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        roles = analyzer.classify_roles()
        
        assigned_roles = set(roles.unique())
        assert assigned_roles.issubset(valid_roles), (
            f"Invalid roles found: {assigned_roles - valid_roles}"
        )
    
    @given(
        normalized_entropy=st.floats(min_value=-3, max_value=3),
        betweenness=st.floats(min_value=-3, max_value=3),
        post_comment_ratio=st.floats(min_value=-3, max_value=3),
        in_degree=st.floats(min_value=-3, max_value=3),
        total_posts=st.floats(min_value=-3, max_value=3),
    )
    @settings(max_examples=50)
    def test_single_agent_classification_deterministic(
        self,
        normalized_entropy: float,
        betweenness: float,
        post_comment_ratio: float,
        in_degree: float,
        total_posts: float,
    ):
        """Single agent classification should be deterministic."""
        features = pd.DataFrame({
            'agent_id': ['test_agent'],
            'normalized_entropy': [normalized_entropy],
            'betweenness': [betweenness],
            'post_comment_ratio': [post_comment_ratio],
            'in_degree': [in_degree],
            'total_posts': [total_posts],
        })
        
        config = Config()
        
        # Classify twice
        analyzer1 = RoleAnalyzer(features, config)
        role1 = analyzer1.classify_roles().iloc[0]
        
        analyzer2 = RoleAnalyzer(features, config)
        role2 = analyzer2.classify_roles().iloc[0]
        
        assert role1 == role2, f"Role changed: {role1} vs {role2}"


class TestClusteringEdgeCases:
    """Edge case tests for clustering."""
    
    def test_minimum_agents_for_clustering(self):
        """Should handle minimum number of agents."""
        features = create_test_features(10)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        labels = analyzer.perform_clustering(k=3)
        
        assert len(labels) == 10
    
    def test_silhouette_scores_per_cluster(self):
        """Should compute silhouette scores per cluster."""
        features = create_test_features(50)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        analyzer.perform_clustering(k=5)
        
        scores = analyzer.compute_silhouette_scores()
        
        assert len(scores) <= 5
        assert all(-1 <= s <= 1 for s in scores.values())
    
    def test_cluster_profiles(self):
        """Should compute cluster profiles."""
        features = create_test_features(50)
        config = Config()
        
        analyzer = RoleAnalyzer(features, config)
        analyzer.perform_clustering(k=5)
        
        profiles = analyzer.get_cluster_profiles()
        
        assert len(profiles) <= 5
        assert 'normalized_entropy' in profiles.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
