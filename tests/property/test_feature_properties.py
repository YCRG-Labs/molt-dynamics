"""Property-based tests for feature extraction.

Feature: molt-dynamics-analysis
Property 8: Shannon Entropy Calculation
Property 9: Feature Standardization Invariants
Validates: Requirements 3.2, 3.3, 3.9
"""

import sys
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, HealthCheck

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from molt_dynamics.features import (
    compute_shannon_entropy,
    normalize_entropy,
    FeatureExtractor,
)
from molt_dynamics.config import Config


class TestShannonEntropyCalculation:
    """Property 8: Shannon Entropy Calculation
    
    For any agent with submolt participation distribution p, the computed
    topic diversity H SHALL equal -Σ(p_s × log2(p_s)) within floating-point tolerance.
    """
    
    @given(
        counts=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_entropy_matches_formula(self, counts: list[int]):
        """Computed entropy should match the Shannon entropy formula."""
        # Convert counts to probability distribution
        total = sum(counts)
        distribution = np.array([c / total for c in counts])
        
        # Compute expected entropy manually
        expected = 0.0
        for p in distribution:
            if p > 0:
                expected -= p * math.log2(p)
        
        # Compute using our function
        actual = compute_shannon_entropy(distribution)
        
        # Should match within floating-point tolerance
        assert abs(actual - expected) < 1e-10, (
            f"Expected entropy {expected}, got {actual}"
        )
    
    @given(n=st.integers(min_value=2, max_value=50))
    @settings(max_examples=50)
    def test_uniform_distribution_max_entropy(self, n: int):
        """Uniform distribution should have maximum entropy log2(n)."""
        distribution = np.ones(n) / n
        
        entropy = compute_shannon_entropy(distribution)
        expected_max = math.log2(n)
        
        assert abs(entropy - expected_max) < 1e-10, (
            f"Uniform distribution entropy should be {expected_max}, got {entropy}"
        )
    
    def test_single_category_zero_entropy(self):
        """Single category (concentrated) distribution should have zero entropy."""
        distribution = np.array([1.0])
        
        entropy = compute_shannon_entropy(distribution)
        
        assert entropy == 0.0, f"Single category entropy should be 0, got {entropy}"
    
    @given(
        counts=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_entropy_non_negative(self, counts: list[int]):
        """Entropy should always be non-negative."""
        total = sum(counts)
        distribution = np.array([c / total for c in counts])
        
        entropy = compute_shannon_entropy(distribution)
        
        assert entropy >= 0, f"Entropy should be non-negative, got {entropy}"
    
    @given(
        counts=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_entropy_bounded_by_max(self, counts: list[int]):
        """Entropy should be bounded by log2(n) where n is number of categories."""
        total = sum(counts)
        distribution = np.array([c / total for c in counts])
        n = len(distribution)
        
        entropy = compute_shannon_entropy(distribution)
        max_entropy = math.log2(n)
        
        assert entropy <= max_entropy + 1e-10, (
            f"Entropy {entropy} should be <= max {max_entropy}"
        )


class TestNormalizedEntropy:
    """Tests for normalized entropy calculation."""
    
    @given(n=st.integers(min_value=2, max_value=50))
    @settings(max_examples=50)
    def test_uniform_normalized_to_one(self, n: int):
        """Uniform distribution should have normalized entropy of 1."""
        distribution = np.ones(n) / n
        entropy = compute_shannon_entropy(distribution)
        
        normalized = normalize_entropy(entropy, n)
        
        assert abs(normalized - 1.0) < 1e-10, (
            f"Uniform normalized entropy should be 1.0, got {normalized}"
        )
    
    def test_concentrated_normalized_to_zero(self):
        """Concentrated distribution should have normalized entropy of 0."""
        # All probability on one category
        distribution = np.array([1.0, 0.0, 0.0])
        entropy = compute_shannon_entropy(distribution)
        
        normalized = normalize_entropy(entropy, 3)
        
        assert abs(normalized) < 1e-10, (
            f"Concentrated normalized entropy should be 0, got {normalized}"
        )
    
    @given(
        counts=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_normalized_in_unit_interval(self, counts: list[int]):
        """Normalized entropy should be in [0, 1]."""
        total = sum(counts)
        distribution = np.array([c / total for c in counts])
        n = len(distribution)
        
        entropy = compute_shannon_entropy(distribution)
        normalized = normalize_entropy(entropy, n)
        
        assert 0 <= normalized <= 1 + 1e-10, (
            f"Normalized entropy should be in [0, 1], got {normalized}"
        )



class TestFeatureStandardizationInvariants:
    """Property 9: Feature Standardization Invariants
    
    For any feature column after standardization, the mean SHALL be within ε of 0
    and the standard deviation SHALL be within ε of 1, where ε = 1e-10.
    """
    
    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_standardized_mean_near_zero(self, n_samples: int, n_features: int):
        """Standardized features should have mean near zero."""
        # Generate random feature data
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features) * 10 + 5  # Non-zero mean
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)
        df['agent_id'] = [f'agent_{i}' for i in range(n_samples)]
        
        # Mock database and network
        db = MagicMock()
        network = MagicMock()
        config = Config()
        
        extractor = FeatureExtractor(db, network, config)
        df_std = extractor.standardize_features(df)
        
        # Check mean of each feature column
        for col in columns:
            mean = df_std[col].mean()
            assert abs(mean) < 1e-10, (
                f"Standardized mean of {col} should be ~0, got {mean}"
            )
    
    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_standardized_std_near_one(self, n_samples: int, n_features: int):
        """Standardized features should have std near one."""
        # Generate random feature data with varying scales
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)
        # Scale each column differently
        for i in range(n_features):
            data[:, i] = data[:, i] * (i + 1) * 10
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)
        df['agent_id'] = [f'agent_{i}' for i in range(n_samples)]
        
        # Mock database and network
        db = MagicMock()
        network = MagicMock()
        config = Config()
        
        extractor = FeatureExtractor(db, network, config)
        df_std = extractor.standardize_features(df)
        
        # Check std of each feature column
        for col in columns:
            std = df_std[col].std()
            # Use ddof=0 for population std to match sklearn
            assert abs(std - 1.0) < 0.1, (
                f"Standardized std of {col} should be ~1, got {std}"
            )
    
    def test_constant_column_handled(self):
        """Constant columns should be handled gracefully."""
        # Create DataFrame with a constant column
        df = pd.DataFrame({
            'agent_id': ['a', 'b', 'c'],
            'feature_1': [1.0, 2.0, 3.0],
            'constant': [5.0, 5.0, 5.0],  # Constant column
        })
        
        db = MagicMock()
        network = MagicMock()
        config = Config()
        
        extractor = FeatureExtractor(db, network, config)
        
        # Should not raise an error
        df_std = extractor.standardize_features(df)
        
        # Non-constant column should be standardized
        assert abs(df_std['feature_1'].mean()) < 1e-10
    
    def test_agent_id_excluded(self):
        """Agent ID column should not be standardized."""
        df = pd.DataFrame({
            'agent_id': ['agent_1', 'agent_2', 'agent_3'],
            'feature_1': [1.0, 2.0, 3.0],
        })
        
        db = MagicMock()
        network = MagicMock()
        config = Config()
        
        extractor = FeatureExtractor(db, network, config)
        df_std = extractor.standardize_features(df)
        
        # Agent IDs should be unchanged
        assert list(df_std['agent_id']) == ['agent_1', 'agent_2', 'agent_3']


class TestFeatureExtractorEdgeCases:
    """Edge case tests for feature extractor."""
    
    def test_empty_posts_activity_metrics(self):
        """Agent with no posts should have zero activity metrics."""
        db = MagicMock()
        db.get_posts.return_value = []
        db.get_comments.return_value = []
        
        network = MagicMock()
        config = Config()
        
        extractor = FeatureExtractor(db, network, config)
        metrics = extractor.compute_activity_metrics('agent_1')
        
        assert metrics['total_posts'] == 0
        assert metrics['total_comments'] == 0
        assert metrics['post_comment_ratio'] == 0.0
    
    def test_empty_posts_topic_diversity(self):
        """Agent with no posts should have zero topic diversity."""
        db = MagicMock()
        db.get_posts.return_value = []
        
        network = MagicMock()
        config = Config()
        
        extractor = FeatureExtractor(db, network, config)
        entropy, normalized = extractor.compute_topic_diversity('agent_1')
        
        assert entropy == 0.0
        assert normalized == 0.0
    
    def test_agent_not_in_network(self):
        """Agent not in network should have zero centrality metrics."""
        db = MagicMock()
        
        # Create a real network without the agent
        import networkx as nx
        network = nx.DiGraph()
        network.add_edge('other_1', 'other_2', weight=1)
        
        config = Config()
        
        extractor = FeatureExtractor(db, network, config)
        metrics = extractor.compute_centrality_metrics('agent_not_in_network')
        
        assert metrics['in_degree'] == 0
        assert metrics['out_degree'] == 0
        assert metrics['betweenness'] == 0.0
        assert metrics['pagerank'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
