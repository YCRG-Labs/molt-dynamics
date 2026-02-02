"""Property-based tests for cascade identification and analysis.

Feature: molt-dynamics-analysis
Property 12: Cascade Minimum Adopter Threshold
Property 13: Cascade Adoption Ordering
Property 14: Contagion Type Classification
Validates: Requirements 5.1, 5.4, 5.6
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from molt_dynamics.rq2_diffusion import (
    CascadeIdentifier,
    CascadeAnalyzer,
    DiffusionModeler,
    verify_cascade_ordering,
)
from molt_dynamics.models import Cascade, Post, Comment
from molt_dynamics.config import Config


class TestCascadeMinimumAdopterThreshold:
    """Property 12: Cascade Minimum Adopter Threshold
    
    For any identified meme cascade, the number of unique adopting agents
    SHALL be at least the configured minimum (default 5).
    """
    
    @given(min_adopters=st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_all_cascades_meet_threshold(self, min_adopters: int):
        """All identified cascades should have >= min_adopters unique adopters."""
        # Create mock database with posts containing repeated phrases
        db = MagicMock()
        config = Config()
        
        # Create posts with a common phrase used by many agents
        posts = []
        for i in range(min_adopters + 5):
            posts.append(Post(
                post_id=f'post_{i}',
                author_id=f'agent_{i}',
                title=f'Title {i}',
                body='This is a common phrase that spreads through the network',
                submolt='test',
                created_at=datetime(2026, 1, 1) + timedelta(hours=i),
            ))
        
        db.get_posts.return_value = posts
        db.get_comments.return_value = []
        
        identifier = CascadeIdentifier(db, config)
        cascades = identifier.identify_meme_cascades(min_adopters=min_adopters)
        
        # All cascades should meet the threshold
        for cascade in cascades:
            unique_adopters = set(a[0] for a in cascade.adoptions)
            assert len(unique_adopters) >= min_adopters, (
                f"Cascade {cascade.cascade_id} has {len(unique_adopters)} adopters, "
                f"expected >= {min_adopters}"
            )
    
    def test_cascades_below_threshold_excluded(self):
        """Cascades with fewer than min_adopters should be excluded."""
        db = MagicMock()
        config = Config()
        
        # Create posts with a phrase used by only 3 agents
        posts = [
            Post(
                post_id=f'post_{i}',
                author_id=f'agent_{i}',
                title=f'Title {i}',
                body='rare phrase only three agents use',
                submolt='test',
                created_at=datetime(2026, 1, 1) + timedelta(hours=i),
            )
            for i in range(3)
        ]
        
        db.get_posts.return_value = posts
        db.get_comments.return_value = []
        
        identifier = CascadeIdentifier(db, config)
        cascades = identifier.identify_meme_cascades(min_adopters=5)
        
        # Should find no cascades with the rare phrase
        rare_cascades = [c for c in cascades if 'rare phrase' in c.content_hash]
        assert len(rare_cascades) == 0


class TestCascadeAdoptionOrdering:
    """Property 13: Cascade Adoption Ordering
    
    For any cascade's adoption sequence, the timestamps SHALL be
    monotonically non-decreasing.
    """
    
    @given(
        n_adoptions=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_adoptions_chronologically_ordered(self, n_adoptions: int):
        """Cascade adoptions should be in chronological order."""
        # Create a cascade with random timestamps, then sort
        base_time = datetime(2026, 1, 1)
        
        adoptions = [
            (f'agent_{i}', base_time + timedelta(hours=i))
            for i in range(n_adoptions)
        ]
        
        cascade = Cascade(
            cascade_id='test_cascade',
            cascade_type='meme',
            seed_agent=adoptions[0][0],
            seed_time=adoptions[0][1],
            adoptions=adoptions,
            content_hash='test',
        )
        
        # Verify ordering
        assert verify_cascade_ordering(cascade), "Adoptions should be ordered"
    
    def test_verify_ordering_detects_violations(self):
        """verify_cascade_ordering should detect out-of-order adoptions."""
        # Create cascade with out-of-order timestamps
        adoptions = [
            ('agent_1', datetime(2026, 1, 1, 12, 0)),
            ('agent_2', datetime(2026, 1, 1, 10, 0)),  # Earlier than previous!
            ('agent_3', datetime(2026, 1, 1, 14, 0)),
        ]
        
        cascade = Cascade(
            cascade_id='test_cascade',
            cascade_type='meme',
            seed_agent='agent_1',
            seed_time=datetime(2026, 1, 1, 12, 0),
            adoptions=adoptions,
            content_hash='test',
        )
        
        assert not verify_cascade_ordering(cascade), "Should detect ordering violation"
    
    def test_single_adoption_always_ordered(self):
        """Single adoption cascade should always be ordered."""
        cascade = Cascade(
            cascade_id='test_cascade',
            cascade_type='meme',
            seed_agent='agent_1',
            seed_time=datetime(2026, 1, 1),
            adoptions=[('agent_1', datetime(2026, 1, 1))],
            content_hash='test',
        )
        
        assert verify_cascade_ordering(cascade)
    
    def test_empty_adoptions_ordered(self):
        """Empty adoption list should be considered ordered."""
        cascade = Cascade(
            cascade_id='test_cascade',
            cascade_type='meme',
            seed_agent='agent_1',
            seed_time=datetime(2026, 1, 1),
            adoptions=[],
            content_hash='test',
        )
        
        assert verify_cascade_ordering(cascade)


class TestContagionTypeClassification:
    """Property 14: Contagion Type Classification
    
    For any diffusion model result where the quadratic exposure coefficient
    β₂ > 0 with p-value < α, the contagion type SHALL be classified as
    "complex"; otherwise "simple".
    """
    
    def test_contagion_type_returns_valid_value(self):
        """Contagion type should be 'simple', 'complex', or 'unknown'."""
        import networkx as nx
        
        # Create simple cascades and network
        cascades = [
            Cascade(
                cascade_id=f'cascade_{i}',
                cascade_type='meme',
                seed_agent='agent_0',
                seed_time=datetime(2026, 1, 1),
                adoptions=[
                    (f'agent_{j}', datetime(2026, 1, 1) + timedelta(hours=j))
                    for j in range(10)
                ],
                content_hash=f'test_{i}',
            )
            for i in range(5)
        ]
        
        network = nx.DiGraph()
        for i in range(10):
            for j in range(i + 1, 10):
                network.add_edge(f'agent_{i}', f'agent_{j}', weight=1)
        
        modeler = DiffusionModeler(cascades, network)
        contagion_type = modeler.test_contagion_type()
        
        assert contagion_type in ['simple', 'complex', 'unknown'], (
            f"Invalid contagion type: {contagion_type}"
        )


class TestCascadeAnalyzerStatistics:
    """Tests for cascade statistics computation."""
    
    def test_cascade_statistics_completeness(self):
        """Cascade statistics should include all required fields."""
        cascades = [
            Cascade(
                cascade_id='cascade_1',
                cascade_type='meme',
                seed_agent='agent_0',
                seed_time=datetime(2026, 1, 1),
                adoptions=[
                    ('agent_0', datetime(2026, 1, 1)),
                    ('agent_1', datetime(2026, 1, 1, 1)),
                    ('agent_2', datetime(2026, 1, 1, 2)),
                ],
                content_hash='test',
            )
        ]
        
        analyzer = CascadeAnalyzer(cascades)
        stats = analyzer.compute_cascade_statistics()
        
        required_columns = [
            'cascade_id', 'cascade_type', 'n_adoptions',
            'n_unique_adopters', 'duration_hours', 'seed_agent'
        ]
        
        for col in required_columns:
            assert col in stats.columns, f"Missing column: {col}"
    
    @given(n_cascades=st.integers(min_value=1, max_value=20))
    @settings(max_examples=10)
    def test_statistics_row_count_matches_cascades(self, n_cascades: int):
        """Statistics DataFrame should have one row per cascade."""
        cascades = [
            Cascade(
                cascade_id=f'cascade_{i}',
                cascade_type='meme',
                seed_agent='agent_0',
                seed_time=datetime(2026, 1, 1),
                adoptions=[('agent_0', datetime(2026, 1, 1))],
                content_hash=f'test_{i}',
            )
            for i in range(n_cascades)
        ]
        
        analyzer = CascadeAnalyzer(cascades)
        stats = analyzer.compute_cascade_statistics()
        
        assert len(stats) == n_cascades


class TestCascadeEdgeCases:
    """Edge case tests for cascade handling."""
    
    def test_empty_cascade_list(self):
        """Should handle empty cascade list gracefully."""
        analyzer = CascadeAnalyzer([])
        stats = analyzer.compute_cascade_statistics()
        
        assert len(stats) == 0
    
    def test_power_law_insufficient_data(self):
        """Power-law test should handle insufficient data."""
        cascades = [
            Cascade(
                cascade_id='cascade_1',
                cascade_type='meme',
                seed_agent='agent_0',
                seed_time=datetime(2026, 1, 1),
                adoptions=[('agent_0', datetime(2026, 1, 1))],
                content_hash='test',
            )
        ]
        
        analyzer = CascadeAnalyzer(cascades)
        result = analyzer.test_power_law()
        
        assert result.get('result') == 'insufficient_data'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
