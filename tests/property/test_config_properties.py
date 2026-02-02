"""Property-based tests for configuration module.

Feature: molt-dynamics-analysis
Property 23: Configuration Override Precedence
Validates: Requirements 10.4
"""

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from molt_dynamics.config import Config


class TestConfigOverridePrecedence:
    """Property 23: Configuration Override Precedence
    
    For any parameter specified in both configuration file and command-line
    arguments, the command-line value SHALL take precedence.
    """
    
    @given(
        yaml_seed=st.integers(min_value=0, max_value=10000),
        cli_seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_cli_overrides_yaml_random_seed(self, yaml_seed: int, cli_seed: int):
        """CLI random_seed should override YAML value."""
        # Create temporary YAML config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(f"random_seed: {yaml_seed}\n")
            yaml_path = f.name
        
        try:
            # Load from YAML
            config = Config.from_yaml(yaml_path)
            
            # Override with CLI argument
            final_config = config.override_from_args(random_seed=cli_seed)
            
            # CLI value should take precedence
            assert final_config.random_seed == cli_seed
        finally:
            os.unlink(yaml_path)
    
    @given(
        yaml_dpi=st.integers(min_value=72, max_value=600),
        cli_dpi=st.integers(min_value=72, max_value=600),
    )
    @settings(max_examples=100)
    def test_cli_overrides_yaml_figure_dpi(self, yaml_dpi: int, cli_dpi: int):
        """CLI figure_dpi should override YAML value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(f"figure_dpi: {yaml_dpi}\n")
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            final_config = config.override_from_args(figure_dpi=cli_dpi)
            assert final_config.figure_dpi == cli_dpi
        finally:
            os.unlink(yaml_path)
    
    @given(
        yaml_delay=st.floats(min_value=0.1, max_value=10.0),
        cli_delay=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_cli_overrides_yaml_rate_limit(self, yaml_delay: float, cli_delay: float):
        """CLI rate_limit_delay should override YAML value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(f"rate_limit_delay: {yaml_delay}\n")
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            final_config = config.override_from_args(rate_limit_delay=cli_delay)
            assert final_config.rate_limit_delay == cli_delay
        finally:
            os.unlink(yaml_path)
    
    @given(
        yaml_topics=st.integers(min_value=5, max_value=50),
        cli_topics=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=100)
    def test_cli_overrides_yaml_lda_topics(self, yaml_topics: int, cli_topics: int):
        """CLI lda_topics should override YAML value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(f"lda_topics: {yaml_topics}\n")
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            final_config = config.override_from_args(lda_topics=cli_topics)
            assert final_config.lda_topics == cli_topics
        finally:
            os.unlink(yaml_path)
    
    @given(
        yaml_iterations=st.integers(min_value=100, max_value=5000),
        cli_iterations=st.integers(min_value=100, max_value=5000),
    )
    @settings(max_examples=100)
    def test_cli_overrides_yaml_bootstrap_iterations(
        self, yaml_iterations: int, cli_iterations: int
    ):
        """CLI bootstrap_iterations should override YAML value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(f"bootstrap_iterations: {yaml_iterations}\n")
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            final_config = config.override_from_args(bootstrap_iterations=cli_iterations)
            assert final_config.bootstrap_iterations == cli_iterations
        finally:
            os.unlink(yaml_path)
    
    def test_none_cli_args_preserve_yaml_values(self):
        """None CLI arguments should not override YAML values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("random_seed: 123\nfigure_dpi: 400\n")
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            # Pass None explicitly - should not override
            final_config = config.override_from_args(random_seed=None, figure_dpi=None)
            assert final_config.random_seed == 123
            assert final_config.figure_dpi == 400
        finally:
            os.unlink(yaml_path)
    
    def test_partial_cli_override(self):
        """Only specified CLI args should override, others preserved."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("random_seed: 100\nfigure_dpi: 200\nlda_topics: 15\n")
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            # Only override random_seed
            final_config = config.override_from_args(random_seed=999)
            
            assert final_config.random_seed == 999  # Overridden
            assert final_config.figure_dpi == 200   # Preserved from YAML
            assert final_config.lda_topics == 15    # Preserved from YAML
        finally:
            os.unlink(yaml_path)


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_valid_config_passes_validation(self):
        """Default config should pass validation."""
        config = Config()
        errors = config.validate()
        assert len(errors) == 0
    
    @given(delay=st.floats(max_value=-0.01))
    @settings(max_examples=50)
    def test_negative_rate_limit_fails_validation(self, delay: float):
        """Negative rate_limit_delay should fail validation."""
        config = Config(rate_limit_delay=delay)
        errors = config.validate()
        assert any("rate_limit_delay" in e for e in errors)
    
    @given(damping=st.floats().filter(lambda x: x <= 0 or x >= 1))
    @settings(max_examples=50)
    def test_invalid_pagerank_damping_fails_validation(self, damping: float):
        """PageRank damping outside (0,1) should fail validation."""
        config = Config(pagerank_damping=damping)
        errors = config.validate()
        assert any("pagerank_damping" in e for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
