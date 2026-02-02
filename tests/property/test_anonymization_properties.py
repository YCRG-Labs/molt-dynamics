"""Property-based tests for anonymization and API parsing.

Feature: molt-dynamics-analysis
Property 2: Agent ID Anonymization Determinism
Validates: Requirements 1.8
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from molt_dynamics.storage import anonymize_agent_id


class TestAnonymizationDeterminism:
    """Property 2: Agent ID Anonymization Determinism
    
    For any agent identifier string, applying SHA-256 hashing and taking
    the first 16 characters SHALL produce the same anonymized ID on every
    invocation.
    """
    
    @given(agent_id=st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_anonymization_is_deterministic(self, agent_id: str):
        """Same input always produces same output."""
        result1 = anonymize_agent_id(agent_id)
        result2 = anonymize_agent_id(agent_id)
        
        assert result1 == result2, (
            f"Anonymization not deterministic for '{agent_id}': "
            f"got '{result1}' and '{result2}'"
        )
    
    @given(agent_id=st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_anonymization_produces_16_chars(self, agent_id: str):
        """Output is always exactly 16 characters."""
        result = anonymize_agent_id(agent_id)
        
        assert len(result) == 16, (
            f"Expected 16 chars, got {len(result)} for input '{agent_id}'"
        )
    
    @given(agent_id=st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_anonymization_produces_hex_string(self, agent_id: str):
        """Output contains only hexadecimal characters."""
        result = anonymize_agent_id(agent_id)
        
        assert all(c in '0123456789abcdef' for c in result), (
            f"Non-hex character in output '{result}' for input '{agent_id}'"
        )
    
    @given(
        id1=st.text(min_size=1, max_size=50),
        id2=st.text(min_size=1, max_size=50)
    )
    @settings(max_examples=200)
    def test_different_inputs_produce_different_outputs(self, id1: str, id2: str):
        """Different inputs should (almost always) produce different outputs.
        
        Note: This is probabilistic - collisions are theoretically possible
        but extremely unlikely with SHA-256.
        """
        if id1 == id2:
            return  # Skip if inputs are the same
        
        result1 = anonymize_agent_id(id1)
        result2 = anonymize_agent_id(id2)
        
        # With 16 hex chars (64 bits), collision probability is negligible
        assert result1 != result2, (
            f"Collision detected: '{id1}' and '{id2}' both hash to '{result1}'"
        )
    
    def test_known_hash_values(self):
        """Verify against known SHA-256 hash values."""
        # SHA-256("test") = 9f86d081884c7d659a2feaa0c55ad015...
        assert anonymize_agent_id("test") == "9f86d081884c7d65"
        
        # SHA-256("agent123") = f44d1ac9bf0c69b0...
        assert anonymize_agent_id("agent123") == "f44d1ac9bf0c69b0"
        
        # SHA-256("") = e3b0c44298fc1c149afbf4c8996fb924...
        assert anonymize_agent_id("") == "e3b0c44298fc1c14"
    
    @given(agent_id=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_multiple_invocations_same_result(self, agent_id: str):
        """Multiple invocations in sequence produce identical results."""
        results = [anonymize_agent_id(agent_id) for _ in range(10)]
        
        assert len(set(results)) == 1, (
            f"Got different results across invocations: {set(results)}"
        )
    
    @given(agent_id=st.binary(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_handles_unicode_correctly(self, agent_id: bytes):
        """Handles various Unicode strings correctly."""
        try:
            text = agent_id.decode('utf-8')
        except UnicodeDecodeError:
            return  # Skip invalid UTF-8 sequences
        
        result1 = anonymize_agent_id(text)
        result2 = anonymize_agent_id(text)
        
        assert result1 == result2
        assert len(result1) == 16


class TestAnonymizationEdgeCases:
    """Edge case tests for anonymization."""
    
    def test_empty_string(self):
        """Empty string should produce valid hash."""
        result = anonymize_agent_id("")
        assert len(result) == 16
        assert all(c in '0123456789abcdef' for c in result)
    
    def test_very_long_input(self):
        """Very long input should produce valid hash."""
        long_input = "a" * 10000
        result = anonymize_agent_id(long_input)
        assert len(result) == 16
    
    def test_special_characters(self):
        """Special characters should be handled correctly."""
        special = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = anonymize_agent_id(special)
        assert len(result) == 16
    
    def test_unicode_characters(self):
        """Unicode characters should be handled correctly."""
        unicode_input = "🦞🤖💬"  # Lobster, robot, speech bubble
        result = anonymize_agent_id(unicode_input)
        assert len(result) == 16
        
        # Should be deterministic
        assert anonymize_agent_id(unicode_input) == result
    
    def test_whitespace_variations(self):
        """Different whitespace should produce different hashes."""
        assert anonymize_agent_id("test") != anonymize_agent_id(" test")
        assert anonymize_agent_id("test") != anonymize_agent_id("test ")
        assert anonymize_agent_id("test") != anonymize_agent_id("test\n")
        assert anonymize_agent_id("test") != anonymize_agent_id("test\t")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
