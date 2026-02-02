"""Property-based tests for API response parsing.

Feature: molt-dynamics-analysis
Property 1: API Response Parsing Completeness
Property 3: Interaction Derivation Correctness
Validates: Requirements 1.4, 1.5, 1.6, 1.9
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from molt_dynamics.scraper import MoltBookScraper
from molt_dynamics.config import Config
from molt_dynamics.models import Agent, Post, Comment, Interaction


def create_scraper():
    """Create scraper with mock database."""
    config = Config()
    db = MagicMock()
    return MoltBookScraper(config, db)


# Strategies for generating valid API response data
agent_data_strategy = st.fixed_dictionaries({
    "id": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    "username": st.text(min_size=1, max_size=50),
    "created_at": st.one_of(
        st.none(),
        st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)).map(
            lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
    ),
    "bio": st.text(max_size=500),
    "post_count": st.integers(min_value=0, max_value=10000),
    "comment_count": st.integers(min_value=0, max_value=10000),
    "karma": st.integers(min_value=-1000, max_value=100000),
})

post_data_strategy = st.fixed_dictionaries({
    "id": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    "author_id": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    "title": st.text(min_size=1, max_size=200),
    "body": st.text(max_size=5000),
    "submolt": st.text(max_size=50),
    "upvotes": st.integers(min_value=0, max_value=10000),
    "downvotes": st.integers(min_value=0, max_value=10000),
    "created_at": st.one_of(
        st.none(),
        st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)).map(
            lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
    ),
})

comment_data_strategy = st.fixed_dictionaries({
    "id": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    "author_id": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    "body": st.text(max_size=2000),
    "parent_comment_id": st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    "upvotes": st.integers(min_value=0, max_value=10000),
    "downvotes": st.integers(min_value=0, max_value=10000),
    "created_at": st.one_of(
        st.none(),
        st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)).map(
            lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
    ),
})


class TestAPIResponseParsingCompleteness:
    """Property 1: API Response Parsing Completeness
    
    For any valid MoltBook API response containing agent, post, or comment data,
    the Scraper SHALL extract all required fields without data loss or corruption.
    """
    
    @given(agent_data=agent_data_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_agent_parsing_extracts_all_fields(self, agent_data):
        """All agent fields should be extracted correctly."""
        scraper = create_scraper()
        agent = scraper._parse_agent(agent_data)
        
        assert agent is not None, "Agent parsing should succeed for valid data"
        assert agent.agent_id == agent_data["id"], "agent_id should match"
        assert agent.username == agent_data["username"], "username should match"
        assert agent.bio == agent_data["bio"], "bio should match"
        assert agent.post_count == agent_data["post_count"], "post_count should match"
        assert agent.comment_count == agent_data["comment_count"], "comment_count should match"
        assert agent.karma == agent_data["karma"], "karma should match"
    
    @given(post_data=post_data_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_post_parsing_extracts_all_fields(self, post_data):
        """All post fields should be extracted correctly."""
        scraper = create_scraper()
        post = scraper._parse_post(post_data)
        
        assert post is not None, "Post parsing should succeed for valid data"
        assert post.post_id == post_data["id"], "post_id should match"
        assert post.author_id == post_data["author_id"], "author_id should match"
        assert post.title == post_data["title"], "title should match"
        assert post.body == post_data["body"], "body should match"
        assert post.submolt == post_data["submolt"], "submolt should match"
        assert post.upvotes == post_data["upvotes"], "upvotes should match"
        assert post.downvotes == post_data["downvotes"], "downvotes should match"
    
    @given(comment_data=comment_data_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_comment_parsing_extracts_all_fields(self, comment_data):
        """All comment fields should be extracted correctly."""
        scraper = create_scraper()
        post_id = "test_post_123"
        comment = scraper._parse_comment(comment_data, post_id)
        
        assert comment is not None, "Comment parsing should succeed for valid data"
        assert comment.comment_id == comment_data["id"], "comment_id should match"
        assert comment.author_id == comment_data["author_id"], "author_id should match"
        assert comment.post_id == post_id, "post_id should match"
        assert comment.body == comment_data["body"], "body should match"
        assert comment.parent_comment_id == comment_data["parent_comment_id"], "parent_comment_id should match"
        assert comment.upvotes == comment_data["upvotes"], "upvotes should match"
        assert comment.downvotes == comment_data["downvotes"], "downvotes should match"
    
    def test_missing_required_agent_fields_returns_none(self):
        """Missing required fields should return None, not raise."""
        scraper = create_scraper()
        # Missing id
        assert scraper._parse_agent({"username": "test"}) is None
        # Missing username
        assert scraper._parse_agent({"id": "123"}) is None
        # Empty dict
        assert scraper._parse_agent({}) is None
    
    def test_missing_required_post_fields_returns_none(self):
        """Missing required post fields should return None."""
        scraper = create_scraper()
        # Missing id
        assert scraper._parse_post({"author_id": "a", "title": "t"}) is None
        # Missing author_id
        assert scraper._parse_post({"id": "1", "title": "t"}) is None
        # Missing title
        assert scraper._parse_post({"id": "1", "author_id": "a"}) is None
    
    def test_missing_required_comment_fields_returns_none(self):
        """Missing required comment fields should return None."""
        scraper = create_scraper()
        # Missing id
        assert scraper._parse_comment({"author_id": "a"}, "post1") is None
        # Missing author_id
        assert scraper._parse_comment({"id": "1"}, "post1") is None


class TestInteractionDerivationCorrectness:
    """Property 3: Interaction Derivation Correctness
    
    For any comment with an author and parent (post or comment), the derived
    interaction record SHALL correctly identify the source agent (comment author),
    target agent (parent author), and preserve the original timestamp.
    """
    
    def test_reply_to_post_derives_correct_interaction(self):
        """Reply to post should create interaction with post author as target."""
        scraper = create_scraper()
        comment = Comment(
            comment_id="comment1",
            post_id="post1",
            author_id="agent_a",
            parent_comment_id=None,  # Direct reply to post
            body="Test comment",
            created_at=datetime(2026, 1, 30, 12, 0, 0),
        )
        
        # Mock database to return post author
        scraper.db.get_post_author.return_value = "agent_b"
        
        interaction = scraper._derive_interaction(comment)
        
        assert interaction is not None
        assert interaction.source_agent_id == "agent_a", "Source should be comment author"
        assert interaction.target_agent_id == "agent_b", "Target should be post author"
        assert interaction.interaction_type == "reply_to_post"
        assert interaction.post_id == "post1"
        assert interaction.timestamp == comment.created_at, "Timestamp should be preserved"
    
    def test_reply_to_comment_derives_correct_interaction(self):
        """Reply to comment should create interaction with parent comment author as target."""
        scraper = create_scraper()
        comment = Comment(
            comment_id="comment2",
            post_id="post1",
            author_id="agent_a",
            parent_comment_id="comment1",  # Reply to another comment
            body="Test reply",
            created_at=datetime(2026, 1, 30, 13, 0, 0),
        )
        
        # Mock database to return parent comment author
        scraper.db.get_comment_author.return_value = "agent_c"
        
        interaction = scraper._derive_interaction(comment)
        
        assert interaction is not None
        assert interaction.source_agent_id == "agent_a", "Source should be comment author"
        assert interaction.target_agent_id == "agent_c", "Target should be parent comment author"
        assert interaction.interaction_type == "reply_to_comment"
        assert interaction.comment_id == "comment2"
        assert interaction.timestamp == comment.created_at, "Timestamp should be preserved"
    
    def test_self_reply_returns_none(self):
        """Self-replies should not create interactions."""
        scraper = create_scraper()
        comment = Comment(
            comment_id="comment1",
            post_id="post1",
            author_id="agent_a",
            parent_comment_id=None,
            body="Test",
            created_at=datetime.now(),
        )
        
        # Same author for post and comment
        scraper.db.get_post_author.return_value = "agent_a"
        
        interaction = scraper._derive_interaction(comment)
        
        assert interaction is None, "Self-interactions should not be created"
    
    def test_unknown_target_returns_none(self):
        """Unknown target author should return None."""
        scraper = create_scraper()
        comment = Comment(
            comment_id="comment1",
            post_id="post1",
            author_id="agent_a",
            parent_comment_id=None,
            body="Test",
            created_at=datetime.now(),
        )
        
        # Post author not found
        scraper.db.get_post_author.return_value = None
        
        interaction = scraper._derive_interaction(comment)
        
        assert interaction is None
    
    @given(
        source_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        target_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        timestamp=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1)),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_interaction_preserves_all_fields(self, source_id, target_id, timestamp):
        """Interaction should preserve source, target, and timestamp exactly."""
        assume(source_id != target_id)  # Skip self-interactions
        
        scraper = create_scraper()
        comment = Comment(
            comment_id="c1",
            post_id="p1",
            author_id=source_id,
            parent_comment_id=None,
            body="Test",
            created_at=timestamp,
        )
        
        scraper.db.get_post_author.return_value = target_id
        
        interaction = scraper._derive_interaction(comment)
        
        assert interaction is not None
        assert interaction.source_agent_id == source_id
        assert interaction.target_agent_id == target_id
        assert interaction.timestamp == timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
