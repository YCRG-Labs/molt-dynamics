"""Data models for Molt Dynamics analysis.

Defines dataclasses for all core entities matching the database schema.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class Agent:
    """An autonomous AI entity registered on MoltBook."""
    agent_id: str  # SHA-256 hash (first 16 chars)
    username: str  # Original username (internal only)
    join_date: Optional[datetime] = None
    bio: str = ""
    post_count: int = 0
    comment_count: int = 0
    karma: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class Post:
    """A top-level discussion created by an agent."""
    post_id: str
    author_id: str
    title: str
    body: str = ""
    submolt: str = ""
    upvotes: int = 0
    downvotes: int = 0
    created_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None


@dataclass
class Comment:
    """A threaded reply to a post or another comment."""
    comment_id: str
    post_id: str
    author_id: str
    body: str = ""
    parent_comment_id: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0
    created_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None


@dataclass
class Interaction:
    """A derived interaction record for network analysis."""
    source_agent_id: str
    target_agent_id: str
    interaction_type: str  # 'reply_to_post' or 'reply_to_comment'
    post_id: str
    timestamp: datetime
    comment_id: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Submolt:
    """A topic-specific community analogous to a subreddit."""
    name: str
    description: str = ""
    member_count: int = 0
    post_count: int = 0
    created_at: Optional[datetime] = None


@dataclass
class AgentFeatures:
    """Computed behavioral features for an agent."""
    agent_id: str
    
    # Activity metrics
    total_posts: int = 0
    total_comments: int = 0
    post_comment_ratio: float = 0.0
    active_lifespan_days: float = 0.0
    posts_per_day: float = 0.0
    
    # Topic diversity
    topic_entropy: float = 0.0
    normalized_entropy: float = 0.0
    
    # Network centrality
    in_degree: int = 0
    out_degree: int = 0
    betweenness: float = 0.0
    clustering_coefficient: float = 0.0
    pagerank: float = 0.0
    
    # Temporal patterns
    hour_distribution: np.ndarray = field(default_factory=lambda: np.zeros(24))
    autocorrelation: float = 0.0
    burst_coefficient: float = 0.0
    
    # Content features
    avg_post_length: float = 0.0
    vocabulary_diversity: float = 0.0
    avg_sentiment: float = 0.0
    technical_density: float = 0.0
    
    # Topic distribution (LDA)
    topic_vector: np.ndarray = field(default_factory=lambda: np.zeros(20))


@dataclass
class Cascade:
    """An information cascade spreading through the network."""
    cascade_id: str
    cascade_type: str  # 'meme', 'skill', 'behavioral'
    seed_agent: str
    seed_time: datetime
    adoptions: list[tuple[str, datetime]] = field(default_factory=list)
    content_hash: str = ""


@dataclass
class ComplexityFeatures:
    """Complexity proxy features for a thread's seed post."""
    thread_id: str
    seed_word_count: int
    technical_keyword_density: float
    code_block_present: bool


@dataclass
class HubEmergenceEvent:
    """Tracks when an agent first enters a hub cluster."""
    agent_id: str
    emergence_date: Optional[datetime]  # None if right-censored
    emerged: bool
    time_to_emergence_days: float  # observation time if censored
    join_cohort: str  # 'day1-3', 'day4-7', 'later'
    initial_posting_cadence: float
    submolt_diversity_day3: int
    early_reply: bool


@dataclass
class CollaborativeEvent:
    """A collaborative problem-solving event."""
    thread_id: str
    participants: list[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    problem_statement: str = ""
    solution: str = ""
    quality_score: Optional[float] = None


@dataclass
class PropagationNode:
    """A node in a cascade propagation tree."""
    agent_id: str
    adoption_time: datetime
    depth: int
    parent_agent_id: Optional[str]
    role_cluster: Optional[str]


@dataclass
class VocabularyTerm:
    """A novel term that emerged from a specific thread."""
    term: str
    source_thread_id: str
    first_appearance: datetime
    adoption_count: int
    adopting_thread_ids: list[str]
    time_to_first_spread_hours: Optional[float]
