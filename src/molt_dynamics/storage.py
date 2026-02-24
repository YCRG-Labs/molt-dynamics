"""JSON-based storage module for Molt Dynamics analysis.

Provides file-based data persistence using JSON format, replacing PostgreSQL.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import hashlib

import pandas as pd

from .config import Config
from .models import Agent, Post, Comment, Interaction, Submolt

logger = logging.getLogger(__name__)


def anonymize_agent_id(agent_id: str) -> str:
    """Anonymize agent identifier using SHA-256 hashing.
    
    Args:
        agent_id: Original agent identifier.
        
    Returns:
        First 16 characters of SHA-256 hash.
    """
    hash_obj = hashlib.sha256(agent_id.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse datetime from string or return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class JSONStorage:
    """JSON file-based storage for MoltBook data."""
    
    def __init__(self, config: Config) -> None:
        """Initialize JSON storage.
        
        Args:
            config: Configuration object with storage settings.
        """
        self.config = config
        self.data_dir = Path(config.output_dir) / "data"
        
        # In-memory data stores
        self._agents: dict[str, dict] = {}
        self._posts: dict[str, dict] = {}
        self._comments: dict[str, dict] = {}
        self._interactions: list[dict] = []
        self._submolts: dict[str, dict] = {}
        self._memberships: dict[str, dict] = {}
        
        # Indexes for fast lookups
        self._posts_by_author: dict[str, list[str]] = {}
        self._comments_by_author: dict[str, list[str]] = {}
    
    def connect(self) -> None:
        """Initialize storage directory and load existing data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_all()
        logger.info(f"JSON storage initialized at {self.data_dir}")
    
    def close(self) -> None:
        """Save all data to disk."""
        self._save_all()
        logger.info("JSON storage saved and closed")
    
    def _load_all(self) -> None:
        """Load all data from JSON files."""
        self._agents = self._load_json("agents.json", {})
        self._posts = self._load_json("posts.json", {})
        self._comments = self._load_json("comments.json", {})
        self._interactions = self._load_json("interactions.json", [])
        self._submolts = self._load_json("submolts.json", {})
        self._memberships = self._load_json("memberships.json", {})
        
        # Build indexes for fast author lookups
        self._build_author_indexes()
    
    def _save_all(self) -> None:
        """Save all data to JSON files."""
        self._save_json("agents.json", self._agents)
        self._save_json("posts.json", self._posts)
        self._save_json("comments.json", self._comments)
        self._save_json("interactions.json", self._interactions)
        self._save_json("submolts.json", self._submolts)
        self._save_json("memberships.json", self._memberships)
    
    def _build_author_indexes(self) -> None:
        """Build indexes for fast author-based lookups."""
        self._posts_by_author = {}
        self._comments_by_author = {}
        
        for post_id, data in self._posts.items():
            author_id = data.get('author_id')
            if author_id:
                if author_id not in self._posts_by_author:
                    self._posts_by_author[author_id] = []
                self._posts_by_author[author_id].append(post_id)
        
        for comment_id, data in self._comments.items():
            author_id = data.get('author_id')
            if author_id:
                if author_id not in self._comments_by_author:
                    self._comments_by_author[author_id] = []
                self._comments_by_author[author_id].append(comment_id)
        
        logger.info(f"Built author indexes: {len(self._posts_by_author)} post authors, {len(self._comments_by_author)} comment authors")
    
    def _load_json(self, filename: str, default: Any) -> Any:
        """Load data from a JSON file."""
        filepath = self.data_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {filename}: {e}")
        return default
    
    def _save_json(self, filename: str, data: Any) -> None:
        """Save data to a JSON file atomically."""
        filepath = self.data_dir / filename
        temp_filepath = self.data_dir / f".{filename}.tmp"
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=DateTimeEncoder, indent=2)
            # Atomic rename - if this fails, original file is intact
            temp_filepath.replace(filepath)
        except IOError as e:
            logger.error(f"Failed to save {filename}: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
    
    def initialize_schema(self) -> None:
        """Initialize empty data structures (no-op for JSON storage)."""
        logger.info("JSON storage schema initialized (in-memory)")
    
    # ==================== Agent Operations ====================
    
    def insert_agent(self, agent: Agent) -> str:
        """Insert or update an agent record.
        
        Args:
            agent: Agent object to insert.
            
        Returns:
            Anonymized agent ID.
        """
        anon_id = anonymize_agent_id(agent.agent_id)
        
        self._agents[anon_id] = {
            'agent_id': anon_id,
            'username': agent.username,
            'join_date': agent.join_date.isoformat() if agent.join_date else None,
            'bio': agent.bio,
            'post_count': agent.post_count,
            'comment_count': agent.comment_count,
            'karma': agent.karma,
            'first_seen': agent.first_seen.isoformat() if agent.first_seen else None,
            'last_seen': agent.last_seen.isoformat() if agent.last_seen else None,
        }
        
        return anon_id
    
    def get_agents(self, filters: Optional[dict] = None) -> list[Agent]:
        """Query agents with optional filters.
        
        Args:
            filters: Optional dict with filter conditions.
            
        Returns:
            List of Agent objects.
        """
        agents = []
        
        for data in self._agents.values():
            # Apply filters
            if filters:
                if "min_posts" in filters and data.get("post_count", 0) < filters["min_posts"]:
                    continue
                if "min_karma" in filters and data.get("karma", 0) < filters["min_karma"]:
                    continue
                if "since" in filters:
                    first_seen = parse_datetime(data.get("first_seen"))
                    if first_seen and first_seen < filters["since"]:
                        continue
            
            agents.append(Agent(
                agent_id=data['agent_id'],
                username=data.get('username', ''),
                join_date=parse_datetime(data.get('join_date')),
                bio=data.get('bio', ''),
                post_count=data.get('post_count', 0),
                comment_count=data.get('comment_count', 0),
                karma=data.get('karma', 0),
                first_seen=parse_datetime(data.get('first_seen')),
                last_seen=parse_datetime(data.get('last_seen')),
            ))
        
        return agents
    
    def get_agent_count(self) -> int:
        """Get total number of agents."""
        return len(self._agents)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get a specific agent by ID.
        
        Args:
            agent_id: Agent identifier (may be original or anonymized).
            
        Returns:
            Agent object or None if not found.
        """
        # Try direct lookup first (for anonymized IDs)
        data = self._agents.get(agent_id)
        
        # If not found, try anonymizing the ID
        if not data:
            anon_id = anonymize_agent_id(agent_id)
            data = self._agents.get(anon_id)
        
        if not data:
            return None
        
        return Agent(
            agent_id=data['agent_id'],
            username=data.get('username', ''),
            join_date=parse_datetime(data.get('join_date')),
            bio=data.get('bio', ''),
            post_count=data.get('post_count', 0),
            comment_count=data.get('comment_count', 0),
            karma=data.get('karma', 0),
            first_seen=parse_datetime(data.get('first_seen')),
            last_seen=parse_datetime(data.get('last_seen')),
        )

    
    # ==================== Post Operations ====================
    
    def insert_post(self, post: Post) -> str:
        """Insert or update a post record.
        
        Args:
            post: Post object to insert.
            
        Returns:
            Post ID.
        """
        # Anonymize author ID
        anon_author_id = anonymize_agent_id(post.author_id)
        
        # Ensure submolt exists
        if post.submolt:
            self._ensure_submolt(post.submolt)
        
        self._posts[post.post_id] = {
            'post_id': post.post_id,
            'author_id': anon_author_id,
            'title': post.title,
            'body': post.body,
            'submolt': post.submolt,
            'upvotes': post.upvotes,
            'downvotes': post.downvotes,
            'created_at': post.created_at.isoformat() if post.created_at else None,
            'scraped_at': (post.scraped_at or datetime.now()).isoformat(),
        }
        
        # Update author index for fast lookups
        if anon_author_id:
            if anon_author_id not in self._posts_by_author:
                self._posts_by_author[anon_author_id] = []
            if post.post_id not in self._posts_by_author[anon_author_id]:
                self._posts_by_author[anon_author_id].append(post.post_id)
        
        # Update agent-submolt membership
        if post.submolt and anon_author_id:
            self._update_membership(anon_author_id, post.submolt, post.created_at)
        
        return post.post_id
    
    def get_posts(self, filters: Optional[dict] = None) -> list[Post]:
        """Query posts with optional filters.
        
        Args:
            filters: Optional dict with filter conditions.
            
        Returns:
            List of Post objects.
        """
        posts = []
        
        # Use author index for fast lookup if filtering by author_id only
        if filters and "author_id" in filters and len(filters) == 1:
            author_id = filters["author_id"]
            post_ids = self._posts_by_author.get(author_id, [])
            for post_id in post_ids:
                data = self._posts.get(post_id)
                if data:
                    posts.append(Post(
                        post_id=data['post_id'],
                        author_id=data.get('author_id', ''),
                        title=data.get('title', ''),
                        body=data.get('body', ''),
                        submolt=data.get('submolt', ''),
                        upvotes=data.get('upvotes', 0),
                        downvotes=data.get('downvotes', 0),
                        created_at=parse_datetime(data.get('created_at')),
                        scraped_at=parse_datetime(data.get('scraped_at')),
                    ))
            posts.sort(key=lambda p: p.created_at or datetime.min)
            return posts
        
        for data in self._posts.values():
            # Apply filters
            if filters:
                if "submolt" in filters and data.get("submolt") != filters["submolt"]:
                    continue
                if "author_id" in filters and data.get("author_id") != filters["author_id"]:
                    continue
                if "since" in filters:
                    created_at = parse_datetime(data.get("created_at"))
                    if created_at and created_at < filters["since"]:
                        continue
                if "until" in filters:
                    created_at = parse_datetime(data.get("created_at"))
                    if created_at and created_at > filters["until"]:
                        continue
            
            posts.append(Post(
                post_id=data['post_id'],
                author_id=data.get('author_id', ''),
                title=data.get('title', ''),
                body=data.get('body', ''),
                submolt=data.get('submolt', ''),
                upvotes=data.get('upvotes', 0),
                downvotes=data.get('downvotes', 0),
                created_at=parse_datetime(data.get('created_at')),
                scraped_at=parse_datetime(data.get('scraped_at')),
            ))
        
        # Sort by created_at
        posts.sort(key=lambda p: p.created_at or datetime.min)
        return posts
    
    # ==================== Comment Operations ====================
    
    def insert_comment(self, comment: Comment) -> str:
        """Insert or update a comment record.
        
        Args:
            comment: Comment object to insert.
            
        Returns:
            Comment ID.
        """
        # Anonymize author ID
        anon_author_id = anonymize_agent_id(comment.author_id)
        
        self._comments[comment.comment_id] = {
            'comment_id': comment.comment_id,
            'post_id': comment.post_id,
            'author_id': anon_author_id,
            'parent_comment_id': comment.parent_comment_id,
            'body': comment.body,
            'upvotes': comment.upvotes,
            'downvotes': comment.downvotes,
            'created_at': comment.created_at.isoformat() if comment.created_at else None,
            'scraped_at': (comment.scraped_at or datetime.now()).isoformat(),
        }
        
        # Update author index for fast lookups
        if anon_author_id:
            if anon_author_id not in self._comments_by_author:
                self._comments_by_author[anon_author_id] = []
            if comment.comment_id not in self._comments_by_author[anon_author_id]:
                self._comments_by_author[anon_author_id].append(comment.comment_id)
        
        return comment.comment_id
    
    def get_comments(self, filters: Optional[dict] = None) -> list[Comment]:
        """Query comments with optional filters.
        
        Args:
            filters: Optional dict with filter conditions.
            
        Returns:
            List of Comment objects.
        """
        comments = []
        
        # Use author index for fast lookup if filtering by author_id only
        if filters and "author_id" in filters and len(filters) == 1:
            author_id = filters["author_id"]
            comment_ids = self._comments_by_author.get(author_id, [])
            for comment_id in comment_ids:
                data = self._comments.get(comment_id)
                if data:
                    comments.append(Comment(
                        comment_id=data['comment_id'],
                        post_id=data.get('post_id', ''),
                        author_id=data.get('author_id', ''),
                        parent_comment_id=data.get('parent_comment_id'),
                        body=data.get('body', ''),
                        upvotes=data.get('upvotes', 0),
                        downvotes=data.get('downvotes', 0),
                        created_at=parse_datetime(data.get('created_at')),
                        scraped_at=parse_datetime(data.get('scraped_at')),
                    ))
            comments.sort(key=lambda c: c.created_at or datetime.min)
            return comments
        
        for data in self._comments.values():
            # Apply filters
            if filters:
                if "post_id" in filters and data.get("post_id") != filters["post_id"]:
                    continue
                if "author_id" in filters and data.get("author_id") != filters["author_id"]:
                    continue
                if "since" in filters:
                    created_at = parse_datetime(data.get("created_at"))
                    if created_at and created_at < filters["since"]:
                        continue
            
            comments.append(Comment(
                comment_id=data['comment_id'],
                post_id=data.get('post_id', ''),
                author_id=data.get('author_id', ''),
                parent_comment_id=data.get('parent_comment_id'),
                body=data.get('body', ''),
                upvotes=data.get('upvotes', 0),
                downvotes=data.get('downvotes', 0),
                created_at=parse_datetime(data.get('created_at')),
                scraped_at=parse_datetime(data.get('scraped_at')),
            ))
        
        # Sort by created_at
        comments.sort(key=lambda c: c.created_at or datetime.min)
        return comments
    
    # ==================== Interaction Operations ====================
    
    def insert_interaction(self, interaction: Interaction) -> None:
        """Insert an interaction record.
        
        Args:
            interaction: Interaction object to insert.
        """
        self._interactions.append({
            'source_agent_id': interaction.source_agent_id,
            'target_agent_id': interaction.target_agent_id,
            'interaction_type': interaction.interaction_type,
            'post_id': interaction.post_id,
            'comment_id': interaction.comment_id,
            'timestamp': interaction.timestamp.isoformat() if interaction.timestamp else None,
        })
    
    def get_interactions(
        self, 
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> list[Interaction]:
        """Query interactions with optional time range filter.
        
        Args:
            time_range: Optional (start, end) datetime tuple.
            
        Returns:
            List of Interaction objects.
        """
        interactions = []
        
        for data in self._interactions:
            timestamp = parse_datetime(data.get('timestamp'))
            
            # Apply time range filter
            if time_range:
                if timestamp:
                    if timestamp < time_range[0] or timestamp > time_range[1]:
                        continue
            
            interactions.append(Interaction(
                source_agent_id=data['source_agent_id'],
                target_agent_id=data['target_agent_id'],
                interaction_type=data.get('interaction_type', ''),
                post_id=data.get('post_id'),
                comment_id=data.get('comment_id'),
                timestamp=timestamp,
            ))
        
        # Sort by timestamp
        interactions.sort(key=lambda i: i.timestamp or datetime.min)
        return interactions
    
    def get_interactions_dataframe(
        self,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> pd.DataFrame:
        """Get interactions as a pandas DataFrame.
        
        Args:
            time_range: Optional (start, end) datetime tuple.
            
        Returns:
            DataFrame with interaction records.
        """
        interactions = self.get_interactions(time_range)
        
        if not interactions:
            return pd.DataFrame(columns=[
                'source_agent_id', 'target_agent_id', 'interaction_type',
                'post_id', 'comment_id', 'timestamp'
            ])
        
        return pd.DataFrame([
            {
                'source_agent_id': i.source_agent_id,
                'target_agent_id': i.target_agent_id,
                'interaction_type': i.interaction_type,
                'post_id': i.post_id,
                'comment_id': i.comment_id,
                'timestamp': i.timestamp,
            }
            for i in interactions
        ])

    
    # ==================== Submolt Operations ====================
    
    def _ensure_submolt(self, name: str) -> None:
        """Ensure a submolt exists in storage."""
        if name not in self._submolts:
            self._submolts[name] = {'name': name}
    
    def insert_submolt(self, submolt: Submolt) -> str:
        """Insert or update a submolt record.
        
        Args:
            submolt: Submolt object to insert.
            
        Returns:
            Submolt name.
        """
        self._submolts[submolt.name] = {
            'name': submolt.name,
            'description': submolt.description,
            'member_count': submolt.member_count,
            'post_count': submolt.post_count,
            'created_at': submolt.created_at.isoformat() if submolt.created_at else None,
        }
        
        return submolt.name
    
    def get_submolts(self) -> list[Submolt]:
        """Get all submolts."""
        submolts = []
        
        for data in self._submolts.values():
            submolts.append(Submolt(
                name=data['name'],
                description=data.get('description', ''),
                member_count=data.get('member_count', 0),
                post_count=data.get('post_count', 0),
                created_at=parse_datetime(data.get('created_at')),
            ))
        
        # Sort by post_count descending
        submolts.sort(key=lambda s: s.post_count or 0, reverse=True)
        return submolts
    
    # ==================== Membership Operations ====================
    
    def _update_membership(
        self, 
        agent_id: str, 
        submolt: str, 
        post_time: Optional[datetime]
    ) -> None:
        """Update agent-submolt membership record."""
        key = f"{agent_id}:{submolt}"
        
        if key in self._memberships:
            self._memberships[key]['post_count'] += 1
            if post_time:
                existing_last = parse_datetime(self._memberships[key].get('last_post'))
                if not existing_last or post_time > existing_last:
                    self._memberships[key]['last_post'] = post_time.isoformat()
        else:
            self._memberships[key] = {
                'agent_id': agent_id,
                'submolt_name': submolt,
                'post_count': 1,
                'first_post': post_time.isoformat() if post_time else None,
                'last_post': post_time.isoformat() if post_time else None,
            }
    
    def get_agent_submolt_memberships(self) -> pd.DataFrame:
        """Get agent-submolt membership data as DataFrame.
        
        Returns:
            DataFrame with columns: agent_id, submolt_name, post_count
        """
        if not self._memberships:
            return pd.DataFrame(columns=['agent_id', 'submolt_name', 'post_count'])
        
        return pd.DataFrame([
            {
                'agent_id': m['agent_id'],
                'submolt_name': m['submolt_name'],
                'post_count': m['post_count'],
            }
            for m in self._memberships.values()
        ]).sort_values(['agent_id', 'submolt_name'])
    
    # ==================== Utility Methods ====================
    
    def get_post_author(self, post_id: str) -> Optional[str]:
        """Get the author ID of a post."""
        post = self._posts.get(post_id)
        return post.get('author_id') if post else None
    
    def get_comment_author(self, comment_id: str) -> Optional[str]:
        """Get the author ID of a comment."""
        comment = self._comments.get(comment_id)
        return comment.get('author_id') if comment else None
    
    def get_parent_comment_author(self, comment_id: str) -> Optional[str]:
        """Get the author of a comment's parent comment."""
        comment = self._comments.get(comment_id)
        if not comment or not comment.get('parent_comment_id'):
            return None
        
        parent = self._comments.get(comment['parent_comment_id'])
        return parent.get('author_id') if parent else None
    
    def save(self) -> None:
        """Explicitly save all data to disk."""
        self._save_all()
        logger.info("Data saved to JSON files")
    
    def get_statistics(self) -> dict:
        """Get storage statistics.
        
        Returns:
            Dict with counts of stored items.
        """
        return {
            'agents': len(self._agents),
            'posts': len(self._posts),
            'comments': len(self._comments),
            'interactions': len(self._interactions),
            'submolts': len(self._submolts),
        }


# Alias for backward compatibility
Database = JSONStorage
