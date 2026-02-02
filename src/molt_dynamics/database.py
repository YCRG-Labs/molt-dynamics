"""Database module for Molt Dynamics analysis.

Provides PostgreSQL interface with connection pooling and query builders.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
import hashlib

import pandas as pd

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

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


class Database:
    """PostgreSQL database interface with connection pooling."""
    
    def __init__(self, config: Config) -> None:
        """Initialize database connection.
        
        Args:
            config: Configuration object with database settings.
        """
        self.config = config
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._schema_path = Path(__file__).parent / "schema.sql"
    
    def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 is required for database operations")
        
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.config.db_host,
                port=self.config.db_port,
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
            )
            logger.info(f"Connected to database {self.config.db_name}")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self) -> None:
        """Close all database connections."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connections closed")
    
    @contextmanager
    def get_connection(self) -> Generator:
        """Get a connection from the pool.
        
        Yields:
            Database connection that is automatically returned to pool.
        """
        if not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)
    
    def initialize_schema(self) -> None:
        """Create database tables from schema file."""
        if not self._schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self._schema_path}")
        
        schema_sql = self._schema_path.read_text()
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        
        logger.info("Database schema initialized")
    
    # ==================== Agent Operations ====================
    
    def insert_agent(self, agent: Agent) -> str:
        """Insert or update an agent record.
        
        Args:
            agent: Agent object to insert.
            
        Returns:
            Anonymized agent ID.
        """
        anon_id = anonymize_agent_id(agent.agent_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agents (
                        agent_id, username, join_date, bio, post_count,
                        comment_count, karma, first_seen, last_seen
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (agent_id) DO UPDATE SET
                        username = EXCLUDED.username,
                        bio = EXCLUDED.bio,
                        post_count = EXCLUDED.post_count,
                        comment_count = EXCLUDED.comment_count,
                        karma = EXCLUDED.karma,
                        last_seen = EXCLUDED.last_seen
                """, (
                    anon_id, agent.username, agent.join_date, agent.bio,
                    agent.post_count, agent.comment_count, agent.karma,
                    agent.first_seen, agent.last_seen
                ))
        
        return anon_id
    
    def get_agents(self, filters: Optional[dict] = None) -> list[Agent]:
        """Query agents with optional filters.
        
        Args:
            filters: Optional dict with filter conditions.
            
        Returns:
            List of Agent objects.
        """
        query = "SELECT * FROM agents"
        params = []
        
        if filters:
            conditions = []
            if "min_posts" in filters:
                conditions.append("post_count >= %s")
                params.append(filters["min_posts"])
            if "min_karma" in filters:
                conditions.append("karma >= %s")
                params.append(filters["min_karma"])
            if "since" in filters:
                conditions.append("first_seen >= %s")
                params.append(filters["since"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        
        return [Agent(**row) for row in rows]
    
    def get_agent_count(self) -> int:
        """Get total number of agents."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM agents")
                return cur.fetchone()[0]
    
    # ==================== Post Operations ====================
    
    def insert_post(self, post: Post) -> str:
        """Insert or update a post record.
        
        Args:
            post: Post object to insert.
            
        Returns:
            Post ID.
        """
        # Ensure submolt exists
        if post.submolt:
            self._ensure_submolt(post.submolt)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO posts (
                        post_id, author_id, title, body, submolt,
                        upvotes, downvotes, created_at, scraped_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (post_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        upvotes = EXCLUDED.upvotes,
                        downvotes = EXCLUDED.downvotes,
                        scraped_at = EXCLUDED.scraped_at
                """, (
                    post.post_id, post.author_id, post.title, post.body,
                    post.submolt or None, post.upvotes, post.downvotes,
                    post.created_at, post.scraped_at or datetime.now()
                ))
        
        # Update agent-submolt membership
        if post.submolt and post.author_id:
            self._update_membership(post.author_id, post.submolt, post.created_at)
        
        return post.post_id
    
    def get_posts(self, filters: Optional[dict] = None) -> list[Post]:
        """Query posts with optional filters.
        
        Args:
            filters: Optional dict with filter conditions.
            
        Returns:
            List of Post objects.
        """
        query = "SELECT * FROM posts"
        params = []
        
        if filters:
            conditions = []
            if "submolt" in filters:
                conditions.append("submolt = %s")
                params.append(filters["submolt"])
            if "author_id" in filters:
                conditions.append("author_id = %s")
                params.append(filters["author_id"])
            if "since" in filters:
                conditions.append("created_at >= %s")
                params.append(filters["since"])
            if "until" in filters:
                conditions.append("created_at <= %s")
                params.append(filters["until"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at"
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        
        return [Post(**row) for row in rows]
    
    # ==================== Comment Operations ====================
    
    def insert_comment(self, comment: Comment) -> str:
        """Insert or update a comment record.
        
        Args:
            comment: Comment object to insert.
            
        Returns:
            Comment ID.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO comments (
                        comment_id, post_id, author_id, parent_comment_id,
                        body, upvotes, downvotes, created_at, scraped_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (comment_id) DO UPDATE SET
                        body = EXCLUDED.body,
                        upvotes = EXCLUDED.upvotes,
                        downvotes = EXCLUDED.downvotes,
                        scraped_at = EXCLUDED.scraped_at
                """, (
                    comment.comment_id, comment.post_id, comment.author_id,
                    comment.parent_comment_id, comment.body, comment.upvotes,
                    comment.downvotes, comment.created_at,
                    comment.scraped_at or datetime.now()
                ))
        
        return comment.comment_id
    
    def get_comments(self, filters: Optional[dict] = None) -> list[Comment]:
        """Query comments with optional filters.
        
        Args:
            filters: Optional dict with filter conditions.
            
        Returns:
            List of Comment objects.
        """
        query = "SELECT * FROM comments"
        params = []
        
        if filters:
            conditions = []
            if "post_id" in filters:
                conditions.append("post_id = %s")
                params.append(filters["post_id"])
            if "author_id" in filters:
                conditions.append("author_id = %s")
                params.append(filters["author_id"])
            if "since" in filters:
                conditions.append("created_at >= %s")
                params.append(filters["since"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at"
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        
        return [Comment(**row) for row in rows]
    
    # ==================== Interaction Operations ====================
    
    def insert_interaction(self, interaction: Interaction) -> None:
        """Insert an interaction record.
        
        Args:
            interaction: Interaction object to insert.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO interactions (
                        source_agent_id, target_agent_id, interaction_type,
                        post_id, comment_id, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    interaction.source_agent_id, interaction.target_agent_id,
                    interaction.interaction_type, interaction.post_id,
                    interaction.comment_id, interaction.timestamp
                ))
    
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
        query = "SELECT * FROM interactions"
        params = []
        
        if time_range:
            query += " WHERE timestamp >= %s AND timestamp <= %s"
            params.extend(time_range)
        
        query += " ORDER BY timestamp"
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        
        return [Interaction(**row) for row in rows]
    
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
        query = "SELECT * FROM interactions"
        params = []
        
        if time_range:
            query += " WHERE timestamp >= %s AND timestamp <= %s"
            params.extend(time_range)
        
        query += " ORDER BY timestamp"
        
        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=params or None)
    
    # ==================== Submolt Operations ====================
    
    def _ensure_submolt(self, name: str) -> None:
        """Ensure a submolt exists in the database."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO submolts (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO NOTHING
                """, (name,))
    
    def insert_submolt(self, submolt: Submolt) -> str:
        """Insert or update a submolt record.
        
        Args:
            submolt: Submolt object to insert.
            
        Returns:
            Submolt name.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO submolts (
                        name, description, member_count, post_count, created_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        description = EXCLUDED.description,
                        member_count = EXCLUDED.member_count,
                        post_count = EXCLUDED.post_count
                """, (
                    submolt.name, submolt.description, submolt.member_count,
                    submolt.post_count, submolt.created_at
                ))
        
        return submolt.name
    
    def get_submolts(self) -> list[Submolt]:
        """Get all submolts."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM submolts ORDER BY post_count DESC")
                rows = cur.fetchall()
        
        return [Submolt(**row) for row in rows]
    
    # ==================== Membership Operations ====================
    
    def _update_membership(
        self, 
        agent_id: str, 
        submolt: str, 
        post_time: Optional[datetime]
    ) -> None:
        """Update agent-submolt membership record."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_submolt_membership (
                        agent_id, submolt_name, post_count, first_post, last_post
                    ) VALUES (%s, %s, 1, %s, %s)
                    ON CONFLICT (agent_id, submolt_name) DO UPDATE SET
                        post_count = agent_submolt_membership.post_count + 1,
                        last_post = GREATEST(
                            agent_submolt_membership.last_post, 
                            EXCLUDED.last_post
                        )
                """, (agent_id, submolt, post_time, post_time))
    
    def get_agent_submolt_memberships(self) -> pd.DataFrame:
        """Get agent-submolt membership data as DataFrame.
        
        Returns:
            DataFrame with columns: agent_id, submolt_name, post_count
        """
        query = """
            SELECT agent_id, submolt_name, post_count
            FROM agent_submolt_membership
            ORDER BY agent_id, submolt_name
        """
        
        with self.get_connection() as conn:
            return pd.read_sql(query, conn)
    
    # ==================== Utility Methods ====================
    
    def get_post_author(self, post_id: str) -> Optional[str]:
        """Get the author ID of a post."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT author_id FROM posts WHERE post_id = %s",
                    (post_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None
    
    def get_comment_author(self, comment_id: str) -> Optional[str]:
        """Get the author ID of a comment."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT author_id FROM comments WHERE comment_id = %s",
                    (comment_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None
    
    def get_parent_comment_author(self, comment_id: str) -> Optional[str]:
        """Get the author of a comment's parent comment."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c2.author_id
                    FROM comments c1
                    JOIN comments c2 ON c1.parent_comment_id = c2.comment_id
                    WHERE c1.comment_id = %s
                """, (comment_id,))
                result = cur.fetchone()
                return result[0] if result else None
