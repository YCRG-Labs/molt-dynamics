"""Dataset loader for MoltBook Observatory Archive from Hugging Face.

Loads pre-collected MoltBook data from the Hugging Face dataset repository
instead of scraping from the API.

Dataset: https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

import pandas as pd

from .config import Config
from .storage import JSONStorage
from .models import Agent, Post, Comment, Submolt

logger = logging.getLogger(__name__)


class MoltBookDatasetLoader:
    """Loader for MoltBook Observatory Archive dataset from Hugging Face.
    
    The dataset is available at:
    https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
    
    Data is stored as date-partitioned Parquet files:
    - data/agents/*.parquet
    - data/posts/*.parquet
    - data/comments/*.parquet
    - data/submolts/*.parquet
    
    Expected to be cloned locally using:
    git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
    """
    
    def __init__(self, config: Config, storage: JSONStorage, dataset_path: str) -> None:
        """Initialize dataset loader.
        
        Args:
            config: Configuration object.
            storage: JSONStorage instance for storing loaded data.
            dataset_path: Path to cloned dataset repository.
        """
        self.config = config
        self.storage = storage
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Please clone it first:\n"
                "git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive"
            )
        
        # Verify data directory exists
        self.data_dir = self.dataset_path / "data"
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found at {self.data_dir}. "
                "The dataset may not be properly cloned."
            )
        
        logger.info(f"Initialized dataset loader from {self.dataset_path}")
    
    def _find_parquet_files(self, subdirectory: str) -> list[Path]:
        """Find all Parquet files in a data subdirectory.
        
        Args:
            subdirectory: Name of subdirectory (e.g., 'posts', 'agents').
            
        Returns:
            List of Path objects for Parquet files, sorted by date.
        """
        subdir_path = self.data_dir / subdirectory
        if not subdir_path.exists():
            logger.warning(f"Subdirectory not found: {subdir_path}")
            return []
        
        files = sorted(subdir_path.glob("*.parquet"))
        logger.info(f"Found {len(files)} Parquet files in {subdirectory}/")
        return files
    
    def _load_parquet_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load a Parquet file into a DataFrame.
        
        Args:
            filepath: Path to Parquet file.
            
        Returns:
            DataFrame or None on error.
        """
        try:
            df = pd.read_parquet(filepath)
            logger.debug(f"Loaded {len(df)} rows from {filepath.name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None
    
    def _parse_datetime(self, value: any) -> Optional[datetime]:
        """Parse datetime from various formats.
        
        Args:
            value: Datetime string, timestamp, or pandas Timestamp.
            
        Returns:
            Parsed datetime or None.
        """
        if value is None or pd.isna(value):
            return None
        
        if isinstance(value, datetime):
            return value
        
        # Handle pandas Timestamp
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().replace(tzinfo=None)
        
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        
        if isinstance(value, str):
            # Try ISO format variants
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
            ]:
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)
                    return dt
                except ValueError:
                    continue
            
            # Try fromisoformat
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                pass
        
        logger.warning(f"Could not parse datetime: {value}")
        return None
    
    def _row_to_agent(self, row: pd.Series) -> Optional[Agent]:
        """Convert a DataFrame row to an Agent object.
        
        Args:
            row: DataFrame row from agents table.
            
        Returns:
            Agent object or None if parsing fails.
        """
        try:
            # Map observatory schema to our Agent model
            return Agent(
                agent_id=str(row['id']),
                username=str(row.get('name', 'unknown')),
                join_date=self._parse_datetime(row.get('created_at')),
                bio=str(row.get('description', '')),
                post_count=0,  # Will be computed from posts
                comment_count=0,  # Will be computed from comments
                karma=int(row.get('karma', 0)),
                first_seen=self._parse_datetime(row.get('first_seen_at')),
                last_seen=self._parse_datetime(row.get('last_seen_at')),
            )
        except Exception as e:
            logger.warning(f"Failed to parse agent row: {e}")
            return None
    
    def _row_to_post(self, row: pd.Series) -> Optional[Post]:
        """Convert a DataFrame row to a Post object.
        
        Args:
            row: DataFrame row from posts table.
            
        Returns:
            Post object or None if parsing fails.
        """
        try:
            # Map observatory schema to our Post model
            return Post(
                post_id=str(row['id']),
                author_id=str(row['agent_id']),
                title=str(row.get('title', '')),
                body=str(row.get('content', '')),
                submolt=str(row.get('submolt', '')),
                upvotes=int(row.get('score', 0)),
                downvotes=0,  # Not tracked in observatory
                created_at=self._parse_datetime(row.get('created_at')),
                scraped_at=self._parse_datetime(row.get('fetched_at')),
            )
        except Exception as e:
            logger.warning(f"Failed to parse post row: {e}")
            return None
    
    def _row_to_comment(self, row: pd.Series) -> Optional[Comment]:
        """Convert a DataFrame row to a Comment object.
        
        Args:
            row: DataFrame row from comments table.
            
        Returns:
            Comment object or None if parsing fails.
        """
        try:
            # Map observatory schema to our Comment model
            return Comment(
                comment_id=str(row['id']),
                post_id=str(row['post_id']),
                author_id=str(row['agent_id']),
                parent_comment_id=str(row['parent_id']) if pd.notna(row.get('parent_id')) else None,
                body=str(row.get('content', '')),
                upvotes=int(row.get('score', 0)),
                downvotes=0,  # Not tracked in observatory
                created_at=self._parse_datetime(row.get('created_at')),
                scraped_at=self._parse_datetime(row.get('fetched_at')),
            )
        except Exception as e:
            logger.warning(f"Failed to parse comment row: {e}")
            return None
    
    def _row_to_submolt(self, row: pd.Series) -> Optional[Submolt]:
        """Convert a DataFrame row to a Submolt object.
        
        Args:
            row: DataFrame row from submolts table.
            
        Returns:
            Submolt object or None if parsing fails.
        """
        try:
            # Map observatory schema to our Submolt model
            return Submolt(
                name=str(row['name']),
                description=str(row.get('description', '')),
                member_count=int(row.get('subscriber_count', 0)),
                post_count=int(row.get('post_count', 0)),
                created_at=self._parse_datetime(row.get('created_at')),
            )
        except Exception as e:
            logger.warning(f"Failed to parse submolt row: {e}")
            return None
    
    def load_agents(self, max_agents: Optional[int] = None) -> int:
        """Load agents from dataset into storage.
        
        Args:
            max_agents: Maximum number of agents to load (None = all).
            
        Returns:
            Number of agents loaded.
        """
        logger.info("Loading agents from dataset...")
        count = 0
        
        parquet_files = self._find_parquet_files("agents")
        
        for filepath in parquet_files:
            df = self._load_parquet_file(filepath)
            if df is None:
                continue
            
            for _, row in df.iterrows():
                if max_agents and count >= max_agents:
                    logger.info(f"Reached max_agents limit of {max_agents}")
                    return count
                
                agent = self._row_to_agent(row)
                if agent:
                    self.storage.insert_agent(agent)
                    count += 1
                    
                    if count % 1000 == 0:
                        logger.info(f"Loaded {count} agents...")
        
        logger.info(f"Loaded {count} agents from dataset")
        return count
    
    def load_posts(self, max_posts: Optional[int] = None) -> int:
        """Load posts from dataset into storage.
        
        Args:
            max_posts: Maximum number of posts to load (None = all).
            
        Returns:
            Number of posts loaded.
        """
        logger.info("Loading posts from dataset...")
        count = 0
        
        parquet_files = self._find_parquet_files("posts")
        
        for filepath in parquet_files:
            df = self._load_parquet_file(filepath)
            if df is None:
                continue
            
            for _, row in df.iterrows():
                if max_posts and count >= max_posts:
                    logger.info(f"Reached max_posts limit of {max_posts}")
                    return count
                
                post = self._row_to_post(row)
                if post:
                    self.storage.insert_post(post)
                    count += 1
                    
                    if count % 1000 == 0:
                        logger.info(f"Loaded {count} posts...")
        
        logger.info(f"Loaded {count} posts from dataset")
        return count
    
    def load_comments(self, max_comments: Optional[int] = None) -> int:
        """Load comments from dataset into storage.
        
        Args:
            max_comments: Maximum number of comments to load (None = all).
            
        Returns:
            Number of comments loaded.
        """
        logger.info("Loading comments from dataset...")
        count = 0
        
        parquet_files = self._find_parquet_files("comments")
        
        for filepath in parquet_files:
            df = self._load_parquet_file(filepath)
            if df is None:
                continue
            
            for _, row in df.iterrows():
                if max_comments and count >= max_comments:
                    logger.info(f"Reached max_comments limit of {max_comments}")
                    return count
                
                comment = self._row_to_comment(row)
                if comment:
                    self.storage.insert_comment(comment)
                    count += 1
                    
                    if count % 1000 == 0:
                        logger.info(f"Loaded {count} comments...")
        
        logger.info(f"Loaded {count} comments from dataset")
        return count
    
    def load_submolts(self) -> int:
        """Load submolts from dataset into storage.
        
        Returns:
            Number of submolts loaded.
        """
        logger.info("Loading submolts from dataset...")
        count = 0
        
        parquet_files = self._find_parquet_files("submolts")
        
        for filepath in parquet_files:
            df = self._load_parquet_file(filepath)
            if df is None:
                continue
            
            for _, row in df.iterrows():
                submolt = self._row_to_submolt(row)
                if submolt:
                    self.storage.insert_submolt(submolt)
                    count += 1
        
        logger.info(f"Loaded {count} submolts from dataset")
        return count
    
    def load_all(
        self, 
        max_agents: Optional[int] = None,
        max_posts: Optional[int] = None,
        max_comments: Optional[int] = None
    ) -> dict:
        """Load all data from dataset into storage.
        
        Args:
            max_agents: Maximum agents to load (None = all).
            max_posts: Maximum posts to load (None = all).
            max_comments: Maximum comments to load (None = all).
            
        Returns:
            Dict with counts of loaded items.
        """
        logger.info("Loading all data from dataset...")
        start_time = datetime.now()
        
        results = {
            "submolts": 0,
            "agents": 0,
            "posts": 0,
            "comments": 0,
            "start_time": start_time.isoformat(),
        }
        
        # Load in order: submolts, agents, posts, comments
        results["submolts"] = self.load_submolts()
        results["agents"] = self.load_agents(max_agents)
        results["posts"] = self.load_posts(max_posts)
        results["comments"] = self.load_comments(max_comments)
        
        # Save to disk
        self.storage.save()
        
        end_time = datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        
        logger.info(f"Dataset loading complete: {results}")
        return results
