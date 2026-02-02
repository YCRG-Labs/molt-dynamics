"""Feature extraction module for Molt Dynamics analysis.

Computes behavioral features for each agent including activity metrics,
topic diversity, network centrality, temporal patterns, and content features.
"""

import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .storage import JSONStorage
from .config import Config
from .models import AgentFeatures

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Computes behavioral features for each agent."""
    
    def __init__(
        self,
        storage: JSONStorage,
        network: nx.DiGraph,
        config: Config,
    ) -> None:
        """Initialize feature extractor.
        
        Args:
            storage: JSONStorage instance for querying agent data.
            network: Directed interaction network.
            config: Configuration parameters.
        """
        self.storage = storage
        self.network = network
        self.config = config
        self._pagerank_cache: Optional[dict] = None
        self._betweenness_cache: Optional[dict] = None
        self._clustering_cache: Optional[dict] = None
    
    def compute_activity_metrics(self, agent_id: str) -> dict:
        """Compute activity metrics for an agent.
        
        Args:
            agent_id: The agent identifier.
            
        Returns:
            Dict with total_posts, total_comments, post_comment_ratio,
            active_lifespan_days, posts_per_day.
        """
        posts = self.storage.get_posts(filters={'author_id': agent_id})
        comments = self.storage.get_comments(filters={'author_id': agent_id})
        
        total_posts = len(posts)
        total_comments = len(comments)
        
        # Post-comment ratio (avoid division by zero)
        if total_comments > 0:
            post_comment_ratio = total_posts / total_comments
        else:
            post_comment_ratio = float(total_posts) if total_posts > 0 else 0.0
        
        # Active lifespan
        timestamps = []
        for post in posts:
            if post.created_at:
                timestamps.append(post.created_at)
        for comment in comments:
            if comment.created_at:
                timestamps.append(comment.created_at)
        
        if len(timestamps) >= 2:
            active_lifespan = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        else:
            active_lifespan = 0.0
        
        # Posts per day
        if active_lifespan > 0:
            posts_per_day = total_posts / active_lifespan
        else:
            posts_per_day = float(total_posts)
        
        return {
            'total_posts': total_posts,
            'total_comments': total_comments,
            'post_comment_ratio': post_comment_ratio,
            'active_lifespan_days': active_lifespan,
            'posts_per_day': posts_per_day,
        }
    
    def compute_topic_diversity(self, agent_id: str) -> tuple[float, float]:
        """Compute topic diversity using Shannon entropy.
        
        Args:
            agent_id: The agent identifier.
            
        Returns:
            Tuple of (topic_entropy, normalized_entropy).
        """
        posts = self.storage.get_posts(filters={'author_id': agent_id})
        
        if not posts:
            return 0.0, 0.0
        
        # Count posts per submolt
        submolt_counts = Counter(post.submolt for post in posts if post.submolt)
        
        if not submolt_counts:
            return 0.0, 0.0
        
        total = sum(submolt_counts.values())
        
        # Compute Shannon entropy: H = -Σ(p * log2(p))
        entropy = 0.0
        for count in submolt_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalized entropy (0 to 1)
        max_entropy = math.log2(len(submolt_counts)) if len(submolt_counts) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return entropy, normalized

    
    def compute_centrality_metrics(self, agent_id: str) -> dict:
        """Compute network centrality metrics for an agent.
        
        Args:
            agent_id: The agent identifier.
            
        Returns:
            Dict with in_degree, out_degree, betweenness, 
            clustering_coefficient, pagerank.
        """
        # Check if agent is in network
        if agent_id not in self.network:
            return {
                'in_degree': 0,
                'out_degree': 0,
                'betweenness': 0.0,
                'clustering_coefficient': 0.0,
                'pagerank': 0.0,
            }
        
        # Degree centrality
        in_degree = self.network.in_degree(agent_id)
        out_degree = self.network.out_degree(agent_id)
        
        # Betweenness centrality (cached for efficiency)
        if self._betweenness_cache is None:
            self._betweenness_cache = nx.betweenness_centrality(self.network)
        betweenness = self._betweenness_cache.get(agent_id, 0.0)
        
        # Clustering coefficient (on undirected version)
        if self._clustering_cache is None:
            undirected = self.network.to_undirected()
            self._clustering_cache = nx.clustering(undirected)
        clustering = self._clustering_cache.get(agent_id, 0.0)
        
        # PageRank
        if self._pagerank_cache is None:
            self._pagerank_cache = nx.pagerank(
                self.network, 
                alpha=self.config.pagerank_damping
            )
        pagerank = self._pagerank_cache.get(agent_id, 0.0)
        
        return {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'betweenness': betweenness,
            'clustering_coefficient': clustering,
            'pagerank': pagerank,
        }
    
    def compute_temporal_features(self, agent_id: str) -> dict:
        """Compute temporal pattern features for an agent.
        
        Args:
            agent_id: The agent identifier.
            
        Returns:
            Dict with hour_distribution (24-dim), autocorrelation, burst_coefficient.
        """
        posts = self.storage.get_posts(filters={'author_id': agent_id})
        comments = self.storage.get_comments(filters={'author_id': agent_id})
        
        # Collect all timestamps
        timestamps = []
        for post in posts:
            if post.created_at:
                timestamps.append(post.created_at)
        for comment in comments:
            if comment.created_at:
                timestamps.append(comment.created_at)
        
        if not timestamps:
            return {
                'hour_distribution': np.zeros(24),
                'autocorrelation': 0.0,
                'burst_coefficient': 0.0,
            }
        
        # Hour distribution (24-dimensional)
        hour_counts = np.zeros(24)
        for ts in timestamps:
            hour_counts[ts.hour] += 1
        
        # Normalize to probability distribution
        total = hour_counts.sum()
        if total > 0:
            hour_distribution = hour_counts / total
        else:
            hour_distribution = hour_counts
        
        # Autocorrelation of inter-event times
        autocorrelation = 0.0
        if len(timestamps) >= 3:
            timestamps_sorted = sorted(timestamps)
            inter_times = [
                (timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds()
                for i in range(len(timestamps_sorted) - 1)
            ]
            if len(inter_times) >= 2 and np.std(inter_times) > 0:
                autocorrelation = np.corrcoef(inter_times[:-1], inter_times[1:])[0, 1]
                if np.isnan(autocorrelation):
                    autocorrelation = 0.0
        
        # Burst coefficient (coefficient of variation of inter-event times)
        burst_coefficient = 0.0
        if len(timestamps) >= 2:
            timestamps_sorted = sorted(timestamps)
            inter_times = [
                (timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds()
                for i in range(len(timestamps_sorted) - 1)
            ]
            mean_inter = np.mean(inter_times)
            std_inter = np.std(inter_times)
            if mean_inter > 0:
                burst_coefficient = std_inter / mean_inter
        
        return {
            'hour_distribution': hour_distribution,
            'autocorrelation': autocorrelation,
            'burst_coefficient': burst_coefficient,
        }
    
    def compute_content_features(self, agent_id: str) -> dict:
        """Compute content-based features for an agent.
        
        Args:
            agent_id: The agent identifier.
            
        Returns:
            Dict with avg_post_length, vocabulary_diversity, 
            avg_sentiment, technical_density.
        """
        posts = self.storage.get_posts(filters={'author_id': agent_id})
        comments = self.storage.get_comments(filters={'author_id': agent_id})
        
        # Collect all text content
        texts = []
        for post in posts:
            if post.body:
                texts.append(post.body)
            if post.title:
                texts.append(post.title)
        for comment in comments:
            if comment.body:
                texts.append(comment.body)
        
        if not texts:
            return {
                'avg_post_length': 0.0,
                'vocabulary_diversity': 0.0,
                'avg_sentiment': 0.0,
                'technical_density': 0.0,
            }
        
        # Average post length
        avg_length = np.mean([len(t) for t in texts])
        
        # Vocabulary diversity (Type-Token Ratio)
        all_words = ' '.join(texts).lower().split()
        if all_words:
            unique_words = set(all_words)
            vocabulary_diversity = len(unique_words) / len(all_words)
        else:
            vocabulary_diversity = 0.0
        
        # Sentiment analysis using VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            sentiments = [analyzer.polarity_scores(t)['compound'] for t in texts]
            avg_sentiment = np.mean(sentiments)
        except ImportError:
            logger.warning("vaderSentiment not available, using neutral sentiment")
            avg_sentiment = 0.0
        
        # Technical density (ratio of technical terms)
        technical_keywords = {
            'function', 'class', 'method', 'variable', 'parameter', 'return',
            'import', 'module', 'package', 'library', 'api', 'endpoint',
            'database', 'query', 'schema', 'table', 'index', 'key',
            'algorithm', 'data', 'structure', 'array', 'list', 'dict',
            'loop', 'condition', 'if', 'else', 'for', 'while', 'try',
            'error', 'exception', 'debug', 'test', 'unit', 'integration',
            'deploy', 'server', 'client', 'request', 'response', 'http',
            'json', 'xml', 'html', 'css', 'javascript', 'python', 'java',
            'code', 'script', 'compile', 'runtime', 'memory', 'cpu',
        }
        
        if all_words:
            technical_count = sum(1 for w in all_words if w in technical_keywords)
            technical_density = technical_count / len(all_words)
        else:
            technical_density = 0.0
        
        return {
            'avg_post_length': avg_length,
            'vocabulary_diversity': vocabulary_diversity,
            'avg_sentiment': avg_sentiment,
            'technical_density': technical_density,
        }

    
    def extract_all_features(self) -> pd.DataFrame:
        """Extract all features for all agents.
        
        Returns:
            DataFrame with one row per agent and all feature columns.
        """
        agents = self.storage.get_agents()
        
        if not agents:
            logger.warning("No agents found in database")
            return pd.DataFrame()
        
        features_list = []
        
        for agent in agents:
            agent_id = agent.agent_id
            
            # Compute all feature groups
            activity = self.compute_activity_metrics(agent_id)
            topic_entropy, normalized_entropy = self.compute_topic_diversity(agent_id)
            centrality = self.compute_centrality_metrics(agent_id)
            temporal = self.compute_temporal_features(agent_id)
            content = self.compute_content_features(agent_id)
            
            # Combine into single dict
            features = {
                'agent_id': agent_id,
                **activity,
                'topic_entropy': topic_entropy,
                'normalized_entropy': normalized_entropy,
                **centrality,
                'autocorrelation': temporal['autocorrelation'],
                'burst_coefficient': temporal['burst_coefficient'],
                **content,
            }
            
            # Add hour distribution as separate columns
            for i, val in enumerate(temporal['hour_distribution']):
                features[f'hour_{i}'] = val
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        logger.info(f"Extracted features for {len(df)} agents")
        
        return df
    
    def standardize_features(
        self, 
        df: pd.DataFrame,
        exclude_cols: list[str] = None,
    ) -> pd.DataFrame:
        """Standardize features to zero mean and unit variance.
        
        Args:
            df: DataFrame with feature columns.
            exclude_cols: Columns to exclude from standardization.
            
        Returns:
            DataFrame with standardized features.
        """
        if exclude_cols is None:
            exclude_cols = ['agent_id']
        
        # Identify numeric columns to standardize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_standardize = [c for c in numeric_cols if c not in exclude_cols]
        
        if not cols_to_standardize:
            return df.copy()
        
        # Create copy and standardize
        df_std = df.copy()
        scaler = StandardScaler()
        
        # Handle constant columns (std=0)
        valid_cols = []
        for col in cols_to_standardize:
            if df[col].std() > 0:
                valid_cols.append(col)
            else:
                logger.warning(f"Skipping constant column: {col}")
        
        if valid_cols:
            df_std[valid_cols] = scaler.fit_transform(df[valid_cols])
        
        return df_std


class TopicModeler:
    """LDA topic modeling for agent content analysis."""
    
    def __init__(
        self,
        n_topics: int = 20,
        random_state: int = 42,
    ) -> None:
        """Initialize topic modeler.
        
        Args:
            n_topics: Number of topics for LDA.
            random_state: Random seed for reproducibility.
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=50,
        )
        self._fitted = False
    
    def fit(self, documents: list[str]) -> None:
        """Fit LDA model on documents.
        
        Args:
            documents: List of text documents.
        """
        if not documents:
            logger.warning("No documents provided for topic modeling")
            return
        
        # Vectorize documents
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit LDA
        self.lda.fit(doc_term_matrix)
        self._fitted = True
        
        logger.info(f"Fitted LDA model with {self.n_topics} topics on {len(documents)} documents")
    
    def transform(self, documents: list[str]) -> np.ndarray:
        """Transform documents to topic distributions.
        
        Args:
            documents: List of text documents.
            
        Returns:
            Array of shape (n_documents, n_topics) with topic distributions.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before transform")
        
        if not documents:
            return np.zeros((0, self.n_topics))
        
        doc_term_matrix = self.vectorizer.transform(documents)
        return self.lda.transform(doc_term_matrix)
    
    def get_agent_topic_distribution(
        self,
        storage: JSONStorage,
        agent_id: str,
    ) -> np.ndarray:
        """Get topic distribution for an agent's content.
        
        Args:
            storage: JSONStorage instance.
            agent_id: The agent identifier.
            
        Returns:
            Topic distribution vector of shape (n_topics,).
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before getting distributions")
        
        posts = storage.get_posts(filters={'author_id': agent_id})
        comments = storage.get_comments(filters={'author_id': agent_id})
        
        # Combine all text
        texts = []
        for post in posts:
            text_parts = []
            if post.title:
                text_parts.append(post.title)
            if post.body:
                text_parts.append(post.body)
            if text_parts:
                texts.append(' '.join(text_parts))
        
        for comment in comments:
            if comment.body:
                texts.append(comment.body)
        
        if not texts:
            return np.zeros(self.n_topics)
        
        # Get topic distributions and average
        distributions = self.transform(texts)
        return distributions.mean(axis=0)
    
    def get_top_words(self, n_words: int = 10) -> list[list[str]]:
        """Get top words for each topic.
        
        Args:
            n_words: Number of top words per topic.
            
        Returns:
            List of word lists, one per topic.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        feature_names = self.vectorizer.get_feature_names_out()
        top_words = []
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words.append([feature_names[i] for i in top_indices])
        
        return top_words


def compute_shannon_entropy(distribution: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.
    
    Args:
        distribution: Probability distribution (must sum to 1).
        
    Returns:
        Shannon entropy in bits.
    """
    # Filter out zero probabilities
    p = distribution[distribution > 0]
    
    if len(p) == 0:
        return 0.0
    
    # H = -Σ(p * log2(p))
    return -np.sum(p * np.log2(p))


def normalize_entropy(entropy: float, n_categories: int) -> float:
    """Normalize entropy to [0, 1] range.
    
    Args:
        entropy: Raw Shannon entropy.
        n_categories: Number of categories in distribution.
        
    Returns:
        Normalized entropy (0 = concentrated, 1 = uniform).
    """
    if n_categories <= 1:
        return 0.0
    
    max_entropy = math.log2(n_categories)
    return entropy / max_entropy if max_entropy > 0 else 0.0
