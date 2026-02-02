"""RQ1: Role emergence analysis module.

Implements spontaneous specialization analysis using k-means clustering,
t-SNE visualization, and role taxonomy classification.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config

logger = logging.getLogger(__name__)


class RoleAnalyzer:
    """Analyzes emergent roles in agent behavior through clustering."""
    
    def __init__(
        self,
        features: pd.DataFrame,
        config: Config,
    ) -> None:
        """Initialize role analyzer.
        
        Args:
            features: DataFrame with standardized agent features.
            config: Configuration parameters.
        """
        self.features = features
        self.config = config
        self._feature_cols = [
            c for c in features.columns 
            if c != 'agent_id' and features[c].dtype in [np.float64, np.int64]
        ]
        self._feature_matrix = features[self._feature_cols].values
        self._cluster_labels: Optional[np.ndarray] = None
        self._kmeans_model: Optional[KMeans] = None
        self._tsne_embedding: Optional[np.ndarray] = None
    
    def find_optimal_k(self) -> tuple[int, dict]:
        """Find optimal number of clusters using silhouette analysis.
        
        Returns:
            Tuple of (optimal_k, dict of k -> silhouette_score).
        """
        k_min, k_max = self.config.kmeans_k_range
        silhouette_scores = {}
        
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(
                n_clusters=k,
                n_init=self.config.kmeans_n_init,
                random_state=self.config.random_seed,
            )
            labels = kmeans.fit_predict(self._feature_matrix)
            
            # Compute silhouette score
            score = silhouette_score(self._feature_matrix, labels)
            silhouette_scores[k] = score
            
            logger.info(f"k={k}: silhouette score = {score:.4f}")
        
        # Find optimal k (highest silhouette score)
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        
        logger.info(f"Optimal k = {optimal_k} with silhouette = {silhouette_scores[optimal_k]:.4f}")
        
        return optimal_k, silhouette_scores
    
    def perform_clustering(self, k: int) -> np.ndarray:
        """Perform k-means clustering with specified k.
        
        Args:
            k: Number of clusters.
            
        Returns:
            Array of cluster labels for each agent.
        """
        self._kmeans_model = KMeans(
            n_clusters=k,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.random_seed,
        )
        self._cluster_labels = self._kmeans_model.fit_predict(self._feature_matrix)
        
        # Log cluster sizes
        unique, counts = np.unique(self._cluster_labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            logger.info(f"Cluster {cluster}: {count} agents ({100*count/len(self._cluster_labels):.1f}%)")
        
        return self._cluster_labels
    
    def compute_silhouette_scores(self) -> dict[int, float]:
        """Compute silhouette scores for each cluster.
        
        Returns:
            Dict mapping cluster label to average silhouette score.
        """
        if self._cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        sample_scores = silhouette_samples(self._feature_matrix, self._cluster_labels)
        
        cluster_scores = {}
        for label in np.unique(self._cluster_labels):
            mask = self._cluster_labels == label
            cluster_scores[label] = sample_scores[mask].mean()
        
        return cluster_scores

    
    def compute_tsne_embedding(
        self,
        perplexity: int = 30,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """Compute t-SNE embedding for visualization.
        
        Args:
            perplexity: t-SNE perplexity parameter.
            max_iter: Maximum number of iterations.
            
        Returns:
            2D embedding array of shape (n_agents, 2).
        """
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(self._feature_matrix) - 1),
            max_iter=max_iter,
            random_state=self.config.random_seed,
        )
        self._tsne_embedding = tsne.fit_transform(self._feature_matrix)
        
        logger.info(f"Computed t-SNE embedding with shape {self._tsne_embedding.shape}")
        
        return self._tsne_embedding
    
    def plot_role_clusters(
        self,
        output_path: str,
        figsize: tuple = (10, 8),
    ) -> None:
        """Generate t-SNE visualization of role clusters.
        
        Args:
            output_path: Path to save the figure.
            figsize: Figure size in inches.
        """
        if self._tsne_embedding is None:
            self.compute_tsne_embedding()
        
        if self._cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            self._tsne_embedding[:, 0],
            self._tsne_embedding[:, 1],
            c=self._cluster_labels,
            cmap='tab10',
            alpha=0.7,
            s=50,
        )
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('Agent Role Clusters (t-SNE Projection)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved role cluster plot to {output_path}")
    
    def classify_roles(self) -> pd.Series:
        """Classify agents into role taxonomy based on feature thresholds.
        
        Role taxonomy:
        - Specialists: High topic concentration, low diversity
        - Connectors: High betweenness centrality, moderate activity
        - Initiators: High post count, low comment ratio
        - Synthesizers: High comment ratio, high in-degree
        - Generalists: Balanced features across all dimensions
        
        Returns:
            Series mapping agent_id to role name.
        """
        roles = []
        
        for idx, row in self.features.iterrows():
            role = self._classify_single_agent(row)
            roles.append(role)
        
        role_series = pd.Series(roles, index=self.features['agent_id'])
        
        # Log role distribution
        role_counts = role_series.value_counts()
        for role, count in role_counts.items():
            logger.info(f"Role '{role}': {count} agents ({100*count/len(role_series):.1f}%)")
        
        return role_series
    
    def _classify_single_agent(self, row: pd.Series) -> str:
        """Classify a single agent based on feature thresholds.
        
        Args:
            row: Feature row for the agent.
            
        Returns:
            Role name string.
        """
        # Get feature values (use standardized values, so 0 is mean)
        normalized_entropy = row.get('normalized_entropy', 0)
        betweenness = row.get('betweenness', 0)
        post_comment_ratio = row.get('post_comment_ratio', 0)
        in_degree = row.get('in_degree', 0)
        total_posts = row.get('total_posts', 0)
        
        # Classification rules (based on standardized features)
        # Specialists: Low diversity (below mean)
        if normalized_entropy < -0.5:
            return 'Specialist'
        
        # Connectors: High betweenness (above mean)
        if betweenness > 0.5:
            return 'Connector'
        
        # Initiators: High post ratio, many posts
        if post_comment_ratio > 0.5 and total_posts > 0:
            return 'Initiator'
        
        # Synthesizers: Low post ratio (more comments), high in-degree
        if post_comment_ratio < -0.5 and in_degree > 0:
            return 'Synthesizer'
        
        # Default: Generalist
        return 'Generalist'
    
    def compute_specialization_over_time(
        self,
        time_column: str = 'first_seen',
    ) -> pd.DataFrame:
        """Compute specialization indices over time.
        
        Args:
            time_column: Column name for temporal ordering.
            
        Returns:
            DataFrame with time-dependent specialization metrics.
        """
        if time_column not in self.features.columns:
            logger.warning(f"Time column '{time_column}' not found, using index")
            time_values = np.arange(len(self.features))
        else:
            time_values = self.features[time_column]
        
        # Compute specialization index (inverse of normalized entropy)
        specialization = 1 - self.features.get('normalized_entropy', pd.Series([0.5] * len(self.features)))
        
        result = pd.DataFrame({
            'agent_id': self.features['agent_id'],
            'time': time_values,
            'specialization_index': specialization,
        })
        
        return result.sort_values('time')
    
    def fit_mixed_effects_model(self) -> dict:
        """Fit linear mixed-effects model for temporal specialization.
        
        Returns:
            Dict with model coefficients and statistics.
        """
        try:
            import statsmodels.formula.api as smf
            
            temporal_data = self.compute_specialization_over_time()
            
            # Simple linear regression as approximation
            # (Full mixed-effects would require repeated measures)
            from scipy import stats
            
            x = np.arange(len(temporal_data))
            y = temporal_data['specialization_index'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err,
            }
        except ImportError:
            logger.warning("statsmodels not available for mixed-effects model")
            return {}
    
    def get_cluster_profiles(self) -> pd.DataFrame:
        """Get feature profiles for each cluster.
        
        Returns:
            DataFrame with mean feature values per cluster.
        """
        if self._cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        df = self.features.copy()
        df['cluster'] = self._cluster_labels
        
        # Compute mean features per cluster
        profiles = df.groupby('cluster')[self._feature_cols].mean()
        
        return profiles
    
    def get_cluster_assignments(self) -> pd.DataFrame:
        """Get cluster assignments for all agents.
        
        Returns:
            DataFrame with agent_id and cluster columns.
        """
        if self._cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        return pd.DataFrame({
            'agent_id': self.features['agent_id'],
            'cluster': self._cluster_labels,
        })
