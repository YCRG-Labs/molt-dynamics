"""RQ1: Role emergence analysis module.

Implements spontaneous specialization analysis using multiple clustering algorithms,
t-SNE visualization, and role taxonomy classification with rigorous statistical validation.

Performs two complementary analyses:
1. Network-based clustering (structural roles) - typically yields strong cluster separation
2. Full-feature clustering (behavioral roles) - captures broader behavioral patterns
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import inspect
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

from .config import Config
from .models import HubEmergenceEvent

logger = logging.getLogger(__name__)


class RoleAnalyzer:
    """Analyzes emergent roles in agent behavior through clustering with rigorous validation.
    
    Performs two complementary analyses:
    1. Network-based clustering (structural roles based on network position)
    2. Full-feature clustering (behavioral roles across all dimensions)
    """
    
    NETWORK_FEATURES = ['in_degree', 'out_degree', 'betweenness', 'clustering_coefficient', 'pagerank']
    
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
        
        # All feature columns (excluding agent_id)
        self._all_feature_cols = [
            c for c in features.columns 
            if c != 'agent_id' and features[c].dtype in [np.float64, np.int64]
        ]
        
        # Filter zero-variance columns
        self._all_feature_cols = self._filter_zero_variance_columns(self._all_feature_cols)
        
        # Network features only (for structural clustering)
        self._network_feature_cols = [c for c in self.NETWORK_FEATURES if c in self._all_feature_cols]
        
        # Full features (all available)
        self._full_feature_cols = self._all_feature_cols
        
        # Prepare feature matrices
        # Network features matrix
        if self._network_feature_cols:
            self._network_scaler = StandardScaler()
            self._network_matrix = self._network_scaler.fit_transform(
                features[self._network_feature_cols].values
            )
            logger.info(f"Network clustering: using {len(self._network_feature_cols)} features: {self._network_feature_cols}")
        else:
            self._network_matrix = None
            self._network_scaler = None
            logger.warning("No network features available")
        
        # Full features matrix (with PCA if many features)
        self._full_scaler = StandardScaler()
        raw_full_matrix = self._full_scaler.fit_transform(features[self._full_feature_cols].values)
        
        # Apply PCA if we have many features
        self._pca = None
        self._pca_n_components = None
        if len(self._full_feature_cols) > 10:
            self._pca_n_components = 10
            self._pca = PCA(n_components=self._pca_n_components, random_state=self.config.random_seed)
            self._full_matrix = self._pca.fit_transform(raw_full_matrix)
            variance_explained = np.sum(self._pca.explained_variance_ratio_)
            logger.info(f"Full clustering: PCA reduced {len(self._full_feature_cols)} to {self._pca_n_components} components "
                       f"({variance_explained:.1%} variance)")
        else:
            self._full_matrix = raw_full_matrix
            logger.info(f"Full clustering: using {len(self._full_feature_cols)} features")
        
        # Results storage for both analyses
        self._network_labels: Optional[np.ndarray] = None
        self._full_labels: Optional[np.ndarray] = None
        self._network_kmeans: Optional[KMeans] = None
        self._full_kmeans: Optional[KMeans] = None
        self._network_optimal_k: Optional[int] = None
        self._full_optimal_k: Optional[int] = None
        self._network_validation: dict = {}
        self._full_validation: dict = {}
        self._tsne_embedding: Optional[np.ndarray] = None
    
    def _filter_zero_variance_columns(self, cols: list[str]) -> list[str]:
        """Filter out columns with zero or near-zero variance."""
        valid_cols = []
        removed_cols = []
        
        for col in cols:
            variance = self.features[col].var()
            if variance > 1e-10:
                valid_cols.append(col)
            else:
                removed_cols.append(col)
        
        if removed_cols:
            logger.warning(f"Removed {len(removed_cols)} zero-variance columns: {removed_cols[:5]}...")
        
        logger.info(f"Using {len(valid_cols)} features for clustering (removed {len(removed_cols)} zero-variance)")
        
        return valid_cols
    
    def find_optimal_k(self, feature_set: str = 'network') -> tuple[int, dict]:
        """Find optimal number of clusters using multiple validation metrics.
        
        Args:
            feature_set: 'network' for network features only, 'full' for all features.
        
        Returns:
            Tuple of (optimal_k, dict of k -> validation metrics).
        """
        if feature_set == 'network':
            X = self._network_matrix
            set_name = "network"
        else:
            X = self._full_matrix
            set_name = "full"
        
        if X is None:
            logger.error(f"No {set_name} features available")
            return 3, {}
        
        k_min, k_max = self.config.kmeans_k_range
        validation_scores = {}
        
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(
                n_clusters=k,
                n_init=self.config.kmeans_n_init,
                random_state=self.config.random_seed,
            )
            labels = kmeans.fit_predict(X)
            
            # Compute multiple validation metrics
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            validation_scores[k] = {
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin,
                'inertia': kmeans.inertia_,
            }
            
            logger.info(f"[{set_name}] k={k}: silhouette={silhouette:.4f}, CH={calinski_harabasz:.2f}, DB={davies_bouldin:.4f}")
        
        # Find optimal k (highest silhouette - most interpretable metric)
        optimal_k = max(validation_scores, key=lambda k: validation_scores[k]['silhouette'])
        
        logger.info(f"[{set_name}] Optimal k = {optimal_k} with silhouette = {validation_scores[optimal_k]['silhouette']:.4f}")
        
        return optimal_k, validation_scores
    
    def perform_clustering(self, k: int, feature_set: str = 'network') -> np.ndarray:
        """Perform k-means clustering with specified k.
        
        Args:
            k: Number of clusters.
            feature_set: 'network' for network features only, 'full' for all features.
            
        Returns:
            Array of cluster labels for each agent.
        """
        if feature_set == 'network':
            X = self._network_matrix
            kmeans = KMeans(
                n_clusters=k,
                n_init=self.config.kmeans_n_init,
                random_state=self.config.random_seed,
            )
            self._network_labels = kmeans.fit_predict(X)
            self._network_kmeans = kmeans
            self._network_optimal_k = k
            labels = self._network_labels
        else:
            X = self._full_matrix
            kmeans = KMeans(
                n_clusters=k,
                n_init=self.config.kmeans_n_init,
                random_state=self.config.random_seed,
            )
            self._full_labels = kmeans.fit_predict(X)
            self._full_kmeans = kmeans
            self._full_optimal_k = k
            labels = self._full_labels
        
        # Log cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            logger.info(f"[{feature_set}] Cluster {cluster}: {count} agents ({100*count/len(labels):.1f}%)")
        
        return labels
    
    def compare_clustering_algorithms(self, k: int, feature_set: str = 'network') -> dict:
        """Compare multiple clustering algorithms for robustness.
        
        Args:
            k: Number of clusters.
            feature_set: 'network' or 'full'.
            
        Returns:
            Dict with results from each algorithm and agreement metrics.
        """
        if feature_set == 'network':
            X = self._network_matrix
        else:
            X = self._full_matrix
        
        results = {}
        n_samples = X.shape[0]
        
        # Threshold for using agglomerative clustering (Ward's method requires O(n²) memory)
        # For 20k samples, distance matrix is ~3GB; beyond that we skip or sample
        AGG_MAX_SAMPLES = 20000
        
        # K-Means
        kmeans = KMeans(n_clusters=k, n_init=self.config.kmeans_n_init, random_state=self.config.random_seed)
        kmeans_labels = kmeans.fit_predict(X)
        results['kmeans'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X, kmeans_labels),
            'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels),
            'davies_bouldin': davies_bouldin_score(X, kmeans_labels),
        }
        
        # Agglomerative Clustering - skip for large datasets due to O(n²) memory
        if n_samples <= AGG_MAX_SAMPLES:
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            agg_labels = agg.fit_predict(X)
            results['agglomerative'] = {
                'labels': agg_labels,
                'silhouette': silhouette_score(X, agg_labels),
                'calinski_harabasz': calinski_harabasz_score(X, agg_labels),
                'davies_bouldin': davies_bouldin_score(X, agg_labels),
            }
        else:
            logger.warning(f"[{feature_set}] Skipping agglomerative clustering: {n_samples} samples exceeds {AGG_MAX_SAMPLES} limit (O(n²) memory)")
            results['agglomerative'] = {
                'labels': None,
                'silhouette': None,
                'calinski_harabasz': None,
                'davies_bouldin': None,
                'skipped': True,
                'reason': f'Dataset too large ({n_samples} > {AGG_MAX_SAMPLES})',
            }
        
        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=k, random_state=self.config.random_seed, n_init=3)
        gmm_labels = gmm.fit_predict(X)
        results['gmm'] = {
            'labels': gmm_labels,
            'silhouette': silhouette_score(X, gmm_labels),
            'calinski_harabasz': calinski_harabasz_score(X, gmm_labels),
            'davies_bouldin': davies_bouldin_score(X, gmm_labels),
            'bic': gmm.bic(X),
            'aic': gmm.aic(X),
        }
        
        # Compute pairwise agreement (Adjusted Rand Index)
        algorithms = ['kmeans', 'agglomerative', 'gmm']
        agreement = {}
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                # Skip comparisons involving skipped algorithms
                if results[alg1].get('skipped') or results[alg2].get('skipped'):
                    agreement[f'{alg1}_vs_{alg2}'] = {'ari': None, 'nmi': None, 'skipped': True}
                    continue
                ari = adjusted_rand_score(results[alg1]['labels'], results[alg2]['labels'])
                nmi = normalized_mutual_info_score(results[alg1]['labels'], results[alg2]['labels'])
                agreement[f'{alg1}_vs_{alg2}'] = {'ari': ari, 'nmi': nmi}
        
        results['algorithm_agreement'] = agreement
        
        # Mean agreement across all pairs (excluding skipped)
        ari_values = [v['ari'] for v in agreement.values() if v.get('ari') is not None]
        if ari_values:
            results['mean_ari'] = np.mean(ari_values)
            results['std_ari'] = np.std(ari_values)
            logger.info(f"[{feature_set}] Algorithm agreement: mean ARI = {results['mean_ari']:.4f} +/- {results['std_ari']:.4f}")
        else:
            results['mean_ari'] = None
            results['std_ari'] = None
            logger.info(f"[{feature_set}] Algorithm agreement: only kmeans vs gmm available")
        
        return results
    
    def bootstrap_stability_analysis(self, k: int, feature_set: str = 'network', n_bootstrap: int = 100) -> dict:
        """Assess clustering stability using bootstrap resampling.
        
        Args:
            k: Number of clusters.
            feature_set: 'network' or 'full'.
            n_bootstrap: Number of bootstrap iterations.
            
            Returns:
            Dict with stability metrics and confidence intervals.
        """
        if feature_set == 'network':
            X = self._network_matrix
            ref_labels = self._network_labels
        else:
            X = self._full_matrix
            ref_labels = self._full_labels
        
        if ref_labels is None:
            raise ValueError("Must perform clustering first")
        
        n_samples = len(X)
        
        def run_single_bootstrap(b):
            """Run a single bootstrap iteration."""
            rng = np.random.RandomState(self.config.random_seed + b)
            
            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            
            # Cluster bootstrap sample
            kmeans_boot = KMeans(n_clusters=k, n_init=5, random_state=b)
            boot_labels = kmeans_boot.fit_predict(X_boot)
            
            # Silhouette on bootstrap
            sil = silhouette_score(X_boot, boot_labels)
            
            # ARI with reference (on overlapping samples)
            unique_indices = np.unique(indices)
            ref_subset = ref_labels[unique_indices]
            
            boot_subset = []
            for idx in unique_indices:
                first_pos = np.where(indices == idx)[0][0]
                boot_subset.append(boot_labels[first_pos])
            boot_subset = np.array(boot_subset)
            
            ari = adjusted_rand_score(ref_subset, boot_subset)
            
            # Per-cluster stability (Jaccard index)
            cluster_jaccards = {}
            for cluster_id in range(k):
                ref_members = set(unique_indices[ref_subset == cluster_id])
                boot_members = set(unique_indices[boot_subset == cluster_id])
                
                if len(ref_members) > 0 or len(boot_members) > 0:
                    jaccard = len(ref_members & boot_members) / len(ref_members | boot_members)
                    cluster_jaccards[cluster_id] = jaccard
            
            return sil, ari, cluster_jaccards
        
        # Run bootstrap iterations in parallel
        results_list = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
            delayed(run_single_bootstrap)(b) for b in range(n_bootstrap)
        )
        
        # Aggregate results
        bootstrap_silhouettes = [r[0] for r in results_list]
        bootstrap_aris = [r[1] for r in results_list]
        cluster_stability = {i: [] for i in range(k)}
        for _, _, cluster_jaccards in results_list:
            for cluster_id, jaccard in cluster_jaccards.items():
                cluster_stability[cluster_id].append(jaccard)
        
        # Compute confidence intervals
        def ci_95(arr):
            arr = np.array(arr)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'ci_lower': float(np.percentile(arr, 2.5)),
                'ci_upper': float(np.percentile(arr, 97.5)),
            }
        
        results = {
            'n_bootstrap': n_bootstrap,
            'silhouette': ci_95(bootstrap_silhouettes),
            'ari_stability': ci_95(bootstrap_aris),
            'cluster_stability': {
                cluster_id: ci_95(stab) if stab else {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0}
                for cluster_id, stab in cluster_stability.items()
            },
        }
        
        logger.info(f"[{feature_set}] Bootstrap stability: silhouette = {results['silhouette']['mean']:.4f} "
                   f"[{results['silhouette']['ci_lower']:.4f}, {results['silhouette']['ci_upper']:.4f}]")
        
        return results
    
    def test_role_distribution(self, feature_set: str = 'network') -> dict:
        """Test if role distribution differs from uniform using chi-square test.
        
        Args:
            feature_set: 'network' or 'full'.
        
        Returns:
            Dict with chi-square statistic, p-value, and effect size.
        """
        if feature_set == 'network':
            labels = self._network_labels
        else:
            labels = self._full_labels
        
        if labels is None:
            raise ValueError("Must perform clustering first")
        
        unique, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique)
        n_total = len(labels)
        
        # Expected counts under uniform distribution
        expected = np.full(n_clusters, n_total / n_clusters)
        
        # Chi-square test
        chi2, p_value = stats.chisquare(counts, expected)
        
        # Effect size (Cramer's V for goodness-of-fit)
        cramers_v = np.sqrt(chi2 / (n_total * (n_clusters - 1)))
        
        # Entropy-based measure of evenness
        proportions = counts / n_total
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        max_entropy = np.log(n_clusters)
        evenness = entropy / max_entropy if max_entropy > 0 else 0
        
        results = {
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'cramers_v': float(cramers_v),
            'entropy': float(entropy),
            'evenness': float(evenness),
            'cluster_counts': {int(k): int(v) for k, v in zip(unique, counts)},
            'cluster_proportions': {int(k): float(v/n_total) for k, v in zip(unique, counts)},
        }
        
        logger.info(f"[{feature_set}] Role distribution test: chi2={chi2:.2f}, p={p_value:.4e}, Cramer's V={cramers_v:.4f}")
        
        return results
    
    def compute_silhouette_scores(self, feature_set: str = 'network') -> dict[int, float]:
        """Compute silhouette scores for each cluster.
        
        Args:
            feature_set: 'network' or 'full'.
        
        Returns:
            Dict mapping cluster label to average silhouette score.
        """
        if feature_set == 'network':
            X = self._network_matrix
            labels = self._network_labels
        else:
            X = self._full_matrix
            labels = self._full_labels
        
        if labels is None:
            raise ValueError("Must perform clustering first")
        
        sample_scores = silhouette_samples(X, labels)
        
        cluster_scores = {}
        for label in np.unique(labels):
            mask = labels == label
            cluster_scores[int(label)] = float(sample_scores[mask].mean())
        
        return cluster_scores
    
    def compute_tsne_embedding(
        self,
        feature_set: str = 'network',
        perplexity: int = 30,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """Compute t-SNE embedding for visualization.
        
        Args:
            feature_set: 'network' or 'full'.
            perplexity: t-SNE perplexity parameter.
            max_iter: Maximum number of iterations.
            
        Returns:
            2D embedding array of shape (n_agents, 2).
        """
        if feature_set == 'network':
            X = self._network_matrix
        else:
            X = self._full_matrix
        
        tsne_kwargs = {
            'n_components': 2,
            'perplexity': min(perplexity, len(X) - 1),
            'random_state': self.config.random_seed,
        }
        tsne_init_params = inspect.signature(TSNE.__init__).parameters
        if 'max_iter' in tsne_init_params:
            tsne_kwargs['max_iter'] = max_iter
        else:
            # Older scikit-learn uses n_iter instead of max_iter
            tsne_kwargs['n_iter'] = max_iter

        tsne = TSNE(**tsne_kwargs)
        self._tsne_embedding = tsne.fit_transform(X)
        
        logger.info(f"[{feature_set}] Computed t-SNE embedding with shape {self._tsne_embedding.shape}")
        
        return self._tsne_embedding
    
    def get_cluster_profiles(self, feature_set: str = 'network') -> pd.DataFrame:
        """Get feature profiles for each cluster.
        
        Args:
            feature_set: 'network' or 'full'.
        
        Returns:
            DataFrame with mean feature values per cluster.
        """
        if feature_set == 'network':
            labels = self._network_labels
            feature_cols = self._network_feature_cols
        else:
            labels = self._full_labels
            feature_cols = self._full_feature_cols
        
        if labels is None:
            raise ValueError("Must perform clustering first")
        
        df = self.features.copy()
        df['cluster'] = labels
        
        # Compute mean features per cluster
        profiles = df.groupby('cluster')[feature_cols].mean()
        
        return profiles
    
    def get_cluster_assignments(self, feature_set: str = 'network') -> pd.DataFrame:
        """Get cluster assignments for all agents.
        
        Args:
            feature_set: 'network' or 'full'.
        
        Returns:
            DataFrame with agent_id and cluster columns.
        """
        if feature_set == 'network':
            labels = self._network_labels
        else:
            labels = self._full_labels
        
        if labels is None:
            raise ValueError("Must perform clustering first")
        
        return pd.DataFrame({
            'agent_id': self.features['agent_id'],
            'cluster': labels,
        })

    
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
            temporal_data = self.compute_specialization_over_time()
            
            # Simple linear regression as approximation
            x = np.arange(len(temporal_data))
            y = temporal_data['specialization_index'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'std_err': float(std_err),
            }
        except Exception as e:
            logger.warning(f"Could not fit temporal model: {e}")
            return {}
    
    def _run_single_analysis(self, feature_set: str, output_path: Path) -> dict:
        """Run complete analysis for one feature set.
        
        Args:
            feature_set: 'network' or 'full'.
            output_path: Directory to save results.
            
        Returns:
            Dict with analysis results and file paths.
        """
        prefix = f"rq1_{feature_set}"
        results = {}
        saved_files = {}
        
        logger.info(f"=== Running {feature_set.upper()} feature analysis ===")
        
        # 1. Find optimal k
        optimal_k, validation_scores = self.find_optimal_k(feature_set)
        
        # Save silhouette analysis
        silhouette_analysis = []
        for k, metrics in validation_scores.items():
            silhouette_analysis.append({
                'k': k,
                'silhouette_score': metrics['silhouette'],
                'calinski_harabasz': metrics['calinski_harabasz'],
                'davies_bouldin': metrics['davies_bouldin'],
                'inertia': metrics['inertia'],
            })
        silhouette_df = pd.DataFrame(silhouette_analysis)
        silhouette_df.to_csv(output_path / f'{prefix}_silhouette_analysis.csv', index=False)
        saved_files['silhouette_analysis'] = str(output_path / f'{prefix}_silhouette_analysis.csv')
        
        # 2. Perform clustering
        self.perform_clustering(int(optimal_k), feature_set)
        
        # 3. Algorithm comparison
        algo_comparison = self.compare_clustering_algorithms(int(optimal_k), feature_set)
        algo_summary = {
            'kmeans': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in algo_comparison['kmeans'].items() if k != 'labels'},
            'agglomerative': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in algo_comparison['agglomerative'].items() if k != 'labels'},
            'gmm': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in algo_comparison['gmm'].items() if k != 'labels'},
            'algorithm_agreement': algo_comparison['algorithm_agreement'],
            'mean_ari': float(algo_comparison['mean_ari']),
            'std_ari': float(algo_comparison['std_ari']),
        }
        with open(output_path / f'{prefix}_algorithm_comparison.json', 'w') as f:
            json.dump(algo_summary, f, indent=2)
        saved_files['algorithm_comparison'] = str(output_path / f'{prefix}_algorithm_comparison.json')
        
        # 4. Bootstrap stability
        bootstrap_results = self.bootstrap_stability_analysis(int(optimal_k), feature_set, n_bootstrap=100)
        with open(output_path / f'{prefix}_bootstrap_stability.json', 'w') as f:
            json.dump(bootstrap_results, f, indent=2)
        saved_files['bootstrap_stability'] = str(output_path / f'{prefix}_bootstrap_stability.json')
        
        # 5. Distribution test
        dist_test = self.test_role_distribution(feature_set)
        with open(output_path / f'{prefix}_distribution_test.json', 'w') as f:
            json.dump(dist_test, f, indent=2)
        saved_files['distribution_test'] = str(output_path / f'{prefix}_distribution_test.json')
        
        # 6. t-SNE embedding
        self.compute_tsne_embedding(feature_set)
        labels = self._network_labels if feature_set == 'network' else self._full_labels
        tsne_df = pd.DataFrame({
            'agent_id': self.features['agent_id'],
            'tsne_x': self._tsne_embedding[:, 0],
            'tsne_y': self._tsne_embedding[:, 1],
            'cluster': labels,
        })
        tsne_df.to_csv(output_path / f'{prefix}_tsne_embedding.csv', index=False)
        saved_files['tsne_embedding'] = str(output_path / f'{prefix}_tsne_embedding.csv')
        
        # 7. Cluster assignments
        cluster_df = self.features.copy()
        cluster_df['cluster'] = labels
        cluster_df.to_csv(output_path / f'{prefix}_cluster_assignments.csv', index=False)
        saved_files['cluster_assignments'] = str(output_path / f'{prefix}_cluster_assignments.csv')
        
        # 8. Cluster profiles
        profiles = self.get_cluster_profiles(feature_set)
        profiles.to_csv(output_path / f'{prefix}_cluster_profiles.csv')
        saved_files['cluster_profiles'] = str(output_path / f'{prefix}_cluster_profiles.csv')
        
        # 9. Per-cluster silhouettes
        cluster_sils = pd.DataFrame([
            {'cluster': k, 'silhouette_score': v}
            for k, v in self.compute_silhouette_scores(feature_set).items()
        ])
        cluster_sils.to_csv(output_path / f'{prefix}_cluster_silhouettes.csv', index=False)
        saved_files['cluster_silhouettes'] = str(output_path / f'{prefix}_cluster_silhouettes.csv')
        
        # 10. Centroids
        kmeans = self._network_kmeans if feature_set == 'network' else self._full_kmeans
        feature_cols = self._network_feature_cols if feature_set == 'network' else (
            [f'PC{i+1}' for i in range(self._pca_n_components)] if self._pca else self._full_feature_cols
        )
        if kmeans is not None:
            centroids_df = pd.DataFrame(
                kmeans.cluster_centers_,
                columns=feature_cols[:kmeans.cluster_centers_.shape[1]]
            )
            centroids_df['cluster'] = range(len(centroids_df))
            centroids_df.to_csv(output_path / f'{prefix}_cluster_centroids.csv', index=False)
            saved_files['cluster_centroids'] = str(output_path / f'{prefix}_cluster_centroids.csv')
        
        # Build results summary
        results = {
            'feature_set': feature_set,
            'n_features': len(self._network_feature_cols) if feature_set == 'network' else len(self._full_feature_cols),
            'features_used': self._network_feature_cols if feature_set == 'network' else (
                [f'PC{i+1}' for i in range(self._pca_n_components)] if self._pca else self._full_feature_cols
            ),
            'pca_applied': feature_set == 'full' and self._pca is not None,
            'pca_variance_explained': float(np.sum(self._pca.explained_variance_ratio_)) if (feature_set == 'full' and self._pca) else None,
            'optimal_k': int(optimal_k),
            'silhouette': float(validation_scores[optimal_k]['silhouette']),
            'calinski_harabasz': float(validation_scores[optimal_k]['calinski_harabasz']),
            'davies_bouldin': float(validation_scores[optimal_k]['davies_bouldin']),
            'algorithm_agreement_ari': float(algo_comparison['mean_ari']),
            'algorithm_agreement_std': float(algo_comparison['std_ari']),
            'bootstrap_silhouette_mean': bootstrap_results['silhouette']['mean'],
            'bootstrap_silhouette_ci': [bootstrap_results['silhouette']['ci_lower'], bootstrap_results['silhouette']['ci_upper']],
            'bootstrap_ari_mean': bootstrap_results['ari_stability']['mean'],
            'distribution_chi2': dist_test['chi2_statistic'],
            'distribution_p_value': dist_test['p_value'],
            'distribution_cramers_v': dist_test['cramers_v'],
            'cluster_sizes': dist_test['cluster_counts'],
            'cluster_proportions': dist_test['cluster_proportions'],
        }
        
        return results, saved_files

    
    def save_all_data(self, output_dir: str) -> dict:
        """Save all RQ1 analysis data for BOTH network and full feature analyses.
        
        This runs two complementary analyses:
        1. Network-based clustering (structural roles) - typically strong separation
        2. Full-feature clustering (behavioral roles) - broader patterns
        
        Args:
            output_dir: Directory to save data files.
            
        Returns:
            Dict with paths to saved files and summary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_saved_files = {}
        
        # Run NETWORK analysis (structural roles)
        network_results, network_files = self._run_single_analysis('network', output_path)
        all_saved_files['network'] = network_files
        
        # Run FULL feature analysis (behavioral roles)
        full_results, full_files = self._run_single_analysis('full', output_path)
        all_saved_files['full'] = full_files
        
        # Role classifications (based on feature thresholds, independent of clustering)
        roles = self.classify_roles()
        role_df = pd.DataFrame({
            'agent_id': roles.index,
            'role': roles.values,
        })
        role_df.to_csv(output_path / 'rq1_role_classifications.csv', index=False)
        all_saved_files['role_classifications'] = str(output_path / 'rq1_role_classifications.csv')
        
        # Temporal specialization
        temporal_spec = self.compute_specialization_over_time()
        temporal_spec.to_csv(output_path / 'rq1_specialization_temporal.csv', index=False)
        all_saved_files['specialization_temporal'] = str(output_path / 'rq1_specialization_temporal.csv')
        
        # Temporal model
        model_results = self.fit_mixed_effects_model()
        with open(output_path / 'rq1_temporal_model.json', 'w') as f:
            json.dump(model_results, f, indent=2)
        all_saved_files['temporal_model'] = str(output_path / 'rq1_temporal_model.json')
        
        # Comprehensive summary comparing both analyses
        role_dist = roles.value_counts().to_dict()
        summary = {
            'total_agents': len(self.features),
            'role_distribution': role_dist,
            
            # Network analysis results (PRIMARY - strong cluster separation)
            'network_analysis': {
                'n_features': network_results['n_features'],
                'features': network_results['features_used'],
                'optimal_k': network_results['optimal_k'],
                'silhouette': network_results['silhouette'],
                'calinski_harabasz': network_results['calinski_harabasz'],
                'davies_bouldin': network_results['davies_bouldin'],
                'algorithm_agreement_ari': network_results['algorithm_agreement_ari'],
                'bootstrap_silhouette_mean': network_results['bootstrap_silhouette_mean'],
                'bootstrap_silhouette_ci': network_results['bootstrap_silhouette_ci'],
                'cluster_sizes': network_results['cluster_sizes'],
                'cluster_proportions': network_results['cluster_proportions'],
                'interpretation': 'Structural roles based on network position',
            },
            
            # Full feature analysis results (SECONDARY - broader behavioral patterns)
            'full_analysis': {
                'n_features_original': len(self._full_feature_cols),
                'n_features_used': self._pca_n_components if self._pca else len(self._full_feature_cols),
                'pca_applied': self._pca is not None,
                'pca_variance_explained': full_results['pca_variance_explained'],
                'optimal_k': full_results['optimal_k'],
                'silhouette': full_results['silhouette'],
                'calinski_harabasz': full_results['calinski_harabasz'],
                'davies_bouldin': full_results['davies_bouldin'],
                'algorithm_agreement_ari': full_results['algorithm_agreement_ari'],
                'bootstrap_silhouette_mean': full_results['bootstrap_silhouette_mean'],
                'bootstrap_silhouette_ci': full_results['bootstrap_silhouette_ci'],
                'cluster_sizes': full_results['cluster_sizes'],
                'cluster_proportions': full_results['cluster_proportions'],
                'interpretation': 'Behavioral roles across all activity dimensions',
            },
            
            # Comparison
            'comparison': {
                'network_silhouette': network_results['silhouette'],
                'full_silhouette': full_results['silhouette'],
                'silhouette_difference': network_results['silhouette'] - full_results['silhouette'],
                'recommendation': 'network' if network_results['silhouette'] > full_results['silhouette'] else 'full',
                'note': 'Network features typically yield cleaner structural roles; full features capture broader behavioral patterns',
            },
        }
        
        with open(output_path / 'rq1_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        all_saved_files['summary'] = str(output_path / 'rq1_summary.json')
        
        logger.info(f"=== RQ1 DUAL ANALYSIS COMPLETE ===")
        logger.info(f"Network silhouette: {network_results['silhouette']:.4f} (k={network_results['optimal_k']})")
        logger.info(f"Full silhouette: {full_results['silhouette']:.4f} (k={full_results['optimal_k']})")
        logger.info(f"Saved files to {output_path}")
        
        return all_saved_files

    # ------------------------------------------------------------------
    # PCA Variance Expansion (P1.3)
    # ------------------------------------------------------------------

    def compute_scree_data(self) -> dict:
        """Return per-component variance explained for scree plot.

        Fits PCA on the full (standardised) feature matrix using all
        available components and returns the explained variance ratio
        for each component.

        Returns:
            Dict mapping ``component_index`` (int, 0-based) to
            ``variance_explained_ratio`` (float).
        """
        raw_matrix = self._full_scaler.transform(
            self.features[self._full_feature_cols].values
        )
        n_components = min(len(self._full_feature_cols), len(self.features))
        pca_full = PCA(n_components=n_components, random_state=self.config.random_seed)
        pca_full.fit(raw_matrix)

        scree = {
            int(i): float(v)
            for i, v in enumerate(pca_full.explained_variance_ratio_)
        }
        logger.info(
            "Scree data: %d components, total variance %.3f",
            len(scree),
            sum(scree.values()),
        )
        return scree

    def evaluate_component_counts(
        self, counts: list[int] | None = None
    ) -> dict:
        """Compute silhouette at k=3 for each PCA component count.

        For each count in *counts*, fits PCA with that many components,
        runs K-means with k=3, and records the silhouette score.  Also
        computes the Adjusted Rand Index (ARI) between the 10-component
        solution and every higher-component solution.

        Args:
            counts: List of component counts to evaluate.
                Defaults to ``[10, 15, 20]``.

        Returns:
            Dict with ``silhouette_scores`` mapping *n_components* →
            silhouette, and ``ari_vs_10`` mapping *n_components* → ARI.
        """
        if counts is None:
            counts = [10, 15, 20]

        raw_matrix = self._full_scaler.transform(
            self.features[self._full_feature_cols].values
        )
        max_components = min(len(self._full_feature_cols), len(self.features))

        silhouette_scores: dict[int, float] = {}
        labels_by_count: dict[int, np.ndarray] = {}

        for n in counts:
            n_actual = min(n, max_components)
            pca = PCA(n_components=n_actual, random_state=self.config.random_seed)
            X_pca = pca.fit_transform(raw_matrix)

            km = KMeans(
                n_clusters=3,
                n_init=self.config.kmeans_n_init,
                random_state=self.config.random_seed,
            )
            labels = km.fit_predict(X_pca)
            sil = silhouette_score(X_pca, labels)

            silhouette_scores[n] = float(sil)
            labels_by_count[n] = labels
            logger.info(
                "Component count %d: silhouette=%.4f", n, sil
            )

        # ARI vs 10-component solution
        ari_vs_10: dict[int, float] = {}
        ref_count = 10
        if ref_count in labels_by_count:
            ref_labels = labels_by_count[ref_count]
            for n, labels in labels_by_count.items():
                if n != ref_count:
                    ari = float(adjusted_rand_score(ref_labels, labels))
                    ari_vs_10[n] = ari
                    logger.info(
                        "ARI(%d vs %d) = %.4f", n, ref_count, ari
                    )

        return {
            "silhouette_scores": {int(k): v for k, v in silhouette_scores.items()},
            "ari_vs_10": {int(k): v for k, v in ari_vs_10.items()},
        }

    def compute_umap_projection(self) -> pd.DataFrame:
        """UMAP 2D projection of the full feature space.

        Uses ``umap-learn`` (lazy-imported) to project the standardised
        feature matrix to two dimensions.  Cluster labels come from the
        current full-feature clustering (falls back to k=3 K-means if
        clustering has not been run yet).

        Returns:
            DataFrame with columns ``(agent_id, umap_x, umap_y,
            cluster_label)``.
        """
        try:
            import umap  # umap-learn
        except ImportError as exc:
            raise ImportError(
                "umap-learn is required for UMAP projection. "
                "Install it with: pip install umap-learn"
            ) from exc

        raw_matrix = self._full_scaler.transform(
            self.features[self._full_feature_cols].values
        )

        reducer = umap.UMAP(
            n_components=2,
            random_state=self.config.random_seed,
        )
        embedding = reducer.fit_transform(raw_matrix)

        # Determine cluster labels
        if self._full_labels is not None:
            cluster_labels = self._full_labels
        else:
            km = KMeans(
                n_clusters=3,
                n_init=self.config.kmeans_n_init,
                random_state=self.config.random_seed,
            )
            cluster_labels = km.fit_predict(raw_matrix)

        result = pd.DataFrame({
            "agent_id": self.features["agent_id"].values,
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
            "cluster_label": cluster_labels,
        })

        logger.info(
            "UMAP projection: %d agents → 2D", len(result)
        )
        return result


class TemporalRoleAnalyzer:
    """Survival analysis for hub emergence using daily network snapshots.

    Builds daily cluster assignments, identifies hub emergence events,
    computes Cox model covariates, and writes results to CSV.
    """

    # Clusters considered "hub" roles (Active Contributors, Specialized Connectors)
    HUB_CLUSTERS = {1, 4}

    def __init__(
        self,
        storage: "JSONStorage",
        network_builder: "NetworkBuilder",
        feature_extractor: "FeatureExtractor",
        config: Config,
    ) -> None:
        """Initialize temporal role analyzer.

        Args:
            storage: JSONStorage instance for querying data.
            network_builder: NetworkBuilder for constructing temporal snapshots.
            feature_extractor: FeatureExtractor for computing agent features.
            config: Configuration parameters.
        """
        self.storage = storage
        self.network_builder = network_builder
        self.feature_extractor = feature_extractor
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_daily_cluster_assignments(self) -> pd.DataFrame:
        """For each day, build network snapshot, run K-means, assign clusters.

        Uses ``NetworkBuilder.get_temporal_snapshots()`` with a 1-day interval
        to obtain daily graphs.  For each snapshot the 5 network-centrality
        features (in_degree, out_degree, betweenness, clustering_coefficient,
        pagerank) are computed for every node, standardised, and clustered
        with K-means (k taken from ``config.kmeans_k_range`` lower bound,
        default 3).

        Returns:
            DataFrame with columns ``(agent_id, date, cluster_id)``.
        """
        snapshots = self.network_builder.get_temporal_snapshots(
            interval=timedelta(days=1),
        )

        if not snapshots:
            logger.warning("No temporal snapshots available")
            return pd.DataFrame(columns=["agent_id", "date", "cluster_id"])

        k = self.config.kmeans_k_range[0]
        rows: list[dict] = []

        for snap_time, graph in snapshots:
            snap_date = snap_time.date() if isinstance(snap_time, datetime) else snap_time
            nodes = list(graph.nodes())
            if len(nodes) < k:
                # Not enough nodes to cluster – assign all to cluster 0
                for node in nodes:
                    rows.append({"agent_id": node, "date": snap_date, "cluster_id": 0})
                continue

            feature_matrix = self._extract_network_features(graph, nodes)

            # Standardise
            scaler = StandardScaler()
            X = scaler.fit_transform(feature_matrix)

            kmeans = KMeans(
                n_clusters=k,
                n_init=min(self.config.kmeans_n_init, 10),
                random_state=self.config.random_seed,
            )
            labels = kmeans.fit_predict(X)

            for node, label in zip(nodes, labels):
                rows.append({"agent_id": node, "date": snap_date, "cluster_id": int(label)})

        df = pd.DataFrame(rows)
        logger.info(
            "Built daily cluster assignments: %d rows across %d days",
            len(df),
            df["date"].nunique() if len(df) else 0,
        )
        return df

    def identify_hub_emergence_events(
        self,
        daily_assignments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Find first day each agent enters Cluster 1 or Cluster 4.

        Args:
            daily_assignments: Output of ``build_daily_cluster_assignments()``.
                If *None*, it will be computed automatically.

        Returns:
            DataFrame with columns ``(agent_id, emergence_date, emerged,
            time_to_emergence_days)``.  Right-censored agents who never
            emerge get ``emerged=False`` and ``time_to_emergence_days``
            equal to their total observation time.
        """
        if daily_assignments is None:
            daily_assignments = self.build_daily_cluster_assignments()

        if daily_assignments.empty:
            return pd.DataFrame(
                columns=["agent_id", "emergence_date", "emerged", "time_to_emergence_days"]
            )

        # Ensure date column is comparable
        daily_assignments = daily_assignments.copy()
        daily_assignments["date"] = pd.to_datetime(daily_assignments["date"])

        # Per-agent first and last observation dates
        agent_obs = daily_assignments.groupby("agent_id")["date"].agg(["min", "max"])
        agent_obs.columns = ["first_date", "last_date"]

        # Filter to hub-cluster rows and find first occurrence per agent
        hub_rows = daily_assignments[daily_assignments["cluster_id"].isin(self.HUB_CLUSTERS)]
        if not hub_rows.empty:
            first_hub = hub_rows.groupby("agent_id")["date"].min().reset_index()
            first_hub.columns = ["agent_id", "emergence_date"]
        else:
            first_hub = pd.DataFrame(columns=["agent_id", "emergence_date"])

        # Build result for every agent
        records: list[dict] = []
        for agent_id, obs in agent_obs.iterrows():
            match = first_hub[first_hub["agent_id"] == agent_id]
            if not match.empty:
                emergence_date = match.iloc[0]["emergence_date"]
                records.append(
                    {
                        "agent_id": agent_id,
                        "emergence_date": emergence_date,
                        "emerged": True,
                        "time_to_emergence_days": (emergence_date - obs["first_date"]).days,
                    }
                )
            else:
                records.append(
                    {
                        "agent_id": agent_id,
                        "emergence_date": None,
                        "emerged": False,
                        "time_to_emergence_days": (obs["last_date"] - obs["first_date"]).days,
                    }
                )

        result = pd.DataFrame(records)
        n_emerged = result["emerged"].sum() if len(result) else 0
        logger.info(
            "Hub emergence events: %d emerged, %d right-censored",
            n_emerged,
            len(result) - n_emerged,
        )
        return result

    def compute_covariates(
        self,
        hub_events: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute Cox model covariates for each agent.

        Covariates:
        - ``join_cohort``: categorical – 'day1-3', 'day4-7', 'later'
        - ``initial_posting_cadence``: posts/day in first 3 days
        - ``submolt_diversity_day3``: distinct submolts by day 3
        - ``early_reply``: received reply within 24 h of first post

        Args:
            hub_events: Output of ``identify_hub_emergence_events()``.
                If *None*, it will be computed automatically.

        Returns:
            DataFrame with hub-event columns plus the four covariates.
        """
        if hub_events is None:
            hub_events = self.identify_hub_emergence_events()

        if hub_events.empty:
            return hub_events

        posts = self.storage.get_posts()
        comments = self.storage.get_comments()

        # Build lookup structures
        posts_by_author: dict[str, list] = {}
        for p in posts:
            posts_by_author.setdefault(p.author_id, []).append(p)

        # Determine the global observation start (earliest post)
        all_post_times = [p.created_at for p in posts if p.created_at]
        if not all_post_times:
            # No posts – return hub_events with empty covariates
            hub_events = hub_events.copy()
            hub_events["join_cohort"] = "later"
            hub_events["initial_posting_cadence"] = 0.0
            hub_events["submolt_diversity_day3"] = 0
            hub_events["early_reply"] = False
            return hub_events

        obs_start = min(all_post_times)

        # Build a mapping: post_id -> list of comment timestamps (for early_reply)
        comments_by_post: dict[str, list[datetime]] = {}
        for c in comments:
            if c.created_at:
                comments_by_post.setdefault(c.post_id, []).append(c.created_at)

        covariates: list[dict] = []
        for _, row in hub_events.iterrows():
            agent_id = row["agent_id"]
            agent_posts = posts_by_author.get(agent_id, [])
            agent_posts_with_time = [p for p in agent_posts if p.created_at]
            agent_posts_with_time.sort(key=lambda p: p.created_at)

            # --- join_cohort ---
            if agent_posts_with_time:
                first_post_time = agent_posts_with_time[0].created_at
                days_since_start = (first_post_time - obs_start).days
            else:
                days_since_start = 999  # no posts → 'later'

            if days_since_start <= 2:  # day 1-3 (0-indexed days 0,1,2)
                join_cohort = "day1-3"
            elif days_since_start <= 6:  # day 4-7
                join_cohort = "day4-7"
            else:
                join_cohort = "later"

            # --- initial_posting_cadence (posts/day in first 3 days) ---
            if agent_posts_with_time:
                cutoff = first_post_time + timedelta(days=3)
                early_posts = [p for p in agent_posts_with_time if p.created_at < cutoff]
                initial_posting_cadence = len(early_posts) / 3.0
            else:
                initial_posting_cadence = 0.0

            # --- submolt_diversity_day3 ---
            if agent_posts_with_time:
                cutoff = first_post_time + timedelta(days=3)
                early_submolts = {
                    p.submolt for p in agent_posts_with_time
                    if p.created_at < cutoff and p.submolt
                }
                submolt_diversity_day3 = len(early_submolts)
            else:
                submolt_diversity_day3 = 0

            # --- early_reply (received reply within 24 h of first post) ---
            early_reply = False
            if agent_posts_with_time:
                first_post = agent_posts_with_time[0]
                reply_deadline = first_post.created_at + timedelta(hours=24)
                post_comments = comments_by_post.get(first_post.post_id, [])
                if any(ct <= reply_deadline for ct in post_comments):
                    early_reply = True

            covariates.append(
                {
                    "agent_id": agent_id,
                    "join_cohort": join_cohort,
                    "initial_posting_cadence": initial_posting_cadence,
                    "submolt_diversity_day3": submolt_diversity_day3,
                    "early_reply": early_reply,
                }
            )

        cov_df = pd.DataFrame(covariates)
        result = hub_events.merge(cov_df, on="agent_id", how="left")
        logger.info("Computed covariates for %d agents", len(result))
        return result

    def fit_cox_model(self) -> dict:
        """Fit Cox PH model using lifelines.CoxPHFitter.

        Calls :meth:`compute_covariates` to obtain the survival data,
        encodes ``join_cohort`` as dummy variables (``day1-3`` as reference),
        fits a Cox proportional-hazards model, and runs the Schoenfeld test
        for the PH assumption.

        Returns:
            Dict with ``hazard_ratios``, ``confidence_intervals``,
            ``concordance_index``, ``log_likelihood``, and
            ``schoenfeld_test`` results.  Also writes
            ``output/rq1_cox_hub_model.json``.
        """
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test

        surv_df = self.compute_covariates()

        if surv_df.empty:
            logger.warning("No survival data available for Cox model")
            return {}

        # Prepare modelling DataFrame
        model_df = surv_df[
            [
                "time_to_emergence_days",
                "emerged",
                "join_cohort",
                "initial_posting_cadence",
                "submolt_diversity_day3",
                "early_reply",
            ]
        ].copy()

        # Encode join_cohort as dummies with day1-3 as reference
        cohort_dummies = pd.get_dummies(
            model_df["join_cohort"], prefix="cohort", dtype=float
        )
        ref_col = "cohort_day1-3"
        dummy_cols = [c for c in cohort_dummies.columns if c != ref_col]
        model_df = pd.concat([model_df, cohort_dummies[dummy_cols]], axis=1)
        model_df.drop(columns=["join_cohort"], inplace=True)

        # Ensure boolean → float
        model_df["early_reply"] = model_df["early_reply"].astype(float)

        # Ensure duration > 0 for lifelines (replace 0 with 0.5 day)
        model_df["time_to_emergence_days"] = model_df[
            "time_to_emergence_days"
        ].clip(lower=0.5)

        # Fit Cox PH model (small penalizer for numerical stability)
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(
            model_df,
            duration_col="time_to_emergence_days",
            event_col="emerged",
        )

        # Extract results
        summary = cph.summary
        hazard_ratios = {
            covar: float(summary.loc[covar, "exp(coef)"])
            for covar in summary.index
        }
        confidence_intervals = {
            covar: {
                "lower": float(
                    summary.loc[covar, "exp(coef) lower 95%"]
                ),
                "upper": float(
                    summary.loc[covar, "exp(coef) upper 95%"]
                ),
            }
            for covar in summary.index
        }
        concordance_index = float(cph.concordance_index_)
        log_likelihood = float(cph.log_likelihood_)

        # Schoenfeld test for PH assumption
        try:
            schoenfeld_result = proportional_hazard_test(
                cph, model_df, time_transform="rank"
            )
            schoenfeld_test = {
                covar: {
                    "test_statistic": float(
                        schoenfeld_result.summary.loc[covar, "test_statistic"]
                    ),
                    "p_value": float(schoenfeld_result.summary.loc[covar, "p"]),
                }
                for covar in schoenfeld_result.summary.index
            }
        except Exception as exc:
            logger.warning("Schoenfeld test failed: %s", exc)
            schoenfeld_test = {"error": str(exc)}

        result = {
            "hazard_ratios": hazard_ratios,
            "confidence_intervals": confidence_intervals,
            "concordance_index": concordance_index,
            "log_likelihood": log_likelihood,
            "schoenfeld_test": schoenfeld_test,
        }

        # Write output
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "rq1_cox_hub_model.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote Cox model results to %s", out_path / "rq1_cox_hub_model.json")

        return result

    def plot_kaplan_meier(self) -> dict:
        """Compute KM survival curves stratified by join_cohort.

        Uses ``lifelines.KaplanMeierFitter`` to compute a survival function
        for each distinct ``join_cohort`` value.

        Returns:
            Dict with survival function data points per cohort, suitable
            for TikZ rendering.  Also writes ``output/rq1_km_curves.json``.
        """
        from lifelines import KaplanMeierFitter

        surv_df = self.compute_covariates()

        if surv_df.empty:
            logger.warning("No survival data available for KM curves")
            return {}

        cohorts = sorted(surv_df["join_cohort"].unique())
        curves: dict[str, dict] = {}

        for cohort in cohorts:
            mask = surv_df["join_cohort"] == cohort
            subset = surv_df[mask]

            durations = subset["time_to_emergence_days"].clip(lower=0.5)
            events = subset["emerged"].astype(int)

            kmf = KaplanMeierFitter()
            kmf.fit(durations, event_observed=events, label=cohort)

            sf = kmf.survival_function_
            curves[cohort] = {
                "timeline": [float(t) for t in sf.index.tolist()],
                "survival_probability": [
                    float(v) for v in sf.iloc[:, 0].tolist()
                ],
                "n_subjects": int(len(subset)),
                "n_events": int(events.sum()),
            }

        result = {"cohorts": curves, "n_cohorts": len(cohorts)}

        # Write output
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "rq1_km_curves.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote KM curves to %s", out_path / "rq1_km_curves.json")

        return result

    def run_and_save(self, output_dir: Optional[str] = None) -> pd.DataFrame:
        """Run the full temporal role analysis and write results to CSV.

        Args:
            output_dir: Directory to write ``rq1_hub_emergence.csv``.
                Defaults to ``config.output_dir``.

        Returns:
            The final DataFrame written to disk.
        """
        daily = self.build_daily_cluster_assignments()
        hub_events = self.identify_hub_emergence_events(daily)
        result = self.compute_covariates(hub_events)

        out_path = Path(output_dir or self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "rq1_hub_emergence.csv"
        result.to_csv(csv_path, index=False)
        logger.info("Wrote hub emergence data to %s", csv_path)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_network_features(
        graph: "nx.DiGraph | nx.Graph",
        nodes: list[str],
    ) -> np.ndarray:
        """Compute the 5 network-centrality features for *nodes* in *graph*.

        Returns an (N, 5) array with columns:
        ``[in_degree, out_degree, betweenness, clustering_coefficient, pagerank]``.
        """
        import networkx as nx

        is_directed = isinstance(graph, nx.DiGraph)

        # Degree
        if is_directed:
            in_deg = dict(graph.in_degree(nodes))
            out_deg = dict(graph.out_degree(nodes))
        else:
            deg = dict(graph.degree(nodes))
            in_deg = deg
            out_deg = deg

        # Betweenness (computed on full graph, then sliced)
        betweenness = nx.betweenness_centrality(graph)

        # Clustering coefficient (needs undirected)
        undirected = graph.to_undirected() if is_directed else graph
        clustering = nx.clustering(undirected)

        # PageRank
        try:
            pagerank = nx.pagerank(graph, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            pagerank = {n: 1.0 / len(graph) for n in graph.nodes()}

        rows = []
        for n in nodes:
            rows.append(
                [
                    in_deg.get(n, 0),
                    out_deg.get(n, 0),
                    betweenness.get(n, 0.0),
                    clustering.get(n, 0.0),
                    pagerank.get(n, 0.0),
                ]
            )
        return np.array(rows, dtype=float)
