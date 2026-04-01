"""RQ2: Information diffusion analysis module.

Implements cascade identification, diffusion modeling, and contagion type
classification for memes, skills, and behavioral patterns.
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .storage import JSONStorage
from .config import Config
from .models import Cascade

logger = logging.getLogger(__name__)


class CascadeIdentifier:
    """Identifies information cascades in MoltBook data."""
    
    def __init__(
        self,
        storage: JSONStorage,
        config: Config,
    ) -> None:
        """Initialize cascade identifier.
        
        Args:
            storage: JSONStorage instance.
            config: Configuration parameters.
        """
        self.storage = storage
        self.config = config
    
    def identify_meme_cascades(
        self,
        min_adopters: int = 10,
        ngram_range: tuple = (3, 5),
        min_ngram_length: int = 10,
    ) -> list[Cascade]:
        """Identify meme cascades using n-gram analysis.
        
        Args:
            min_adopters: Minimum number of unique adopters.
            ngram_range: Range of n-gram sizes to analyze.
            min_ngram_length: Minimum character length for n-gram.
            
        Returns:
            List of identified meme cascades.
        """
        posts = self.storage.get_posts()
        comments = self.storage.get_comments()
        
        # Collect all content with timestamps and authors
        content_items = []
        for post in posts:
            if post.body and post.created_at:
                content_items.append({
                    'text': post.body,
                    'author': post.author_id,
                    'timestamp': post.created_at,
                    'type': 'post',
                })
        for comment in comments:
            if comment.body and comment.created_at:
                content_items.append({
                    'text': comment.body,
                    'author': comment.author_id,
                    'timestamp': comment.created_at,
                    'type': 'comment',
                })
        
        if not content_items:
            return []
        
        # Common stopword n-grams to exclude
        stopword_patterns = {
            'the', 'and', 'for', 'that', 'this', 'with', 'you', 'are', 'have',
            'was', 'were', 'been', 'being', 'would', 'could', 'should', 'will',
            'can', 'may', 'might', 'must', 'shall', 'need', 'want', 'like',
        }
        
        # Extract n-grams and track adoptions
        ngram_adoptions = defaultdict(list)  # ngram -> [(author, timestamp), ...]
        
        for item in content_items:
            text = item['text'].lower()
            words = re.findall(r'\b\w+\b', text)
            
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(words) - n + 1):
                    ngram_words = words[i:i+n]
                    ngram = ' '.join(ngram_words)
                    
                    # Skip short n-grams
                    if len(ngram) < min_ngram_length:
                        continue
                    
                    # Skip if mostly stopwords
                    stopword_count = sum(1 for w in ngram_words if w in stopword_patterns)
                    if stopword_count > len(ngram_words) * 0.5:
                        continue
                    
                    ngram_adoptions[ngram].append((item['author'], item['timestamp']))
        
        # Filter to cascades with minimum adopters
        cascades = []
        for ngram, adoptions in ngram_adoptions.items():
            # Get unique adopters
            unique_adopters = set(a[0] for a in adoptions)
            
            if len(unique_adopters) >= min_adopters:
                # Sort by timestamp
                sorted_adoptions = sorted(adoptions, key=lambda x: x[1])
                
                # Create cascade
                cascade = Cascade(
                    cascade_id=hashlib.md5(ngram.encode()).hexdigest()[:16],
                    cascade_type='meme',
                    seed_agent=sorted_adoptions[0][0],
                    seed_time=sorted_adoptions[0][1],
                    adoptions=sorted_adoptions,
                    content_hash=ngram,
                )
                cascades.append(cascade)
        
        # Limit to top cascades by size to avoid memory issues
        cascades.sort(key=lambda c: len(set(a[0] for a in c.adoptions)), reverse=True)
        cascades = cascades[:10000]  # Keep top 10k
        
        logger.info(f"Identified {len(cascades)} meme cascades with >= {min_adopters} adopters")
        return cascades

    
    def identify_skill_cascades(
        self,
        min_adopters: int = 5,
    ) -> list[Cascade]:
        """Identify skill cascades from code/module sharing.
        
        Args:
            min_adopters: Minimum number of unique adopters.
            
        Returns:
            List of identified skill cascades.
        """
        posts = self.storage.get_posts()
        comments = self.storage.get_comments()
        
        # Extract code blocks and track adoptions
        code_adoptions = defaultdict(list)
        
        code_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        
        for post in posts:
            if post.body and post.created_at:
                code_blocks = code_pattern.findall(post.body)
                for code in code_blocks:
                    # Hash the code content
                    code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
                    code_adoptions[code_hash].append((post.author_id, post.created_at))
        
        for comment in comments:
            if comment.body and comment.created_at:
                code_blocks = code_pattern.findall(comment.body)
                for code in code_blocks:
                    code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
                    code_adoptions[code_hash].append((comment.author_id, comment.created_at))
        
        # Filter to cascades with minimum adopters
        cascades = []
        for code_hash, adoptions in code_adoptions.items():
            unique_adopters = set(a[0] for a in adoptions)
            
            if len(unique_adopters) >= min_adopters:
                sorted_adoptions = sorted(adoptions, key=lambda x: x[1])
                
                cascade = Cascade(
                    cascade_id=code_hash,
                    cascade_type='skill',
                    seed_agent=sorted_adoptions[0][0],
                    seed_time=sorted_adoptions[0][1],
                    adoptions=sorted_adoptions,
                    content_hash=code_hash,
                )
                cascades.append(cascade)
        
        logger.info(f"Identified {len(cascades)} skill cascades")
        return cascades
    
    def identify_behavioral_cascades(
        self,
        min_adopters: int = 5,
    ) -> list[Cascade]:
        """Identify behavioral cascades from formatting styles.
        
        Args:
            min_adopters: Minimum number of unique adopters.
            
        Returns:
            List of identified behavioral cascades.
        """
        posts = self.storage.get_posts()
        comments = self.storage.get_comments()
        
        # Track formatting patterns
        pattern_adoptions = defaultdict(list)
        
        # Define behavioral patterns to track
        patterns = {
            'emoji_heavy': re.compile(r'[\U0001F600-\U0001F64F]{3,}'),
            'bullet_list': re.compile(r'^\s*[-*]\s+', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+', re.MULTILINE),
            'header_style': re.compile(r'^#+\s+', re.MULTILINE),
            'quote_style': re.compile(r'^>\s+', re.MULTILINE),
            'link_heavy': re.compile(r'\[.*?\]\(.*?\)'),
        }
        
        for post in posts:
            if post.body and post.created_at:
                for pattern_name, pattern in patterns.items():
                    if pattern.search(post.body):
                        pattern_adoptions[pattern_name].append(
                            (post.author_id, post.created_at)
                        )
        
        for comment in comments:
            if comment.body and comment.created_at:
                for pattern_name, pattern in patterns.items():
                    if pattern.search(comment.body):
                        pattern_adoptions[pattern_name].append(
                            (comment.author_id, comment.created_at)
                        )
        
        # Filter to cascades with minimum adopters
        cascades = []
        for pattern_name, adoptions in pattern_adoptions.items():
            unique_adopters = set(a[0] for a in adoptions)
            
            if len(unique_adopters) >= min_adopters:
                sorted_adoptions = sorted(adoptions, key=lambda x: x[1])
                
                cascade = Cascade(
                    cascade_id=f"behavioral_{pattern_name}",
                    cascade_type='behavioral',
                    seed_agent=sorted_adoptions[0][0],
                    seed_time=sorted_adoptions[0][1],
                    adoptions=sorted_adoptions,
                    content_hash=pattern_name,
                )
                cascades.append(cascade)
        
        logger.info(f"Identified {len(cascades)} behavioral cascades")
        return cascades


class DiffusionModeler:
    """Models information diffusion dynamics."""
    
    def __init__(
        self,
        cascades: list[Cascade],
        network: nx.DiGraph,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize diffusion modeler.
        
        Args:
            cascades: List of identified cascades.
            network: Interaction network.
            config: Configuration object.
        """
        self.cascades = cascades
        self.network = network
        self.config = config or Config()
    
    def compute_exposures(self, cascade: Cascade) -> pd.DataFrame:
        """Compute exposure counts for each agent in a cascade.
        
        Args:
            cascade: The cascade to analyze.
            
        Returns:
            DataFrame with agent, exposure_count, adopted, adoption_time.
        """
        # Track who has adopted and when
        adopted_agents = {}
        for agent, timestamp in cascade.adoptions:
            if agent not in adopted_agents:
                adopted_agents[agent] = timestamp
        
        # Compute exposures for each agent
        exposure_data = []
        
        for agent in self.network.nodes():
            # Count neighbors who adopted before this agent
            exposure_count = 0
            
            for neighbor in self.network.predecessors(agent):
                if neighbor in adopted_agents:
                    if agent in adopted_agents:
                        # Only count if neighbor adopted before this agent
                        if adopted_agents[neighbor] < adopted_agents[agent]:
                            exposure_count += 1
                    else:
                        exposure_count += 1
            
            exposure_data.append({
                'agent': agent,
                'exposure_count': exposure_count,
                'adopted': agent in adopted_agents,
                'adoption_time': adopted_agents.get(agent),
            })
        
        return pd.DataFrame(exposure_data)

    
    def fit_logistic_model(self) -> dict:
        """Fit logistic regression model for adoption probability with statistical tests.
        
        Returns:
            Dict with model coefficients, confidence intervals, and statistics.
        """
        from sklearn.linear_model import LogisticRegression
        
        # Aggregate exposure data across all cascades in parallel
        def process_cascade(cascade):
            exposure_df = self.compute_exposures(cascade)
            exposure_df['cascade_id'] = cascade.cascade_id
            return exposure_df
        
        logger.info(f"Computing exposures for {len(self.cascades)} cascades...")
        all_exposures = Parallel(n_jobs=self.config.n_jobs, verbose=1)(
            delayed(process_cascade)(cascade) for cascade in self.cascades
        )
        
        if not all_exposures:
            return {'result': 'no_data'}
        
        # Filter out empty DataFrames before concatenation
        all_exposures = [df for df in all_exposures if not df.empty]
        if not all_exposures:
            return {'result': 'no_data'}
        
        logger.info(f"Concatenating {len(all_exposures)} exposure DataFrames...")
        df = pd.concat(all_exposures, ignore_index=True)
        logger.info(f"Total exposure records: {len(df):,}")
        
        # Prepare features: linear and quadratic exposure
        X = df[['exposure_count']].copy()
        X['exposure_squared'] = X['exposure_count'] ** 2
        y = df['adopted'].astype(int)
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            logger.warning("Insufficient class diversity for logistic regression")
            return {'result': 'insufficient_class_diversity'}
        
        # Fit model using statsmodels for proper inference
        try:
            import statsmodels.api as sm
            
            # Use numpy arrays directly to avoid DataFrame copy overhead
            X_arr = X.values
            X_with_const = np.column_stack([np.ones(len(X_arr)), X_arr])
            y_arr = y.values
            
            logger.info("Fitting logistic model...")
            model = sm.Logit(y_arr, X_with_const)
            result = model.fit(disp=0)
            
            # Extract coefficients with confidence intervals
            params = result.params
            conf_int = result.conf_int()
            pvalues = result.pvalues
            
            # Odds ratios
            odds_ratios = np.exp(params)
            or_ci_lower = np.exp(conf_int[:, 0])
            or_ci_upper = np.exp(conf_int[:, 1])
            
            # Model fit statistics
            pseudo_r2 = result.prsquared
            llr_pvalue = result.llr_pvalue
            aic = result.aic
            bic = result.bic
            
            results = {
                'intercept': float(params[0]),
                'intercept_p': float(pvalues[0]),
                'beta_linear': float(params[1]),
                'beta_linear_se': float(result.bse[1]),
                'beta_linear_p': float(pvalues[1]),
                'beta_linear_ci': [float(conf_int[1, 0]), float(conf_int[1, 1])],
                'beta_quadratic': float(params[2]),
                'beta_quadratic_se': float(result.bse[2]),
                'beta_quadratic_p': float(pvalues[2]),
                'beta_quadratic_ci': [float(conf_int[2, 0]), float(conf_int[2, 1])],
                'odds_ratio_linear': float(odds_ratios[1]),
                'odds_ratio_linear_ci': [float(or_ci_lower[1]), float(or_ci_upper[1])],
                'odds_ratio_quadratic': float(odds_ratios[2]),
                'odds_ratio_quadratic_ci': [float(or_ci_lower[2]), float(or_ci_upper[2])],
                'pseudo_r2': float(pseudo_r2),
                'llr_p_value': float(llr_pvalue),
                'aic': float(aic),
                'bic': float(bic),
                'n_observations': len(df),
                'n_adopted': int(y.sum()),
                'adoption_rate': float(y.mean()),
            }
            
            logger.info(f"Logistic model: beta_linear={results['beta_linear']:.4f} (p={results['beta_linear_p']:.4e}), "
                       f"beta_quad={results['beta_quadratic']:.4f} (p={results['beta_quadratic_p']:.4e})")
            
            return results
            
        except ImportError:
            # Fallback to sklearn (no p-values)
            logger.warning("statsmodels not available, using sklearn (no p-values)")
            
            model = LogisticRegression(random_state=42)
            model.fit(X, y)
            
            return {
                'intercept': float(model.intercept_[0]),
                'beta_linear': float(model.coef_[0][0]),
                'beta_quadratic': float(model.coef_[0][1]),
                'n_observations': len(df),
                'warning': 'Install statsmodels for p-values and confidence intervals',
            }
    
    def fit_cox_hazards(self, max_cascades: int = 100) -> dict:
        """Fit Cox proportional hazards model for adoption timing.
        
        Args:
            max_cascades: Maximum number of cascades to use (for performance).
        
        Returns:
            Dict with model coefficients, confidence intervals, and statistics.
        """
        try:
            from lifelines import CoxPHFitter
            
            # Prepare survival data (sample cascades for performance)
            survival_data = []
            sample_cascades = self.cascades[:max_cascades] if len(self.cascades) > max_cascades else self.cascades
            
            logger.info(f"Fitting Cox model on {len(sample_cascades)} cascades...")
            
            for i, cascade in enumerate(sample_cascades):
                if i % 20 == 0:
                    logger.info(f"Processing cascade {i+1}/{len(sample_cascades)} for Cox model...")
                
                exposure_df = self.compute_exposures(cascade)
                
                for _, row in exposure_df.iterrows():
                    if row['adopted'] and row['adoption_time']:
                        # Time to adoption from cascade start
                        duration = (row['adoption_time'] - cascade.seed_time).total_seconds() / 3600
                        if duration > 0:
                            survival_data.append({
                                'duration': duration,
                                'event': 1,
                                'exposure': row['exposure_count'],
                            })
                    elif not row['adopted'] and row['exposure_count'] > 0:
                        # Censored observation (exposed but didn't adopt)
                        timestamps = [a[1] for a in cascade.adoptions]
                        if timestamps:
                            max_time = (max(timestamps) - cascade.seed_time).total_seconds() / 3600
                            if max_time > 0:
                                survival_data.append({
                                    'duration': max_time,
                                    'event': 0,  # Censored
                                    'exposure': row['exposure_count'],
                                })
            
            if len(survival_data) < 20:
                logger.warning(f"Insufficient survival data: {len(survival_data)} observations")
                return {'result': 'insufficient_data', 'n_observations': len(survival_data)}
            
            df = pd.DataFrame(survival_data)
            
            # Check for variance in exposure
            if df['exposure'].var() < 1e-10:
                return {'result': 'no_variance_in_exposure', 'n_observations': len(df)}
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(df, duration_col='duration', event_col='event')
            
            # Extract results with confidence intervals
            coef = cph.params_['exposure']
            se = cph.standard_errors_['exposure']
            hr = np.exp(coef)
            hr_lower = np.exp(coef - 1.96 * se)
            hr_upper = np.exp(coef + 1.96 * se)
            
            # Model fit statistics
            c_index = cph.concordance_index_
            log_likelihood = cph.log_likelihood_
            
            # Proportional hazards test
            try:
                ph_test = cph.check_assumptions(df, show_plots=False, p_value_threshold=0.05)
                ph_violated = False
            except:
                ph_violated = None
            
            results = {
                'coefficient': float(coef),
                'standard_error': float(se),
                'hazard_ratio': float(hr),
                'hazard_ratio_ci_lower': float(hr_lower),
                'hazard_ratio_ci_upper': float(hr_upper),
                'p_value': float(cph.summary['p']['exposure']),
                'z_statistic': float(cph.summary['z']['exposure']),
                'concordance_index': float(c_index),
                'log_likelihood': float(log_likelihood),
                'n_observations': len(df),
                'n_events': int(df['event'].sum()),
                'proportional_hazards_violated': ph_violated,
            }
            
            logger.info(f"Cox model: HR={hr:.3f} [{hr_lower:.3f}, {hr_upper:.3f}], "
                       f"p={results['p_value']:.4e}, C-index={c_index:.3f}")
            
            return results
            
        except ImportError:
            logger.warning("lifelines not available for Cox model - install with: pip install lifelines")
            return {'result': 'lifelines_not_installed', 'install_command': 'pip install lifelines'}
        except Exception as e:
            logger.error(f"Cox model fitting failed: {e}")
            return {'result': 'fitting_failed', 'error': str(e)}
    
    def test_contagion_type(self) -> dict:
        """Classify contagion as simple or complex with statistical evidence.
        
        Simple contagion: Linear exposure effect (β₂ ≈ 0, not significant)
        Complex contagion: Quadratic exposure effect (β₂ > 0, significant)
        
        Returns:
            Dict with classification and supporting evidence.
        """
        model_results = self.fit_logistic_model()
        
        if not model_results or 'result' in model_results:
            return {'classification': 'unknown', 'reason': 'model_fitting_failed'}
        
        beta_quadratic = model_results.get('beta_quadratic', 0)
        beta_quad_p = model_results.get('beta_quadratic_p', 1.0)
        
        # Classification based on statistical significance
        alpha = 0.05  # Significance level
        
        if beta_quad_p < alpha and beta_quadratic > 0:
            classification = 'complex'
            evidence = 'Significant positive quadratic term indicates reinforcement effects'
        elif beta_quad_p < alpha and beta_quadratic < 0:
            classification = 'saturating'
            evidence = 'Significant negative quadratic term indicates diminishing returns'
        else:
            classification = 'simple'
            evidence = 'Non-significant quadratic term suggests linear exposure effect'
        
        return {
            'classification': classification,
            'evidence': evidence,
            'beta_quadratic': beta_quadratic,
            'beta_quadratic_p': beta_quad_p,
            'significance_level': alpha,
            'is_significant': beta_quad_p < alpha,
        }


    def fit_logistic_model_by_type(self) -> dict[str, dict]:
        """Run logistic regression adoption model separately per cascade type.

        Groups cascades by ``cascade_type``, then for each type computes
        exposures and fits a logistic regression with linear + quadratic
        exposure terms (mirroring :meth:`fit_logistic_model`).

        Returns:
            Dict keyed by cascade_type with ``beta_linear``, ``beta_quadratic``,
            ``p_values``, ``pseudo_r2``, and ``contagion_type`` for each.
            Types with too few cascades or insufficient class diversity report
            null values gracefully.
        """
        from collections import defaultdict as _defaultdict

        # Group cascades by type
        type_cascades: dict[str, list[Cascade]] = _defaultdict(list)
        for cascade in self.cascades:
            type_cascades[cascade.cascade_type].append(cascade)

        results: dict[str, dict] = {}

        for cascade_type, cascades in type_cascades.items():
            if len(cascades) < 2:
                logger.warning(
                    f"Too few cascades for logistic fit on type '{cascade_type}': "
                    f"{len(cascades)} cascades"
                )
                results[cascade_type] = {
                    'beta_linear': None,
                    'beta_quadratic': None,
                    'p_values': None,
                    'pseudo_r2': None,
                    'contagion_type': None,
                    'n_cascades': len(cascades),
                    'result': 'insufficient_data',
                }
                continue

            # Compute exposures for each cascade in this type subset
            exposure_frames = []
            for cascade in cascades:
                exposure_df = self.compute_exposures(cascade)
                exposure_df['cascade_id'] = cascade.cascade_id
                exposure_frames.append(exposure_df)

            exposure_frames = [df for df in exposure_frames if not df.empty]
            if not exposure_frames:
                results[cascade_type] = {
                    'beta_linear': None,
                    'beta_quadratic': None,
                    'p_values': None,
                    'pseudo_r2': None,
                    'contagion_type': None,
                    'n_cascades': len(cascades),
                    'result': 'no_exposure_data',
                }
                continue

            df = pd.concat(exposure_frames, ignore_index=True)

            X = df[['exposure_count']].copy()
            X['exposure_squared'] = X['exposure_count'] ** 2
            y = df['adopted'].astype(int)

            if len(y.unique()) < 2:
                logger.warning(
                    f"Insufficient class diversity for type '{cascade_type}'"
                )
                results[cascade_type] = {
                    'beta_linear': None,
                    'beta_quadratic': None,
                    'p_values': None,
                    'pseudo_r2': None,
                    'contagion_type': None,
                    'n_cascades': len(cascades),
                    'n_observations': len(df),
                    'result': 'insufficient_class_diversity',
                }
                continue

            try:
                import statsmodels.api as sm

                X_arr = X.values
                X_with_const = np.column_stack([np.ones(len(X_arr)), X_arr])
                y_arr = y.values

                model = sm.Logit(y_arr, X_with_const)
                # Use regularization to handle near-perfect separation
                fit_result = model.fit_regularized(
                    method='l1', alpha=1e-4, disp=0, trim_mode='off'
                )

                params = fit_result.params
                beta_linear = float(params[1])
                beta_quadratic = float(params[2])

                # For regularized fits we need to get p-values via
                # the unregularized model if possible; fall back gracefully.
                try:
                    unreg_result = model.fit(disp=0, maxiter=300)
                    pvalues = unreg_result.pvalues
                    pseudo_r2 = float(unreg_result.prsquared)
                    p_values_dict = {
                        'intercept': float(pvalues[0]),
                        'beta_linear': float(pvalues[1]),
                        'beta_quadratic': float(pvalues[2]),
                    }
                    # Prefer unregularized coefficients when available
                    beta_linear = float(unreg_result.params[1])
                    beta_quadratic = float(unreg_result.params[2])
                except Exception:
                    p_values_dict = None
                    pseudo_r2 = None

                contagion_type = (
                    'saturating' if beta_quadratic < 0 else 'accelerating'
                )

                results[cascade_type] = {
                    'beta_linear': beta_linear,
                    'beta_quadratic': beta_quadratic,
                    'p_values': p_values_dict,
                    'pseudo_r2': pseudo_r2,
                    'contagion_type': contagion_type,
                    'n_cascades': len(cascades),
                    'n_observations': len(df),
                    'n_adopted': int(y.sum()),
                    'adoption_rate': float(y.mean()),
                }

                logger.info(
                    f"Logistic model [{cascade_type}]: "
                    f"beta_linear={beta_linear:.4f}, "
                    f"beta_quadratic={beta_quadratic:.4f}, "
                    f"contagion={contagion_type}"
                )

            except ImportError:
                logger.warning(
                    "statsmodels not available; skipping logistic fit for "
                    f"type '{cascade_type}'"
                )
                results[cascade_type] = {
                    'beta_linear': None,
                    'beta_quadratic': None,
                    'p_values': None,
                    'pseudo_r2': None,
                    'contagion_type': None,
                    'n_cascades': len(cascades),
                    'result': 'statsmodels_not_installed',
                }
            except Exception as e:
                logger.error(
                    f"Logistic model fitting failed for type '{cascade_type}': {e}"
                )
                results[cascade_type] = {
                    'beta_linear': None,
                    'beta_quadratic': None,
                    'p_values': None,
                    'pseudo_r2': None,
                    'contagion_type': None,
                    'n_cascades': len(cascades),
                    'result': 'fitting_failed',
                    'error': str(e),
                }

        # Write output
        output_path = Path('output')
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'rq2_logistic_by_type.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Logistic by type results written to output/rq2_logistic_by_type.json "
            f"({len(results)} types)"
        )

        return results



class CascadeAnalyzer:
    """Analyzes cascade statistics and distributions."""
    
    def __init__(self, cascades: list[Cascade]) -> None:
        """Initialize cascade analyzer.
        
        Args:
            cascades: List of cascades to analyze.
        """
        self.cascades = cascades
    
    def compute_cascade_statistics(self) -> pd.DataFrame:
        """Compute statistics for each cascade.
        
        Returns:
            DataFrame with cascade metrics.
        """
        stats_list = []
        
        for cascade in self.cascades:
            unique_adopters = set(a[0] for a in cascade.adoptions)
            
            if cascade.adoptions:
                timestamps = [a[1] for a in cascade.adoptions]
                duration = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            else:
                duration = 0
            
            stats_list.append({
                'cascade_id': cascade.cascade_id,
                'cascade_type': cascade.cascade_type,
                'n_adoptions': len(cascade.adoptions),
                'n_unique_adopters': len(unique_adopters),
                'duration_hours': duration,
                'seed_agent': cascade.seed_agent,
            })
        
        return pd.DataFrame(stats_list)
    
    def test_power_law(self) -> dict:
        """Test if cascade sizes follow a power-law distribution.
        
        Uses the powerlaw package implementing Clauset et al. (2009) methodology
        for rigorous power-law testing with comparison to alternative distributions.
        
        Returns:
            Dict with test results including p-values and distribution comparisons.
        """
        sizes = [len(set(a[0] for a in c.adoptions)) for c in self.cascades]
        
        if len(sizes) < 10:
            return {'result': 'insufficient_data', 'n_cascades': len(sizes)}
        
        sizes = np.array(sizes, dtype=float)
        
        try:
            import powerlaw
            
            # Fit power-law distribution
            fit = powerlaw.Fit(sizes, discrete=True, verbose=False)
            
            # Get power-law parameters
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            sigma = fit.power_law.sigma  # Standard error of alpha
            
            # Compare power-law to alternative distributions
            # Lognormal comparison
            R_lognormal, p_lognormal = fit.distribution_compare('power_law', 'lognormal')
            
            # Exponential comparison
            R_exponential, p_exponential = fit.distribution_compare('power_law', 'exponential')
            
            # Truncated power-law comparison
            R_truncated, p_truncated = fit.distribution_compare('power_law', 'truncated_power_law')
            
            # KS test for goodness of fit
            # Note: fit.power_law.KS() has a bug in some versions, use D attribute instead
            try:
                ks_stat = fit.power_law.D
            except:
                # Fallback: compute KS statistic manually
                from scipy import stats as scipy_stats
                theoretical_cdf = fit.power_law.cdf(sizes[sizes >= xmin])
                empirical_cdf = np.arange(1, len(sizes[sizes >= xmin]) + 1) / len(sizes[sizes >= xmin])
                ks_stat = np.max(np.abs(theoretical_cdf - empirical_cdf)) if len(theoretical_cdf) > 0 else None
            
            # p-value placeholder (bootstrap is expensive)
            p_value = None
            
            results = {
                'alpha': float(alpha),
                'alpha_se': float(sigma),
                'alpha_ci_lower': float(alpha - 1.96 * sigma),
                'alpha_ci_upper': float(alpha + 1.96 * sigma),
                'xmin': float(xmin),
                'ks_statistic': float(ks_stat),
                'n_cascades': len(sizes),
                'n_above_xmin': int(np.sum(sizes >= xmin)),
                # Distribution comparisons (positive R favors power-law)
                'vs_lognormal': {
                    'loglikelihood_ratio': float(R_lognormal),
                    'p_value': float(p_lognormal),
                    'favors_powerlaw': R_lognormal > 0 and p_lognormal < 0.05,
                },
                'vs_exponential': {
                    'loglikelihood_ratio': float(R_exponential),
                    'p_value': float(p_exponential),
                    'favors_powerlaw': R_exponential > 0 and p_exponential < 0.05,
                },
                'vs_truncated_powerlaw': {
                    'loglikelihood_ratio': float(R_truncated),
                    'p_value': float(p_truncated),
                    'favors_powerlaw': R_truncated > 0 and p_truncated < 0.05,
                },
                'method': 'powerlaw_package_clauset2009',
            }
            
            logger.info(f"Power-law fit: α={alpha:.3f}±{sigma:.3f}, xmin={xmin}, KS={ks_stat:.4f}")
            logger.info(f"vs lognormal: R={R_lognormal:.3f}, p={p_lognormal:.4f}")
            
            return results
            
        except ImportError:
            logger.warning("powerlaw package not available, using manual estimation")
            
            # Fallback to manual MLE estimation
            x_min = sizes.min()
            n = len(sizes)
            
            if x_min <= 0:
                return {'result': 'invalid_data'}
            
            alpha = 1 + n / np.sum(np.log(sizes / x_min))
            
            # Standard error approximation
            sigma = (alpha - 1) / np.sqrt(n)
            
            # KS test against fitted power-law
            sorted_sizes = np.sort(sizes)
            empirical_cdf = np.arange(1, n + 1) / n
            theoretical_cdf = 1 - (x_min / sorted_sizes) ** (alpha - 1)
            ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
            
            return {
                'alpha': float(alpha),
                'alpha_se': float(sigma),
                'alpha_ci_lower': float(alpha - 1.96 * sigma),
                'alpha_ci_upper': float(alpha + 1.96 * sigma),
                'xmin': float(x_min),
                'ks_statistic': float(ks_stat),
                'n_cascades': n,
                'method': 'manual_mle_fallback',
                'warning': 'Install powerlaw package for rigorous testing',
            }
    
    def test_power_law_by_type(self) -> dict[str, dict]:
        """Fit separate power-law distributions for meme and skill cascades.

        Uses the powerlaw package implementing Clauset et al. (2009) methodology
        for rigorous power-law testing, applied independently to each cascade type.

        Returns:
            Dict keyed by cascade_type ('meme', 'skill', etc.) with each value
            containing: alpha, x_min, ks_statistic, lognormal_lr, lognormal_p.
            Types with insufficient data report null values.
        """
        # Group cascades by type
        type_cascades: dict[str, list[Cascade]] = defaultdict(list)
        for cascade in self.cascades:
            type_cascades[cascade.cascade_type].append(cascade)

        results: dict[str, dict] = {}

        for cascade_type, cascades in type_cascades.items():
            # Compute unique adopter counts per cascade
            sizes = [len(set(a[0] for a in c.adoptions)) for c in cascades]

            if len(sizes) < 10:
                logger.warning(
                    f"Insufficient cascades for power-law fit on type '{cascade_type}': "
                    f"{len(sizes)} cascades (need >= 10)"
                )
                results[cascade_type] = {
                    'alpha': None,
                    'x_min': None,
                    'ks_statistic': None,
                    'lognormal_lr': None,
                    'lognormal_p': None,
                    'n_cascades': len(sizes),
                    'result': 'insufficient_data',
                }
                continue

            sizes_arr = np.array(sizes, dtype=float)

            try:
                import powerlaw

                fit = powerlaw.Fit(sizes_arr, discrete=True, verbose=False)

                alpha = fit.power_law.alpha
                xmin = fit.power_law.xmin
                sigma = fit.power_law.sigma

                # Lognormal comparison
                R_lognormal, p_lognormal = fit.distribution_compare(
                    'power_law', 'lognormal'
                )

                # KS statistic
                try:
                    ks_stat = fit.power_law.D
                except Exception:
                    ks_stat = None

                results[cascade_type] = {
                    'alpha': float(alpha),
                    'x_min': float(xmin),
                    'ks_statistic': float(ks_stat) if ks_stat is not None else None,
                    'lognormal_lr': float(R_lognormal),
                    'lognormal_p': float(p_lognormal),
                    'alpha_se': float(sigma),
                    'n_cascades': len(sizes),
                    'n_above_xmin': int(np.sum(sizes_arr >= xmin)),
                }

                logger.info(
                    f"Power-law fit [{cascade_type}]: α={alpha:.3f}±{sigma:.3f}, "
                    f"xmin={xmin}, KS={ks_stat}, "
                    f"vs lognormal: R={R_lognormal:.3f}, p={p_lognormal:.4f}"
                )

            except ImportError:
                logger.warning(
                    "powerlaw package not available; skipping fit for "
                    f"type '{cascade_type}'"
                )
                results[cascade_type] = {
                    'alpha': None,
                    'x_min': None,
                    'ks_statistic': None,
                    'lognormal_lr': None,
                    'lognormal_p': None,
                    'n_cascades': len(sizes),
                    'result': 'powerlaw_not_installed',
                }

        # Write output
        output_path = Path('output')
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'rq2_power_law_by_type.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Power-law by type results written to output/rq2_power_law_by_type.json "
            f"({len(results)} types)"
        )

        return results

    def export_figure6_data(self) -> dict:
        """Export meme and skill cascade size arrays for two-panel Figure 6.

        Groups cascades by cascade_type, computes cascade sizes (unique adopter
        counts), and optionally includes power-law fit parameters (alpha, x_min)
        from :meth:`test_power_law_by_type` when available.

        Writes output to ``output/rq2_figure6_data.json``.

        Returns:
            Dict keyed by 'meme' and 'skill' with sizes, n_cascades, alpha, x_min.
        """
        # Group cascades by type
        type_cascades: dict[str, list[Cascade]] = defaultdict(list)
        for cascade in self.cascades:
            type_cascades[cascade.cascade_type].append(cascade)

        # Try to get power-law fit results
        power_law_results: dict[str, dict] = {}
        try:
            power_law_results = self.test_power_law_by_type()
        except Exception:
            logger.warning("Could not obtain power-law fit results for figure 6 data")

        data: dict[str, dict] = {}
        for ctype in ("meme", "skill"):
            cascades = type_cascades.get(ctype, [])
            sizes = sorted(
                [len(set(a[0] for a in c.adoptions)) for c in cascades],
                reverse=True,
            )
            pl = power_law_results.get(ctype, {})
            data[ctype] = {
                "sizes": sizes,
                "n_cascades": len(sizes),
                "alpha": pl.get("alpha"),
                "x_min": pl.get("x_min"),
            }

        # Write output
        output_path = Path("output")
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "rq2_figure6_data.json", "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Figure 6 data written to output/rq2_figure6_data.json "
            f"(meme: {data['meme']['n_cascades']}, skill: {data['skill']['n_cascades']})"
        )

        return data

    def compare_distributions(self) -> dict:
        """Compare cascade size distributions across types.
        
        Returns:
            Dict with comparison statistics.
        """
        type_sizes = defaultdict(list)
        
        for cascade in self.cascades:
            size = len(set(a[0] for a in cascade.adoptions))
            type_sizes[cascade.cascade_type].append(size)
        
        results = {}
        
        types = list(type_sizes.keys())
        for i, type1 in enumerate(types):
            for type2 in types[i+1:]:
                sizes1 = type_sizes[type1]
                sizes2 = type_sizes[type2]
                
                if len(sizes1) >= 5 and len(sizes2) >= 5:
                    stat, p_value = stats.mannwhitneyu(sizes1, sizes2)
                    results[f'{type1}_vs_{type2}'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'mean_1': np.mean(sizes1),
                        'mean_2': np.mean(sizes2),
                    }
        
        return results

    def export_figure6_data(self) -> dict:
        """Export meme and skill cascade size arrays for two-panel Figure 6.

        Groups cascades by cascade_type, computes cascade sizes (unique adopter
        counts), and optionally includes power-law fit parameters (alpha, x_min)
        from :meth:`test_power_law_by_type` when available.

        Writes output to ``output/rq2_figure6_data.json``.

        Returns:
            Dict keyed by 'meme' and 'skill' with sizes, n_cascades, alpha, x_min.
        """
        # Group cascades by type
        type_cascades: dict[str, list[Cascade]] = defaultdict(list)
        for cascade in self.cascades:
            type_cascades[cascade.cascade_type].append(cascade)

        # Try to get power-law fit results
        power_law_results: dict[str, dict] = {}
        try:
            power_law_results = self.test_power_law_by_type()
        except Exception:
            logger.warning("Could not obtain power-law fit results for figure 6 data")

        data: dict[str, dict] = {}
        for ctype in ("meme", "skill"):
            cascades = type_cascades.get(ctype, [])
            sizes = sorted(
                [len(set(a[0] for a in c.adoptions)) for c in cascades],
                reverse=True,
            )
            pl = power_law_results.get(ctype, {})
            data[ctype] = {
                "sizes": sizes,
                "n_cascades": len(sizes),
                "alpha": pl.get("alpha"),
                "x_min": pl.get("x_min"),
            }

        # Write output
        output_path = Path("output")
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "rq2_figure6_data.json", "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Figure 6 data written to output/rq2_figure6_data.json "
            f"(meme: {data['meme']['n_cascades']}, skill: {data['skill']['n_cascades']})"
        )

        return data



class EmbeddingCascadeDetector:
    """Detects behavioral cascades via sentence-embedding drift.

    Uses a sentence encoder to compute weekly centroid embeddings per submolt,
    then identifies significant centroid shifts as behavioral cascade events.
    """

    BATCH_SIZE = 1000

    def __init__(
        self,
        storage: JSONStorage,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize embedding cascade detector.

        Args:
            storage: JSONStorage instance for post data.
            model_name: Sentence-transformer model name.
        """
        self.storage = storage
        self.model_name = model_name
        self.detection_failed = False
        self.methodology_note = ""
        self._centroids: Optional[pd.DataFrame] = None
        self._shifts: Optional[list[dict]] = None

        # Lazy import of sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingCascadeDetector. "
                "Install with: pip install sentence-transformers"
            )

    def compute_weekly_centroids(self) -> pd.DataFrame:
        """Compute per-submolt weekly centroid embeddings.

        Groups posts by (submolt, ISO week), encodes post bodies in batches
        of 1000, and computes the element-wise mean as the centroid.

        Returns:
            DataFrame with columns: submolt, week, centroid_embedding.
        """
        posts = self.storage.get_posts()

        # Build records with submolt, ISO week, and body
        records: list[dict] = []
        for post in posts:
            if post.body and post.created_at and post.submolt:
                iso_cal = post.created_at.isocalendar()
                week_key = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
                records.append({
                    "submolt": post.submolt,
                    "week": week_key,
                    "body": post.body,
                    "author_id": post.author_id,
                    "post_id": post.post_id,
                    "created_at": post.created_at,
                })

        if not records:
            self._centroids = pd.DataFrame(
                columns=["submolt", "week", "centroid_embedding"]
            )
            return self._centroids

        df = pd.DataFrame(records)

        centroid_rows: list[dict] = []
        for (submolt, week), group in df.groupby(["submolt", "week"]):
            bodies = group["body"].tolist()

            # Encode in batches of BATCH_SIZE to avoid OOM
            all_embeddings = []
            for i in range(0, len(bodies), self.BATCH_SIZE):
                batch = bodies[i : i + self.BATCH_SIZE]
                embeddings = self.encoder.encode(batch, show_progress_bar=False)
                all_embeddings.append(np.array(embeddings))

            stacked = np.vstack(all_embeddings)
            centroid = stacked.mean(axis=0)

            centroid_rows.append({
                "submolt": submolt,
                "week": week,
                "centroid_embedding": centroid,
            })

        self._centroids = pd.DataFrame(centroid_rows)
        logger.info(
            f"Computed {len(centroid_rows)} weekly centroids across "
            f"{self._centroids['submolt'].nunique()} submolts"
        )
        return self._centroids

    def detect_significant_shifts(
        self, threshold_sd: float = 2.0
    ) -> list[dict]:
        """Identify submolts with centroid shifts > threshold_sd standard deviations.

        For each submolt, computes cosine distance between consecutive weekly
        centroids. A shift is significant if cosine_distance > mean + threshold_sd * SD
        across all submolt-week pairs.

        Args:
            threshold_sd: Number of standard deviations above mean for significance.

        Returns:
            List of dicts with submolt, week_from, week_to, shift_magnitude,
            seed_agent, seed_post_id.
        """
        if self._centroids is None:
            self.compute_weekly_centroids()

        centroids_df = self._centroids
        if centroids_df.empty:
            self._shifts = []
            return self._shifts

        # Compute all consecutive cosine distances
        all_shift_records: list[dict] = []

        for submolt, group in centroids_df.groupby("submolt"):
            group_sorted = group.sort_values("week").reset_index(drop=True)
            if len(group_sorted) < 2:
                continue

            for i in range(len(group_sorted) - 1):
                c1 = group_sorted.iloc[i]["centroid_embedding"]
                c2 = group_sorted.iloc[i + 1]["centroid_embedding"]

                # Cosine distance = 1 - cosine_similarity
                dot = np.dot(c1, c2)
                norm1 = np.linalg.norm(c1)
                norm2 = np.linalg.norm(c2)
                if norm1 == 0 or norm2 == 0:
                    cos_dist = 1.0
                else:
                    cos_sim = dot / (norm1 * norm2)
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    cos_dist = 1.0 - cos_sim

                all_shift_records.append({
                    "submolt": submolt,
                    "week_from": group_sorted.iloc[i]["week"],
                    "week_to": group_sorted.iloc[i + 1]["week"],
                    "shift_magnitude": float(cos_dist),
                })

        if not all_shift_records:
            self._shifts = []
            return self._shifts

        # Compute threshold: mean + threshold_sd * SD
        magnitudes = np.array([r["shift_magnitude"] for r in all_shift_records])
        mean_shift = float(np.mean(magnitudes))
        sd_shift = float(np.std(magnitudes))
        threshold = mean_shift + threshold_sd * sd_shift

        # Identify significant shifts
        significant: list[dict] = []
        posts = self.storage.get_posts()

        for record in all_shift_records:
            if record["shift_magnitude"] > threshold:
                # Find seed agent: first poster in the target week for this submolt
                seed_agent = ""
                seed_post_id = ""
                week_to = record["week_to"]
                submolt = record["submolt"]

                for post in posts:
                    if (
                        post.submolt == submolt
                        and post.created_at
                        and post.body
                    ):
                        iso_cal = post.created_at.isocalendar()
                        post_week = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
                        if post_week == week_to:
                            seed_agent = post.author_id
                            seed_post_id = post.post_id
                            break

                significant.append({
                    "submolt": submolt,
                    "week_from": record["week_from"],
                    "week_to": record["week_to"],
                    "shift_magnitude": record["shift_magnitude"],
                    "seed_agent": seed_agent,
                    "seed_post_id": seed_post_id,
                })

        logger.info(
            f"Detected {len(significant)} significant shifts "
            f"(threshold={threshold:.4f}, mean={mean_shift:.4f}, sd={sd_shift:.4f})"
        )
        self._shifts = significant
        return self._shifts

    def build_behavioral_cascades(self) -> list[Cascade]:
        """Convert detected shifts into Cascade objects.

        If fewer than 50 cascades detected, returns empty list and sets
        self.detection_failed = True with a methodology note.

        Returns:
            List of Cascade objects with cascade_type='behavioral'.
        """
        if self._shifts is None:
            self.detect_significant_shifts()

        shifts = self._shifts
        cascades: list[Cascade] = []

        for i, shift in enumerate(shifts):
            cascade = Cascade(
                cascade_id=f"emb_behavioral_{i:04d}",
                cascade_type="behavioral",
                seed_agent=shift.get("seed_agent", ""),
                seed_time=datetime.now(),  # placeholder; real time from data
                adoptions=[],
                content_hash=f"{shift['submolt']}_{shift['week_from']}_{shift['week_to']}",
            )
            cascades.append(cascade)

        if len(cascades) < 50:
            self.detection_failed = True
            self.methodology_note = (
                "Embedding-drift behavioral cascade detection identified fewer than "
                f"50 cascades ({len(cascades)} detected). Behavioral cascades are "
                "excluded from quantitative analyses. The regex-based approach "
                "similarly yielded limited behavioral cascade counts, suggesting "
                "that behavioral pattern propagation in this dataset operates at "
                "a granularity not well captured by weekly centroid drift."
            )
            logger.warning(
                f"Behavioral cascade detection failed: only {len(cascades)} cascades "
                f"detected (minimum 50 required)"
            )
            return []
        else:
            self.detection_failed = False
            self.methodology_note = ""
            logger.info(f"Built {len(cascades)} behavioral cascades from embedding drift")
            return cascades


def verify_cascade_ordering(cascade: Cascade) -> bool:
    """Verify that cascade adoptions are in chronological order.
    
    Args:
        cascade: Cascade to verify.
        
    Returns:
        True if adoptions are monotonically ordered by timestamp.
    """
    if len(cascade.adoptions) <= 1:
        return True
    
    for i in range(len(cascade.adoptions) - 1):
        if cascade.adoptions[i][1] > cascade.adoptions[i + 1][1]:
            return False
    
    return True


def save_rq2_data(
    storage: JSONStorage,
    network: nx.DiGraph,
    config: Config,
    output_dir: str,
) -> dict:
    """Save all RQ2 analysis data to files.
    
    Args:
        storage: JSONStorage instance.
        network: Interaction network.
        config: Configuration parameters.
        output_dir: Directory to save data files.
        
    Returns:
        Dict with paths to saved files and summary statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    # Identify all cascades
    identifier = CascadeIdentifier(storage, config)
    
    logger.info("Identifying meme cascades...")
    meme_cascades = identifier.identify_meme_cascades()
    
    logger.info("Identifying skill cascades...")
    skill_cascades = identifier.identify_skill_cascades()
    
    logger.info("Identifying behavioral cascades...")
    behavioral_cascades = identifier.identify_behavioral_cascades()
    
    all_cascades = meme_cascades + skill_cascades + behavioral_cascades
    
    # 1. Save cascade metadata
    cascade_metadata = []
    for cascade in all_cascades:
        unique_adopters = set(a[0] for a in cascade.adoptions)
        timestamps = [a[1] for a in cascade.adoptions]
        
        cascade_metadata.append({
            'cascade_id': cascade.cascade_id,
            'cascade_type': cascade.cascade_type,
            'seed_agent': cascade.seed_agent,
            'seed_time': cascade.seed_time.isoformat() if cascade.seed_time else None,
            'n_adoptions': len(cascade.adoptions),
            'n_unique_adopters': len(unique_adopters),
            'duration_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600 if len(timestamps) > 1 else 0,
            'content_hash': cascade.content_hash,
        })
    
    cascade_df = pd.DataFrame(cascade_metadata)
    cascade_df.to_csv(output_path / 'rq2_cascade_metadata.csv', index=False)
    saved_files['cascade_metadata'] = str(output_path / 'rq2_cascade_metadata.csv')
    
    # 2. Save all adoptions (detailed)
    all_adoptions = []
    for cascade in all_cascades:
        for agent, timestamp in cascade.adoptions:
            all_adoptions.append({
                'cascade_id': cascade.cascade_id,
                'cascade_type': cascade.cascade_type,
                'agent_id': agent,
                'adoption_time': timestamp.isoformat() if timestamp else None,
            })
    
    adoptions_df = pd.DataFrame(all_adoptions)
    adoptions_df.to_csv(output_path / 'rq2_cascade_adoptions.csv', index=False)
    saved_files['cascade_adoptions'] = str(output_path / 'rq2_cascade_adoptions.csv')
    
    # 3. Cascade statistics
    analyzer = CascadeAnalyzer(all_cascades)
    cascade_stats = analyzer.compute_cascade_statistics()
    cascade_stats.to_csv(output_path / 'rq2_cascade_statistics.csv', index=False)
    saved_files['cascade_statistics'] = str(output_path / 'rq2_cascade_statistics.csv')
    
    # Helper function to convert numpy types recursively
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    # 4. Power-law analysis
    power_law_results = analyzer.test_power_law()
    power_law_clean = convert_numpy_types(power_law_results)
    with open(output_path / 'rq2_power_law_analysis.json', 'w') as f:
        json.dump(power_law_clean, f, indent=2)
    saved_files['power_law_analysis'] = str(output_path / 'rq2_power_law_analysis.json')
    
    # 5. Distribution comparison across types
    distribution_comparison = analyzer.compare_distributions()
    dist_clean = convert_numpy_types(distribution_comparison)
    with open(output_path / 'rq2_distribution_comparison.json', 'w') as f:
        json.dump(dist_clean, f, indent=2)
    saved_files['distribution_comparison'] = str(output_path / 'rq2_distribution_comparison.json')
    
    # 6. Diffusion modeling
    modeler = DiffusionModeler(all_cascades, network, config)
    
    # Logistic model results
    logistic_results = modeler.fit_logistic_model()
    logistic_clean = convert_numpy_types(logistic_results)
    with open(output_path / 'rq2_logistic_model.json', 'w') as f:
        json.dump(logistic_clean, f, indent=2)
    saved_files['logistic_model'] = str(output_path / 'rq2_logistic_model.json')
    
    # Cox hazards model
    cox_results = modeler.fit_cox_hazards()
    cox_clean = convert_numpy_types(cox_results)
    with open(output_path / 'rq2_cox_hazards.json', 'w') as f:
        json.dump(cox_clean, f, indent=2)
    saved_files['cox_hazards'] = str(output_path / 'rq2_cox_hazards.json')
    
    # Contagion type classification
    contagion_result = modeler.test_contagion_type()
    contagion_clean = convert_numpy_types(contagion_result)
    with open(output_path / 'rq2_contagion_classification.json', 'w') as f:
        json.dump(contagion_clean, f, indent=2)
    saved_files['contagion_classification'] = str(output_path / 'rq2_contagion_classification.json')
    
    # 7. Exposure data for sample cascades
    exposure_data = []
    sample_cascades = all_cascades[:50]  # Sample first 50 cascades only
    for cascade in sample_cascades:
        exposure_df = modeler.compute_exposures(cascade)
        exposure_df['cascade_id'] = cascade.cascade_id
        exposure_data.append(exposure_df)
    
    if exposure_data:
        all_exposures = pd.concat(exposure_data, ignore_index=True)
        all_exposures.to_csv(output_path / 'rq2_exposure_data.csv', index=False)
        saved_files['exposure_data'] = str(output_path / 'rq2_exposure_data.csv')
    
    # 8. Cascade size distribution
    sizes = [len(set(a[0] for a in c.adoptions)) for c in all_cascades]
    size_dist = pd.DataFrame({
        'cascade_id': [c.cascade_id for c in all_cascades],
        'cascade_type': [c.cascade_type for c in all_cascades],
        'size': sizes,
    })
    size_dist.to_csv(output_path / 'rq2_cascade_sizes.csv', index=False)
    saved_files['cascade_sizes'] = str(output_path / 'rq2_cascade_sizes.csv')
    
    # 9. Summary statistics
    summary = {
        'n_meme_cascades': len(meme_cascades),
        'n_skill_cascades': len(skill_cascades),
        'n_behavioral_cascades': len(behavioral_cascades),
        'total_cascades': len(all_cascades),
        'total_adoptions': len(all_adoptions),
        # Power-law results
        'power_law_alpha': float(power_law_results.get('alpha')) if power_law_results.get('alpha') is not None else None,
        'power_law_alpha_ci': [
            float(power_law_results.get('alpha_ci_lower')) if power_law_results.get('alpha_ci_lower') is not None else None,
            float(power_law_results.get('alpha_ci_upper')) if power_law_results.get('alpha_ci_upper') is not None else None,
        ],
        'power_law_ks': float(power_law_results.get('ks_statistic')) if power_law_results.get('ks_statistic') is not None else None,
        'power_law_method': power_law_results.get('method'),
        # Contagion classification
        'contagion_type': contagion_result.get('classification'),
        'contagion_evidence': contagion_result.get('evidence'),
        # Logistic model
        'logistic_beta_linear': float(logistic_results.get('beta_linear')) if logistic_results.get('beta_linear') is not None else None,
        'logistic_beta_linear_p': float(logistic_results.get('beta_linear_p')) if logistic_results.get('beta_linear_p') is not None else None,
        'logistic_beta_quadratic': float(logistic_results.get('beta_quadratic')) if logistic_results.get('beta_quadratic') is not None else None,
        'logistic_beta_quadratic_p': float(logistic_results.get('beta_quadratic_p')) if logistic_results.get('beta_quadratic_p') is not None else None,
        'logistic_pseudo_r2': float(logistic_results.get('pseudo_r2')) if logistic_results.get('pseudo_r2') is not None else None,
        # Cox model
        'cox_hazard_ratio': float(cox_results.get('hazard_ratio')) if cox_results.get('hazard_ratio') is not None else None,
        'cox_hazard_ratio_ci': [
            float(cox_results.get('hazard_ratio_ci_lower')) if cox_results.get('hazard_ratio_ci_lower') is not None else None,
            float(cox_results.get('hazard_ratio_ci_upper')) if cox_results.get('hazard_ratio_ci_upper') is not None else None,
        ],
        'cox_p_value': float(cox_results.get('p_value')) if cox_results.get('p_value') is not None else None,
        'cox_concordance_index': float(cox_results.get('concordance_index')) if cox_results.get('concordance_index') is not None else None,
        # Cascade size statistics
        'mean_cascade_size': float(np.mean(sizes)) if sizes else 0,
        'median_cascade_size': float(np.median(sizes)) if sizes else 0,
        'max_cascade_size': int(max(sizes)) if sizes else 0,
    }
    with open(output_path / 'rq2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    saved_files['summary'] = str(output_path / 'rq2_summary.json')
    
    logger.info(f"Saved {len(saved_files)} RQ2 data files to {output_path}")
    return saved_files
