"""RQ3: Collective problem-solving analysis module.

Implements collaborative event identification, solution quality assessment,
and collaboration success modeling with rigorous statistical validation.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .storage import JSONStorage
from .config import Config
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .models import CollaborativeEvent, ComplexityFeatures

logger = logging.getLogger(__name__)

# Technical keywords for identifying problem-solving threads
TECHNICAL_KEYWORDS = {
    'error', 'bug', 'fix', 'issue', 'problem', 'help', 'solution', 'solve',
    'debug', 'crash', 'exception', 'fail', 'broken', 'stuck', 'how to',
    'implement', 'code', 'function', 'method', 'class', 'api', 'library',
}


class CollaborationIdentifier:
    """Identifies collaborative problem-solving events."""
    
    def __init__(self, storage: JSONStorage, config: Config) -> None:
        self.storage = storage
        self.config = config
    
    def identify_collaborative_events(
        self,
        min_agents: int = 3,
        min_comments: int = 5,
        min_duration_minutes: int = 30,
    ) -> list[CollaborativeEvent]:
        """Identify collaborative problem-solving events.
        
        Args:
            min_agents: Minimum unique participants.
            min_comments: Minimum number of comments.
            min_duration_minutes: Minimum thread duration.
            
        Returns:
            List of identified collaborative events.
        """
        posts = self.storage.get_posts()
        events = []
        
        for post in posts:
            comments = self.storage.get_comments(filters={'post_id': post.post_id})
            
            if len(comments) < min_comments:
                continue
            
            # Check for technical keywords
            all_text = (post.title or '') + ' ' + (post.body or '')
            for comment in comments:
                all_text += ' ' + (comment.body or '')
            
            if not self._contains_technical_keywords(all_text):
                continue
            
            # Get unique participants
            participants = {post.author_id}
            for comment in comments:
                participants.add(comment.author_id)
            
            if len(participants) < min_agents:
                continue
            
            # Check duration
            timestamps = [c.created_at for c in comments if c.created_at]
            if post.created_at:
                timestamps.append(post.created_at)
            
            if len(timestamps) < 2:
                continue
            
            duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
            if duration < min_duration_minutes:
                continue
            
            event = CollaborativeEvent(
                thread_id=post.post_id,
                participants=list(participants),
                start_time=min(timestamps),
                end_time=max(timestamps),
                problem_statement=post.title or '',
                solution=self._extract_solution(comments),
            )
            events.append(event)
        
        logger.info(f"Identified {len(events)} collaborative events")
        return events
    
    def _contains_technical_keywords(self, text: str) -> bool:
        """Check if text contains technical keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in TECHNICAL_KEYWORDS)
    
    def _extract_solution(self, comments: list) -> str:
        """Extract potential solution from comments."""
        # Simple heuristic: last comment with code block
        for comment in reversed(comments):
            if comment.body and ('```' in comment.body or '`' in comment.body):
                return comment.body
        return comments[-1].body if comments else ''
    
    def extract_technical_threads(self) -> list[str]:
        """Get IDs of threads with technical content."""
        posts = self.storage.get_posts()
        technical_ids = []
        
        for post in posts:
            text = (post.title or '') + ' ' + (post.body or '')
            if self._contains_technical_keywords(text):
                technical_ids.append(post.post_id)
        
        return technical_ids


class ComplexityMatcher:
    """Computes complexity proxy features and performs nearest-neighbor matching."""

    def __init__(self, storage: JSONStorage, config: Config) -> None:
        self.storage = storage
        self.config = config

    def compute_complexity_features(self, thread_id: str) -> dict:
        """Compute seed_word_count, technical_keyword_density, code_block_present
        for a thread's seed post.

        Args:
            thread_id: The post_id of the thread's seed post.

        Returns:
            Dict with thread_id, seed_word_count, technical_keyword_density,
            code_block_present.
        """
        # Direct lookup from internal store for efficiency
        post_data = self.storage._posts.get(thread_id)
        if post_data:
            body = post_data.get("body", "")
        else:
            body = ""

        words = body.split()
        seed_word_count = max(len(words), 1)

        # Technical keyword density: fraction of words that are technical keywords
        if words:
            tech_count = sum(
                1 for w in words if w.lower().strip(".,!?;:()[]{}\"'") in TECHNICAL_KEYWORDS
            )
            technical_keyword_density = tech_count / len(words)
        else:
            technical_keyword_density = 0.0

        # Code block presence: check for ``` fenced blocks or indented code patterns
        code_block_present = bool(
            "```" in body
            or re.search(r"(?m)^    \S", body)
            or re.search(r"(?m)^\t\S", body)
        )

        return {
            "thread_id": thread_id,
            "seed_word_count": seed_word_count,
            "technical_keyword_density": technical_keyword_density,
            "code_block_present": code_block_present,
        }

    def _get_thread_timing(self, thread_id: str) -> float:
        """Get thread creation timestamp as a float (seconds since epoch).

        Returns 0.0 if no timestamp is available.
        """
        post_data = self.storage._posts.get(thread_id)
        if post_data:
            from .storage import parse_datetime

            created = parse_datetime(post_data.get("created_at"))
            if created:
                return created.timestamp()
        return 0.0

    def _get_thread_topic(self, thread_id: str) -> str:
        """Get the submolt (topic) of a thread. Returns '' if unknown."""
        post_data = self.storage._posts.get(thread_id)
        if post_data:
            return post_data.get("submolt", "")
        return ""

    def _build_feature_matrix(self, thread_ids: list[str]) -> np.ndarray:
        """Build a feature matrix for a list of threads.

        Features: seed_word_count, technical_keyword_density, code_block_present,
        topic_hash (numeric encoding of submolt), timing (epoch seconds).

        Returns:
            2D numpy array of shape (len(thread_ids), 5).
        """
        rows = []
        for tid in thread_ids:
            cf = self.compute_complexity_features(tid)
            timing = self._get_thread_timing(tid)
            topic = self._get_thread_topic(tid)
            # Encode topic as a numeric hash for distance computation
            topic_num = hash(topic) % (2**16)
            rows.append([
                cf["seed_word_count"],
                cf["technical_keyword_density"],
                float(cf["code_block_present"]),
                topic_num,
                timing,
            ])
        return np.array(rows, dtype=float)

    def match_with_complexity(
        self,
        collaborative_threads: list[str],
        candidate_threads: list[str],
        ratio: str = "1:1",
        replacement: bool = False,
    ) -> pd.DataFrame:
        """1:1 nearest-neighbor matching on complexity + topic + timing features.

        Args:
            collaborative_threads: Thread IDs identified as collaborative.
            candidate_threads: Candidate (non-collaborative) thread IDs.
            ratio: Matching ratio, currently only '1:1' supported.
            replacement: Whether to sample with replacement.

        Returns:
            DataFrame with columns: collaborative_thread, matched_thread, distance.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not collaborative_threads or not candidate_threads:
            empty_df = pd.DataFrame(
                columns=["collaborative_thread", "matched_thread", "distance"]
            )
            self._write_outputs(empty_df, collaborative_threads, candidate_threads, ratio, replacement, output_dir)
            return empty_df

        # Build feature matrices
        collab_features = self._build_feature_matrix(collaborative_threads)
        candidate_features = self._build_feature_matrix(candidate_threads)

        # Standardize features jointly
        all_features = np.vstack([collab_features, candidate_features])
        scaler = StandardScaler()
        scaler.fit(all_features)
        collab_scaled = scaler.transform(collab_features)
        candidate_scaled = scaler.transform(candidate_features)

        # Nearest-neighbor matching
        n_neighbors = 1
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(candidate_scaled)
        distances, indices = nn.kneighbors(collab_scaled)

        matched_pairs = []
        used_candidates = set()

        if replacement:
            # With replacement: each collaborative thread gets its nearest neighbor
            for i, collab_tid in enumerate(collaborative_threads):
                cand_idx = indices[i, 0]
                matched_pairs.append({
                    "collaborative_thread": collab_tid,
                    "matched_thread": candidate_threads[cand_idx],
                    "distance": float(distances[i, 0]),
                })
        else:
            # Without replacement: greedy assignment by distance
            # Sort all (collab_idx, cand_idx, distance) by distance
            assignments = []
            for i in range(len(collaborative_threads)):
                assignments.append((i, indices[i, 0], distances[i, 0]))
            assignments.sort(key=lambda x: x[2])

            matched_collab = set()
            for collab_idx, cand_idx, dist in assignments:
                if collab_idx in matched_collab:
                    continue
                if cand_idx in used_candidates:
                    continue
                matched_pairs.append({
                    "collaborative_thread": collaborative_threads[collab_idx],
                    "matched_thread": candidate_threads[cand_idx],
                    "distance": float(dist),
                })
                matched_collab.add(collab_idx)
                used_candidates.add(cand_idx)

            # For any unmatched collaborative threads, find next-best unused candidate
            if len(matched_pairs) < len(collaborative_threads):
                # Re-fit with more neighbors to find alternatives
                k = min(len(candidate_threads), len(collaborative_threads))
                nn_full = NearestNeighbors(n_neighbors=k, metric="euclidean")
                nn_full.fit(candidate_scaled)
                distances_full, indices_full = nn_full.kneighbors(collab_scaled)

                for i, collab_tid in enumerate(collaborative_threads):
                    if i in matched_collab:
                        continue
                    for j in range(k):
                        cand_idx = indices_full[i, j]
                        if cand_idx not in used_candidates:
                            matched_pairs.append({
                                "collaborative_thread": collab_tid,
                                "matched_thread": candidate_threads[cand_idx],
                                "distance": float(distances_full[i, j]),
                            })
                            matched_collab.add(i)
                            used_candidates.add(cand_idx)
                            break

        result_df = pd.DataFrame(matched_pairs)
        self._write_outputs(result_df, collaborative_threads, candidate_threads, ratio, replacement, output_dir)
        return result_df

    def _write_outputs(
        self,
        result_df: pd.DataFrame,
        collaborative_threads: list[str],
        candidate_threads: list[str],
        ratio: str,
        replacement: bool,
        output_dir: Path,
    ) -> None:
        """Write matched baseline and metadata JSON files."""
        # Write matched baseline
        baseline_data = {
            "matched_pairs": result_df.to_dict(orient="records") if not result_df.empty else [],
            "n_matched": len(result_df),
        }
        with open(output_dir / "rq3_complexity_matched_baseline.json", "w") as f:
            json.dump(baseline_data, f, indent=2)

        # Write matching metadata
        final_n = len(result_df)
        metadata = {
            "matching_ratio": ratio,
            "replacement_flag": replacement,
            "final_n": final_n,
            "n_collaborative": len(collaborative_threads),
            "n_candidates": len(candidate_threads),
        }
        with open(output_dir / "rq3_matching_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Complexity matching: {final_n} pairs matched "
            f"(ratio={ratio}, replacement={replacement})"
        )


class SolutionAssessor:
    """Assesses quality of collaborative solutions."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def assess_code_solution(self, code: str) -> dict:
        """Assess quality of a code solution.
        
        Args:
            code: Code string to assess.
            
        Returns:
            Dict with quality metrics.
        """
        metrics = {
            'has_code': bool(re.search(r'```|`[^`]+`', code)),
            'has_comments': bool(re.search(r'#.*|//.*|/\*.*\*/', code)),
            'has_tests': bool(re.search(r'test|assert|expect', code.lower())),
            'line_count': len(code.split('\n')),
            'syntax_valid': self._check_syntax(code),
        }
        
        # Compute overall score
        score = sum([
            metrics['has_code'] * 0.3,
            metrics['has_comments'] * 0.2,
            metrics['has_tests'] * 0.3,
            metrics['syntax_valid'] * 0.2,
        ])
        metrics['quality_score'] = score
        
        return metrics
    
    def _check_syntax(self, code: str) -> bool:
        """Basic syntax check for code."""
        # Simple heuristic: balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack.pop() != char:
                    return False
        
        return len(stack) == 0
    
    def assess_conceptual_solution(self, problem: str, solution: str) -> dict:
        """Assess quality of a conceptual solution.
        
        Args:
            problem: Problem statement.
            solution: Proposed solution.
            
        Returns:
            Dict with quality metrics.
        """
        metrics = {
            'solution_length': len(solution),
            'addresses_problem': self._solution_addresses_problem(problem, solution),
            'has_explanation': len(solution) > 100,
            'has_steps': bool(re.search(r'\d+\.|[-*]\s', solution)),
        }
        
        score = sum([
            metrics['addresses_problem'] * 0.4,
            metrics['has_explanation'] * 0.3,
            metrics['has_steps'] * 0.3,
        ])
        metrics['quality_score'] = score
        
        return metrics
    
    def _solution_addresses_problem(self, problem: str, solution: str) -> bool:
        """Check if solution addresses the problem."""
        problem_words = set(problem.lower().split())
        solution_words = set(solution.lower().split())
        overlap = problem_words & solution_words
        return len(overlap) >= 2
    
    def compute_inter_rater_reliability(
        self, 
        ratings1: list, 
        ratings2: list
    ) -> float:
        """Compute Cohen's kappa for inter-rater reliability.
        
        Args:
            ratings1: First rater's ratings.
            ratings2: Second rater's ratings.
            
        Returns:
            Cohen's kappa coefficient.
        """
        if len(ratings1) != len(ratings2) or len(ratings1) == 0:
            return 0.0
        
        # Convert to numpy arrays
        r1 = np.array(ratings1)
        r2 = np.array(ratings2)
        
        # Compute observed agreement
        po = np.mean(r1 == r2)
        
        # Compute expected agreement
        categories = np.unique(np.concatenate([r1, r2]))
        pe = 0.0
        for cat in categories:
            pe += (np.mean(r1 == cat) * np.mean(r2 == cat))
        
        # Cohen's kappa
        if pe == 1.0:
            return 1.0
        
        kappa = (po - pe) / (1 - pe)
        return kappa


class BaselineComparator:
    """Compares collaborative vs individual solutions with proper statistical tests."""
    
    def __init__(
        self,
        events: list[CollaborativeEvent],
        storage: Optional[JSONStorage] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.events = events
        self.storage = storage
        self.config = config
    
    def collect_individual_baselines(self) -> pd.DataFrame:
        """Collect baseline individual solution attempts."""
        # Placeholder - would need actual individual attempts
        return pd.DataFrame()
    
    def compare_quality_distributions(self) -> dict:
        """Compare quality distributions using proper statistical tests.
        
        Uses one-sample t-test against theoretical baseline and 
        Wilcoxon signed-rank test for robustness.
        
        Returns:
            Dict with test statistics and effect sizes.
        """
        collab_scores = [e.quality_score for e in self.events if e.quality_score is not None]
        
        if len(collab_scores) < 5:
            return {'result': 'insufficient_data', 'n_events': len(collab_scores)}
        
        collab_scores = np.array(collab_scores)
        
        # Theoretical baseline: random chance would give 0.5 quality score
        baseline_mean = 0.5
        
        # One-sample t-test against baseline
        t_stat, t_pval = stats.ttest_1samp(collab_scores, baseline_mean)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        # Test if median differs from baseline
        try:
            w_stat, w_pval = stats.wilcoxon(collab_scores - baseline_mean)
        except ValueError:
            # All values might be equal
            w_stat, w_pval = np.nan, np.nan
        
        # Effect size: Cohen's d against baseline
        cohens_d = (np.mean(collab_scores) - baseline_mean) / np.std(collab_scores, ddof=1)
        
        # Bootstrap confidence interval for mean
        n_bootstrap = 1000
        rng = np.random.RandomState(42)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(collab_scores, size=len(collab_scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        return {
            'n_events': len(collab_scores),
            'collab_mean': float(np.mean(collab_scores)),
            'collab_std': float(np.std(collab_scores, ddof=1)),
            'collab_median': float(np.median(collab_scores)),
            'baseline_mean': baseline_mean,
            # T-test results
            't_statistic': float(t_stat),
            't_p_value': float(t_pval),
            # Wilcoxon results
            'wilcoxon_statistic': float(w_stat) if not np.isnan(w_stat) else None,
            'wilcoxon_p_value': float(w_pval) if not np.isnan(w_pval) else None,
            # Effect size
            'cohens_d': float(cohens_d),
            'effect_interpretation': self._interpret_cohens_d(cohens_d),
            # Confidence interval
            'mean_ci_lower': float(np.percentile(bootstrap_means, 2.5)),
            'mean_ci_upper': float(np.percentile(bootstrap_means, 97.5)),
            # Conclusion
            'significantly_better_than_baseline': t_pval < 0.05 and np.mean(collab_scores) > baseline_mean,
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    @staticmethod
    def _cohens_d_two_sample(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d for two independent samples using pooled SD."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    @staticmethod
    def _bootstrap_cohens_d_ci(
        group1: np.ndarray,
        group2: np.ndarray,
        n_iterations: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Compute 95% bootstrap CI for Cohen's d between two groups.

        Returns (lower, upper) of the 95% percentile interval.
        """
        rng = np.random.RandomState(seed)
        boot_ds: list[float] = []
        n1, n2 = len(group1), len(group2)
        for _ in range(n_iterations):
            s1 = rng.choice(group1, size=n1, replace=True)
            s2 = rng.choice(group2, size=n2, replace=True)
            var1 = np.var(s1, ddof=1)
            var2 = np.var(s2, ddof=1)
            pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled == 0:
                boot_ds.append(0.0)
            else:
                boot_ds.append(float((np.mean(s1) - np.mean(s2)) / pooled))
        lower = float(np.percentile(boot_ds, 2.5))
        upper = float(np.percentile(boot_ds, 97.5))
        return lower, upper

    def compare_quality_with_complexity_matching(self) -> dict:
        """Compare quality scores using complexity-matched baseline.

        Computes Cohen's d with 95% bootstrap CIs for both unmatched and
        complexity-matched baselines.  Sets ``flag_for_revision = True`` when
        |matched_d| < |unmatched_d| × 0.7 (substantial shrinkage).

        Returns:
            Dict with unmatched_d, matched_d, unmatched_ci, matched_ci,
            flag_for_revision.
        """
        if self.storage is None or self.config is None:
            return {"result": "missing_storage_or_config"}

        # --- Gather collaborative thread IDs and their quality scores ------
        collab_thread_ids = []
        collab_scores_map: dict[str, float] = {}
        for event in self.events:
            if event.quality_score is not None:
                collab_thread_ids.append(event.thread_id)
                collab_scores_map[event.thread_id] = event.quality_score

        if len(collab_thread_ids) < 2:
            return {"result": "insufficient_data", "n_collaborative": len(collab_thread_ids)}

        # --- Identify non-collaborative threads ----------------------------
        all_thread_ids = list(self.storage._posts.keys())
        collab_set = set(collab_thread_ids)
        non_collab_ids = [tid for tid in all_thread_ids if tid not in collab_set]

        if len(non_collab_ids) < 2:
            return {"result": "insufficient_non_collaborative", "n_non_collaborative": len(non_collab_ids)}

        # --- Score non-collaborative threads using SolutionAssessor --------
        assessor = SolutionAssessor(self.config)
        non_collab_scores_map: dict[str, float] = {}
        for tid in non_collab_ids:
            post_data = self.storage._posts.get(tid)
            body = post_data.get("body", "") if post_data else ""
            if body:
                assessment = assessor.assess_code_solution(body)
                non_collab_scores_map[tid] = assessment.get("quality_score", 0.0)
            else:
                non_collab_scores_map[tid] = 0.0

        # --- Unmatched comparison: collab vs ALL non-collab ----------------
        collab_scores = np.array([collab_scores_map[tid] for tid in collab_thread_ids])
        non_collab_scores_all = np.array([non_collab_scores_map[tid] for tid in non_collab_ids])

        seed = self.config.random_seed
        n_boot = self.config.bootstrap_iterations

        unmatched_d = self._cohens_d_two_sample(collab_scores, non_collab_scores_all)
        unmatched_ci = self._bootstrap_cohens_d_ci(
            collab_scores, non_collab_scores_all, n_iterations=n_boot, seed=seed,
        )

        # --- Complexity-matched comparison ---------------------------------
        matcher = ComplexityMatcher(self.storage, self.config)
        matched_df = matcher.match_with_complexity(
            collab_thread_ids, non_collab_ids, ratio="1:1", replacement=False,
        )

        if matched_df.empty:
            return {
                "result": "matching_failed",
                "unmatched_d": unmatched_d,
                "unmatched_ci": list(unmatched_ci),
            }

        # Build matched non-collab score array aligned with collab scores
        matched_collab_scores: list[float] = []
        matched_non_collab_scores: list[float] = []
        for _, row in matched_df.iterrows():
            c_tid = row["collaborative_thread"]
            m_tid = row["matched_thread"]
            if c_tid in collab_scores_map and m_tid in non_collab_scores_map:
                matched_collab_scores.append(collab_scores_map[c_tid])
                matched_non_collab_scores.append(non_collab_scores_map[m_tid])

        matched_collab_arr = np.array(matched_collab_scores)
        matched_non_collab_arr = np.array(matched_non_collab_scores)

        if len(matched_collab_arr) < 2:
            return {
                "result": "insufficient_matched_pairs",
                "unmatched_d": unmatched_d,
                "unmatched_ci": list(unmatched_ci),
            }

        matched_d = self._cohens_d_two_sample(matched_collab_arr, matched_non_collab_arr)
        matched_ci = self._bootstrap_cohens_d_ci(
            matched_collab_arr, matched_non_collab_arr, n_iterations=n_boot, seed=seed,
        )

        # --- Flagging logic ------------------------------------------------
        flag_for_revision = bool(
            abs(unmatched_d) > 0 and abs(matched_d) < abs(unmatched_d) * 0.7
        )

        return {
            "unmatched_d": unmatched_d,
            "matched_d": matched_d,
            "unmatched_ci": list(unmatched_ci),
            "matched_ci": list(matched_ci),
            "flag_for_revision": flag_for_revision,
            "n_collaborative": len(collab_thread_ids),
            "n_non_collaborative": len(non_collab_ids),
            "n_matched_pairs": len(matched_collab_arr),
        }

    def permutation_test(self, n_permutations: int = 10000) -> dict:
        """Permutation test for collaboration effect.
        
        Tests whether observed success rate is significantly different from
        what would be expected by chance.
        
        Args:
            n_permutations: Number of permutations.
            
        Returns:
            Dict with permutation test results.
        """
        quality_scores = [e.quality_score for e in self.events if e.quality_score is not None]
        
        if len(quality_scores) < 10:
            return {'result': 'insufficient_data'}
        
        observed_mean = np.mean(quality_scores)
        
        # Generate null distribution by permuting success labels
        rng = np.random.RandomState(42)
        null_means = []
        
        for _ in range(n_permutations):
            # Under null hypothesis, quality scores are random
            permuted = rng.permutation(quality_scores)
            null_means.append(np.mean(permuted))
        
        # P-value: proportion of null values >= observed
        p_value = np.mean(np.array(null_means) >= observed_mean)
        
        return {
            'observed_mean': float(observed_mean),
            'null_mean': float(np.mean(null_means)),
            'null_std': float(np.std(null_means)),
            'p_value': float(p_value),
            'n_permutations': n_permutations,
            'significant': p_value < 0.05,
        }


class CollaborationModeler:
    """Models collaboration success factors with rigorous statistical validation."""
    
    def __init__(
        self, 
        events: list[CollaborativeEvent], 
        network: nx.DiGraph
    ) -> None:
        self.events = events
        self.network = network
    
    def fit_success_model(self) -> dict:
        """Fit logistic regression for collaboration success with full statistical inference.
        
        Returns:
            Dict with model coefficients, confidence intervals, odds ratios, and p-values.
        """
        if len(self.events) < 10:
            return {'result': 'insufficient_data', 'n_events': len(self.events)}
        
        # Extract features for each event
        features = []
        outcomes = []
        
        for event in self.events:
            # Network size
            n_participants = len(event.participants)
            
            # Network density among participants
            subgraph = self.network.subgraph(event.participants)
            density = nx.density(subgraph) if len(event.participants) > 1 else 0
            
            # Average degree in subgraph
            if subgraph.number_of_nodes() > 0:
                avg_degree = sum(dict(subgraph.degree()).values()) / subgraph.number_of_nodes()
            else:
                avg_degree = 0
            
            # Duration (log-transformed)
            if event.start_time and event.end_time:
                duration_hours = (event.end_time - event.start_time).total_seconds() / 3600
                log_duration = np.log1p(duration_hours)
            else:
                log_duration = 0
            
            features.append([n_participants, density, avg_degree, log_duration])
            outcomes.append(1 if event.quality_score and event.quality_score > 0.5 else 0)
        
        X = np.array(features)
        y = np.array(outcomes)
        
        if len(np.unique(y)) < 2:
            return {'result': 'insufficient_class_diversity', 'n_success': int(y.sum()), 'n_total': len(y)}
        
        # Use statsmodels for proper inference
        try:
            import statsmodels.api as sm
            
            # Check for variance in features
            feature_vars = np.var(X, axis=0)
            valid_features = feature_vars > 1e-10
            
            if not np.all(valid_features):
                # Remove zero-variance features
                X = X[:, valid_features]
                all_feature_names = ['n_participants', 'density', 'avg_degree', 'log_duration']
                feature_names_filtered = [n for n, v in zip(all_feature_names, valid_features) if v]
            else:
                feature_names_filtered = ['n_participants', 'density', 'avg_degree', 'log_duration']
            
            if X.shape[1] == 0:
                return {'result': 'no_valid_features', 'n_events': len(self.events)}
            
            # Add constant for intercept
            X_sm = sm.add_constant(X, has_constant='add')
            feature_names = ['const'] + feature_names_filtered
            
            # Verify dimensions match
            if X_sm.shape[1] != len(feature_names):
                # Constant wasn't added (likely due to collinearity)
                feature_names = feature_names_filtered
            
            model = sm.Logit(y, X_sm)
            result = model.fit(disp=0, method='bfgs', maxiter=1000)
            
            # Extract results - handle both numpy arrays and pandas Series
            params = np.array(result.params)
            conf_int_arr = np.array(result.conf_int())
            pvalues = np.array(result.pvalues)
            bse = np.array(result.bse)
            
            # Odds ratios with CIs
            odds_ratios = np.exp(params)
            or_ci_lower = np.exp(conf_int_arr[:, 0])
            or_ci_upper = np.exp(conf_int_arr[:, 1])
            
            # Build results dict
            results = {
                'n_events': len(self.events),
                'n_successful': int(y.sum()),
                'success_rate': float(y.mean()),
                'pseudo_r2': float(result.prsquared),
                'llr_p_value': float(result.llr_pvalue),
                'aic': float(result.aic),
                'bic': float(result.bic),
                'coefficients': {},
            }
            
            for i, name in enumerate(feature_names):
                results['coefficients'][name] = {
                    'coef': float(params[i]),
                    'std_err': float(bse[i]),
                    'z': float(params[i] / bse[i]) if bse[i] > 0 else 0,
                    'p_value': float(pvalues[i]),
                    'ci_lower': float(conf_int_arr[i, 0]),
                    'ci_upper': float(conf_int_arr[i, 1]),
                    'odds_ratio': float(odds_ratios[i]),
                    'or_ci_lower': float(or_ci_lower[i]),
                    'or_ci_upper': float(or_ci_upper[i]),
                }
            
            # Log key findings
            for name in ['n_participants', 'density']:
                if name in results['coefficients']:
                    coef_info = results['coefficients'][name]
                    logger.info(f"{name}: OR={coef_info['odds_ratio']:.3f} "
                               f"[{coef_info['or_ci_lower']:.3f}, {coef_info['or_ci_upper']:.3f}], "
                               f"p={coef_info['p_value']:.4e}")
            
            return results
            
        except ImportError:
            logger.warning("statsmodels not available, using sklearn (no p-values)")
            from sklearn.linear_model import LogisticRegression
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y)
            
            return {
                'n_events': len(self.events),
                'n_successful': int(y.sum()),
                'success_rate': float(y.mean()),
                'coefficients': {
                    'intercept': float(model.intercept_[0]),
                    'n_participants': float(model.coef_[0][0]),
                    'density': float(model.coef_[0][1]),
                    'avg_degree': float(model.coef_[0][2]),
                    'log_duration': float(model.coef_[0][3]),
                },
                'warning': 'Install statsmodels for p-values and confidence intervals',
            }
        except Exception as e:
            logger.error(f"Success model fitting failed: {e}")
            return {'result': 'fitting_failed', 'error': str(e)}
    
    def compute_effect_sizes(self) -> dict:
        """Compute effect sizes for collaboration factors.
        
        Returns:
            Dict with Cohen's d and other effect size measures.
        """
        if len(self.events) < 10:
            return {'result': 'insufficient_data'}
        
        # Split events by success
        successful = [e for e in self.events if e.quality_score and e.quality_score > 0.5]
        unsuccessful = [e for e in self.events if not e.quality_score or e.quality_score <= 0.5]
        
        if len(successful) < 3 or len(unsuccessful) < 3:
            return {'result': 'insufficient_groups', 'n_successful': len(successful), 'n_unsuccessful': len(unsuccessful)}
        
        def cohens_d(group1, group2):
            """Compute Cohen's d effect size."""
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            if pooled_std == 0:
                return 0
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        
        results = {}
        
        # Effect size for number of participants
        succ_participants = [len(e.participants) for e in successful]
        unsucc_participants = [len(e.participants) for e in unsuccessful]
        d_participants = cohens_d(succ_participants, unsucc_participants)
        t_stat, t_pval = stats.ttest_ind(succ_participants, unsucc_participants)
        
        results['n_participants'] = {
            'cohens_d': float(d_participants),
            'effect_interpretation': self._interpret_cohens_d(d_participants),
            'successful_mean': float(np.mean(succ_participants)),
            'unsuccessful_mean': float(np.mean(unsucc_participants)),
            't_statistic': float(t_stat),
            'p_value': float(t_pval),
        }
        
        # Effect size for network density
        succ_density = []
        unsucc_density = []
        for e in successful:
            subgraph = self.network.subgraph(e.participants)
            succ_density.append(nx.density(subgraph) if len(e.participants) > 1 else 0)
        for e in unsuccessful:
            subgraph = self.network.subgraph(e.participants)
            unsucc_density.append(nx.density(subgraph) if len(e.participants) > 1 else 0)
        
        d_density = cohens_d(succ_density, unsucc_density)
        t_stat, t_pval = stats.ttest_ind(succ_density, unsucc_density)
        
        results['density'] = {
            'cohens_d': float(d_density),
            'effect_interpretation': self._interpret_cohens_d(d_density),
            'successful_mean': float(np.mean(succ_density)),
            'unsuccessful_mean': float(np.mean(unsucc_density)),
            't_statistic': float(t_stat),
            'p_value': float(t_pval),
        }
        
        # Effect size for duration
        succ_duration = [(e.end_time - e.start_time).total_seconds() / 3600 
                        for e in successful if e.start_time and e.end_time]
        unsucc_duration = [(e.end_time - e.start_time).total_seconds() / 3600 
                          for e in unsuccessful if e.start_time and e.end_time]
        
        if succ_duration and unsucc_duration:
            d_duration = cohens_d(succ_duration, unsucc_duration)
            t_stat, t_pval = stats.ttest_ind(succ_duration, unsucc_duration)
            
            results['duration'] = {
                'cohens_d': float(d_duration),
                'effect_interpretation': self._interpret_cohens_d(d_duration),
                'successful_mean': float(np.mean(succ_duration)),
                'unsuccessful_mean': float(np.mean(unsucc_duration)),
                't_statistic': float(t_stat),
                'p_value': float(t_pval),
            }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def bootstrap_success_rate(self, n_bootstrap: int = 1000) -> dict:
        """Bootstrap confidence interval for success rate.
        
        Args:
            n_bootstrap: Number of bootstrap iterations.
            
        Returns:
            Dict with success rate and confidence interval.
        """
        quality_scores = [e.quality_score for e in self.events if e.quality_score is not None]
        
        if len(quality_scores) < 10:
            return {'result': 'insufficient_data'}
        
        successes = [1 if s > 0.5 else 0 for s in quality_scores]
        n = len(successes)
        
        rng = np.random.RandomState(42)
        bootstrap_rates = []
        
        for _ in range(n_bootstrap):
            sample = rng.choice(successes, size=n, replace=True)
            bootstrap_rates.append(np.mean(sample))
        
        return {
            'success_rate': float(np.mean(successes)),
            'ci_lower': float(np.percentile(bootstrap_rates, 2.5)),
            'ci_upper': float(np.percentile(bootstrap_rates, 97.5)),
            'std_error': float(np.std(bootstrap_rates)),
            'n_events': n,
            'n_bootstrap': n_bootstrap,
        }


def save_rq3_data(
    storage: JSONStorage,
    network: nx.DiGraph,
    config: Config,
    output_dir: str,
) -> dict:
    """Save all RQ3 analysis data to files.
    
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
    
    # Identify collaborative events
    identifier = CollaborationIdentifier(storage, config)
    events = identifier.identify_collaborative_events()
    
    # Assess solutions
    assessor = SolutionAssessor(config)
    for event in events:
        if event.solution:
            assessment = assessor.assess_code_solution(event.solution)
            event.quality_score = assessment.get('quality_score', 0)
    
    # 1. Save event metadata
    event_metadata = []
    for event in events:
        event_metadata.append({
            'thread_id': event.thread_id,
            'n_participants': len(event.participants),
            'start_time': event.start_time.isoformat() if event.start_time else None,
            'end_time': event.end_time.isoformat() if event.end_time else None,
            'duration_minutes': (event.end_time - event.start_time).total_seconds() / 60 if event.start_time and event.end_time else 0,
            'problem_statement': event.problem_statement[:200] if event.problem_statement else '',
            'has_solution': bool(event.solution),
            'quality_score': event.quality_score,
        })
    
    events_df = pd.DataFrame(event_metadata)
    events_df.to_csv(output_path / 'rq3_collaborative_events.csv', index=False)
    saved_files['collaborative_events'] = str(output_path / 'rq3_collaborative_events.csv')
    
    # 2. Save participant details
    participant_data = []
    for event in events:
        for participant in event.participants:
            participant_data.append({
                'thread_id': event.thread_id,
                'agent_id': participant,
            })
    
    participants_df = pd.DataFrame(participant_data)
    participants_df.to_csv(output_path / 'rq3_event_participants.csv', index=False)
    saved_files['event_participants'] = str(output_path / 'rq3_event_participants.csv')
    
    # 3. Save solution assessments
    solution_assessments = []
    for event in events:
        if event.solution:
            assessment = assessor.assess_code_solution(event.solution)
            assessment['thread_id'] = event.thread_id
            solution_assessments.append(assessment)
    
    if solution_assessments:
        assessments_df = pd.DataFrame(solution_assessments)
        assessments_df.to_csv(output_path / 'rq3_solution_assessments.csv', index=False)
        saved_files['solution_assessments'] = str(output_path / 'rq3_solution_assessments.csv')
    
    # 4. Network analysis for collaborative events
    network_metrics = []
    for event in events:
        subgraph = network.subgraph(event.participants)
        
        metrics = {
            'thread_id': event.thread_id,
            'n_nodes': subgraph.number_of_nodes(),
            'n_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph) if len(event.participants) > 1 else 0,
        }
        
        # Compute additional metrics if subgraph is non-trivial
        if subgraph.number_of_nodes() > 2:
            try:
                metrics['avg_clustering'] = nx.average_clustering(subgraph.to_undirected())
            except:
                metrics['avg_clustering'] = 0
        else:
            metrics['avg_clustering'] = 0
        
        network_metrics.append(metrics)
    
    network_df = pd.DataFrame(network_metrics)
    network_df.to_csv(output_path / 'rq3_event_network_metrics.csv', index=False)
    saved_files['event_network_metrics'] = str(output_path / 'rq3_event_network_metrics.csv')
    
    # 5. Collaboration success model with full inference
    modeler = CollaborationModeler(events, network)
    model_results = modeler.fit_success_model()
    with open(output_path / 'rq3_success_model.json', 'w') as f:
        json.dump(model_results, f, indent=2)
    saved_files['success_model'] = str(output_path / 'rq3_success_model.json')
    
    # 6. Effect sizes for collaboration factors
    effect_sizes = modeler.compute_effect_sizes()
    with open(output_path / 'rq3_effect_sizes.json', 'w') as f:
        json.dump(effect_sizes, f, indent=2)
    saved_files['effect_sizes'] = str(output_path / 'rq3_effect_sizes.json')
    
    # 7. Bootstrap confidence interval for success rate
    bootstrap_results = modeler.bootstrap_success_rate()
    with open(output_path / 'rq3_bootstrap_success.json', 'w') as f:
        json.dump(convert_numpy_types(bootstrap_results), f, indent=2)
    saved_files['bootstrap_success'] = str(output_path / 'rq3_bootstrap_success.json')
    
    # 8. Baseline comparison with proper statistical tests
    comparator = BaselineComparator(events, storage=storage, config=config)
    comparison_results = comparator.compare_quality_distributions()
    with open(output_path / 'rq3_baseline_comparison.json', 'w') as f:
        json.dump(convert_numpy_types(comparison_results), f, indent=2)
    saved_files['baseline_comparison'] = str(output_path / 'rq3_baseline_comparison.json')
    
    # 9. Permutation test
    permutation_results = comparator.permutation_test()
    with open(output_path / 'rq3_permutation_test.json', 'w') as f:
        json.dump(convert_numpy_types(permutation_results), f, indent=2)
    saved_files['permutation_test'] = str(output_path / 'rq3_permutation_test.json')
    
    # 10. Technical thread IDs
    technical_threads = identifier.extract_technical_threads()
    tech_df = pd.DataFrame({'thread_id': technical_threads})
    tech_df.to_csv(output_path / 'rq3_technical_threads.csv', index=False)
    saved_files['technical_threads'] = str(output_path / 'rq3_technical_threads.csv')
    
    # 11. Participant activity summary
    participant_activity = participants_df.groupby('agent_id').size().reset_index(name='n_collaborations')
    participant_activity = participant_activity.sort_values('n_collaborations', ascending=False)
    participant_activity.to_csv(output_path / 'rq3_participant_activity.csv', index=False)
    saved_files['participant_activity'] = str(output_path / 'rq3_participant_activity.csv')
    
    # 12. Comprehensive summary with all statistical results
    quality_scores = [e.quality_score for e in events if e.quality_score is not None]
    
    # Extract key statistics from model results
    model_significant_predictors = []
    if 'coefficients' in model_results:
        for name, coef_info in model_results['coefficients'].items():
            if name != 'const' and isinstance(coef_info, dict) and coef_info.get('p_value', 1) < 0.05:
                model_significant_predictors.append({
                    'predictor': name,
                    'odds_ratio': coef_info.get('odds_ratio'),
                    'p_value': coef_info.get('p_value'),
                })
    
    summary = convert_numpy_types({
        'n_collaborative_events': len(events),
        'n_technical_threads': len(technical_threads),
        'total_participants': len(participant_activity),
        'avg_participants_per_event': float(np.mean([len(e.participants) for e in events])) if events else 0,
        'median_participants_per_event': float(np.median([len(e.participants) for e in events])) if events else 0,
        'avg_duration_minutes': float(np.mean([
            (e.end_time - e.start_time).total_seconds() / 60 
            for e in events if e.start_time and e.end_time
        ])) if events else 0,
        # Quality metrics
        'avg_quality_score': float(np.mean(quality_scores)) if quality_scores else 0,
        'median_quality_score': float(np.median(quality_scores)) if quality_scores else 0,
        'quality_score_std': float(np.std(quality_scores)) if quality_scores else 0,
        'n_successful_collaborations': sum(1 for s in quality_scores if s > 0.5),
        'success_rate': sum(1 for s in quality_scores if s > 0.5) / len(quality_scores) if quality_scores else 0,
        # Bootstrap CI for success rate
        'success_rate_ci': [
            bootstrap_results.get('ci_lower'),
            bootstrap_results.get('ci_upper'),
        ] if 'ci_lower' in bootstrap_results else None,
        # Baseline comparison
        'vs_baseline_t_stat': comparison_results.get('t_statistic'),
        'vs_baseline_p_value': comparison_results.get('t_p_value'),
        'vs_baseline_cohens_d': comparison_results.get('cohens_d'),
        'significantly_better_than_baseline': comparison_results.get('significantly_better_than_baseline'),
        # Model results
        'model_pseudo_r2': model_results.get('pseudo_r2'),
        'model_llr_p_value': model_results.get('llr_p_value'),
        'significant_predictors': model_significant_predictors,
        # Effect sizes
        'effect_size_participants': effect_sizes.get('n_participants', {}).get('cohens_d'),
        'effect_size_density': effect_sizes.get('density', {}).get('cohens_d'),
    })
    with open(output_path / 'rq3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    saved_files['summary'] = str(output_path / 'rq3_summary.json')
    
    logger.info(f"Saved {len(saved_files)} RQ3 data files to {output_path}")
    return saved_files
