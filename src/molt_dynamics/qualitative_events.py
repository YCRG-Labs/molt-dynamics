"""Qualitative cascade analysis module (P0.4).

Provides CascadeTracer for reconstructing propagation trees of individual
cascades and comparing them against population-level baselines.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import pandas as pd
import re

from .models import Cascade, PropagationNode, VocabularyTerm
from .storage import JSONStorage

logger = logging.getLogger(__name__)


class CascadeTracer:
    """Reconstructs propagation trees for individual cascades.

    Parameters
    ----------
    storage : JSONStorage
        Storage backend for post/comment data.
    cascades : list[Cascade]
        Pre-identified cascades (typically from ``CascadeIdentifier``).
    network_builder : object, optional
        A ``NetworkBuilder`` instance used to infer propagation paths.
    role_assignments : dict[str, str], optional
        Mapping of agent_id → role cluster name (from RQ1 clustering).
    """

    def __init__(
        self,
        storage: JSONStorage,
        cascades: list[Cascade],
        network_builder=None,
        role_assignments: Optional[dict[str, str]] = None,
    ) -> None:
        self.storage = storage
        self.cascades = cascades
        self.network_builder = network_builder
        self.role_assignments = role_assignments or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_top_cascades(self, n: int = 10) -> list[Cascade]:
        """Return the *n* largest meme cascades by unique adoption count.

        Only cascades with ``cascade_type == 'meme'`` are considered.
        """
        meme_cascades = [c for c in self.cascades if c.cascade_type == "meme"]
        meme_cascades.sort(
            key=lambda c: len(set(a[0] for a in c.adoptions)), reverse=True
        )
        return meme_cascades[:n]

    def find_cascade_by_content(self, keyword: str) -> Optional[Cascade]:
        """Search cascades for content matching *keyword* (case-insensitive).

        Checks ``content_hash`` and ``cascade_id`` fields.  Returns the first
        match or ``None``.
        """
        kw = keyword.lower()
        for cascade in self.cascades:
            if kw in cascade.content_hash.lower() or kw in cascade.cascade_id.lower():
                return cascade
        return None

    def reconstruct_propagation_tree(self, cascade: Cascade) -> dict:
        """Build a propagation tree for *cascade*.

        The tree is rooted at the seed agent.  For each adopter the parent is
        chosen as the network neighbour that adopted earliest before them.  If
        no network neighbour adopted earlier the seed is used as parent.

        Returns a dict with keys:
        - ``nodes``: list of ``PropagationNode`` dicts
        - ``depth``: maximum tree depth
        - ``breadth``: maximum number of nodes at any single depth level
        - ``time_to_peak``: hours from seed to the time-step with the most
          adoptions
        """
        # Deduplicate adoptions – keep first occurrence per agent
        seen: dict[str, datetime] = {}
        for agent_id, ts in cascade.adoptions:
            if agent_id not in seen:
                seen[agent_id] = ts

        # Build the network graph (if available) for parent inference
        network = None
        if self.network_builder is not None:
            try:
                network = self.network_builder.build_interaction_network()
            except Exception:
                logger.warning("Could not build network; falling back to seed-parent")

        # Seed node
        seed_id = cascade.seed_agent
        seed_time = cascade.seed_time
        nodes: list[PropagationNode] = [
            PropagationNode(
                agent_id=seed_id,
                adoption_time=seed_time,
                depth=0,
                parent_agent_id=None,
                role_cluster=self.role_assignments.get(seed_id),
            )
        ]

        # Adopters sorted by time
        adopters = sorted(
            ((aid, ts) for aid, ts in seen.items() if aid != seed_id),
            key=lambda x: x[1],
        )

        adopted_times: dict[str, datetime] = {seed_id: seed_time}
        depth_map: dict[str, int] = {seed_id: 0}

        for agent_id, adoption_time in adopters:
            parent_id = self._find_parent(
                agent_id, adoption_time, adopted_times, depth_map, network
            )
            depth = depth_map.get(parent_id, 0) + 1
            depth_map[agent_id] = depth
            adopted_times[agent_id] = adoption_time
            nodes.append(
                PropagationNode(
                    agent_id=agent_id,
                    adoption_time=adoption_time,
                    depth=depth,
                    parent_agent_id=parent_id,
                    role_cluster=self.role_assignments.get(agent_id),
                )
            )

        # Metrics
        max_depth = max(n.depth for n in nodes) if nodes else 0
        depth_counts: dict[int, int] = defaultdict(int)
        for n in nodes:
            depth_counts[n.depth] += 1
        breadth = max(depth_counts.values()) if depth_counts else 0

        time_to_peak = self._compute_time_to_peak(cascade)

        tree = {
            "cascade_id": cascade.cascade_id,
            "seed_agent": seed_id,
            "seed_time": seed_time.isoformat(),
            "nodes": [
                {
                    "agent_id": n.agent_id,
                    "adoption_time": n.adoption_time.isoformat(),
                    "depth": n.depth,
                    "parent_agent_id": n.parent_agent_id,
                    "role_cluster": n.role_cluster,
                }
                for n in nodes
            ],
            "depth": max_depth,
            "breadth": breadth,
            "time_to_peak_hours": time_to_peak,
        }
        return tree

    def compute_adopter_role_distribution(self, cascade: Cascade) -> dict:
        """Map each adopter to their RQ1 role cluster and return fractions.

        Returns a dict with:
        - ``counts``: ``{role_name: int}``
        - ``fractions``: ``{role_name: float}`` summing to 1.0
        """
        unique_adopters: set[str] = set()
        for agent_id, _ in cascade.adoptions:
            unique_adopters.add(agent_id)
        # Include seed
        unique_adopters.add(cascade.seed_agent)

        counts: dict[str, int] = defaultdict(int)
        for agent_id in unique_adopters:
            role = self.role_assignments.get(agent_id, "Unknown")
            counts[role] += 1

        total = sum(counts.values())
        fractions = {role: count / total for role, count in counts.items()} if total > 0 else {}

        return {"counts": dict(counts), "fractions": fractions}

    def compare_to_median_cascade(self, cascade: Cascade) -> dict:
        """Compare *cascade* against the median meme cascade of similar size.

        "Similar size" is defined as cascades whose unique adopter count is
        within ±50 % of the target cascade's count (minimum bucket of 3).

        Returns comparative metrics:
        - ``speed_ratio``: target duration / median duration
        - ``depth_ratio``: target depth / median depth
        - ``role_distribution_divergence``: symmetric KL divergence between
          the target's role distribution and the median bucket's distribution
        """
        target_size = len(set(a[0] for a in cascade.adoptions))
        lo = max(1, int(target_size * 0.5))
        hi = int(target_size * 1.5)

        similar: list[Cascade] = []
        for c in self.cascades:
            if c.cascade_type != "meme" or c.cascade_id == cascade.cascade_id:
                continue
            c_size = len(set(a[0] for a in c.adoptions))
            if lo <= c_size <= hi:
                similar.append(c)

        if len(similar) < 1:
            return {
                "speed_ratio": None,
                "depth_ratio": None,
                "role_distribution_divergence": None,
                "n_similar": 0,
            }

        # Duration (hours)
        target_duration = self._cascade_duration_hours(cascade)
        median_duration = float(np.median([self._cascade_duration_hours(c) for c in similar]))

        # Depth via tree reconstruction
        target_tree = self.reconstruct_propagation_tree(cascade)
        target_depth = target_tree["depth"]

        similar_depths = []
        for c in similar:
            t = self.reconstruct_propagation_tree(c)
            similar_depths.append(t["depth"])
        median_depth = float(np.median(similar_depths))

        speed_ratio = (target_duration / median_duration) if median_duration > 0 else None
        depth_ratio = (target_depth / median_depth) if median_depth > 0 else None

        # Role distribution divergence (symmetric KL)
        target_dist = self.compute_adopter_role_distribution(cascade)["fractions"]
        bucket_counts: dict[str, int] = defaultdict(int)
        bucket_total = 0
        for c in similar:
            dist = self.compute_adopter_role_distribution(c)["counts"]
            for role, cnt in dist.items():
                bucket_counts[role] += cnt
                bucket_total += cnt
        bucket_fracs = {r: cnt / bucket_total for r, cnt in bucket_counts.items()} if bucket_total else {}

        divergence = self._symmetric_kl(target_dist, bucket_fracs)

        result = {
            "speed_ratio": speed_ratio,
            "depth_ratio": depth_ratio,
            "role_distribution_divergence": divergence,
            "n_similar": len(similar),
            "target_duration_hours": target_duration,
            "median_duration_hours": median_duration,
            "target_depth": target_depth,
            "median_depth": median_depth,
        }
        return result

    def write_outputs(self, cascade: Cascade) -> None:
        """Write qualitative analysis outputs for *cascade*.

        Produces:
        - ``output/qualitative_crustafarianism_tree.json``
        - ``output/qualitative_cascade_comparison.json``
        """
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        tree = self.reconstruct_propagation_tree(cascade)
        with open(output_dir / "qualitative_crustafarianism_tree.json", "w") as f:
            json.dump(tree, f, indent=2)
        logger.info("Wrote output/qualitative_crustafarianism_tree.json")

        comparison = self.compare_to_median_cascade(cascade)
        with open(output_dir / "qualitative_cascade_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Wrote output/qualitative_cascade_comparison.json")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_parent(
        self,
        agent_id: str,
        adoption_time: datetime,
        adopted_times: dict[str, datetime],
        depth_map: dict[str, int],
        network,
    ) -> str:
        """Find the most likely parent for *agent_id* in the propagation tree.

        Prefers the network neighbour that adopted most recently before
        *adoption_time*.  Falls back to the globally most-recent adopter,
        then to the seed.
        """
        if network is not None:
            predecessors = set()
            try:
                predecessors = set(network.predecessors(agent_id))
            except Exception:
                pass
            try:
                predecessors |= set(network.successors(agent_id))
            except Exception:
                pass

            best_parent: Optional[str] = None
            best_time: Optional[datetime] = None
            for pred in predecessors:
                pred_time = adopted_times.get(pred)
                if pred_time is not None and pred_time < adoption_time:
                    if best_time is None or pred_time > best_time:
                        best_parent = pred
                        best_time = pred_time
            if best_parent is not None:
                return best_parent

        # Fallback: most recent adopter before this agent
        best_parent = None
        best_time = None
        for aid, t in adopted_times.items():
            if t < adoption_time:
                if best_time is None or t > best_time:
                    best_parent = aid
                    best_time = t

        if best_parent is not None:
            return best_parent

        # Ultimate fallback: seed (first key in adopted_times)
        return next(iter(adopted_times))

    @staticmethod
    def _cascade_duration_hours(cascade: Cascade) -> float:
        """Return cascade duration in hours."""
        if not cascade.adoptions:
            return 0.0
        timestamps = [ts for _, ts in cascade.adoptions]
        return (max(timestamps) - min(timestamps)).total_seconds() / 3600.0

    @staticmethod
    def _compute_time_to_peak(cascade: Cascade) -> float:
        """Hours from seed time to the hour-bucket with the most adoptions."""
        if not cascade.adoptions:
            return 0.0
        seed_time = cascade.seed_time
        # Bucket adoptions by hour offset
        hour_counts: dict[int, int] = defaultdict(int)
        for _, ts in cascade.adoptions:
            hour_offset = int((ts - seed_time).total_seconds() / 3600)
            hour_counts[hour_offset] += 1
        if not hour_counts:
            return 0.0
        peak_hour = max(hour_counts, key=hour_counts.get)  # type: ignore[arg-type]
        return max(0.0, float(peak_hour))

    @staticmethod
    def _symmetric_kl(p: dict[str, float], q: dict[str, float]) -> float:
        """Compute symmetric KL divergence between two role distributions.

        Uses a small epsilon to avoid log(0).
        """
        all_keys = set(p) | set(q)
        if not all_keys:
            return 0.0
        eps = 1e-10
        kl_pq = 0.0
        kl_qp = 0.0
        for k in all_keys:
            pk = p.get(k, 0.0) + eps
            qk = q.get(k, 0.0) + eps
            kl_pq += pk * np.log(pk / qk)
            kl_qp += qk * np.log(qk / pk)
        return float((kl_pq + kl_qp) / 2.0)


class VocabularyEmergenceAnalyzer:
    """Tracks novel terminology emergence from specific threads.

    Extracts unigrams and bigrams from post/comment bodies, identifies
    terms that appear for the first time in a given thread, and tracks
    their subsequent spread to other threads.

    Parameters
    ----------
    storage : JSONStorage
        Storage backend for post/comment data.
    """

    # Simple tokenisation pattern: sequences of word characters
    _TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_]{1,}\b")

    def __init__(self, storage: JSONStorage) -> None:
        self.storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_thread_vocabulary(self, post_ids: list[str]) -> set[str]:
        """Extract unique unigrams and bigrams from a set of posts/comments.

        For each post id the method collects the post body **and** all
        comments on that post, tokenises them, and returns the union of
        all unigrams and bigrams.

        Parameters
        ----------
        post_ids : list[str]
            Post identifiers whose content (and comments) should be
            included.

        Returns
        -------
        set[str]
            Unique n-grams (unigrams + bigrams).
        """
        vocab: set[str] = set()
        for pid in post_ids:
            bodies = self._collect_bodies_for_post(pid)
            for body in bodies:
                tokens = self._tokenise(body)
                # Unigrams
                vocab.update(tokens)
                # Bigrams
                for i in range(len(tokens) - 1):
                    vocab.add(f"{tokens[i]} {tokens[i + 1]}")
        return vocab

    def find_novel_terms(
        self, thread_vocab: set[str], pre_thread_corpus: set[str]
    ) -> list[str]:
        """Identify terms that appear for the first time in the thread.

        Returns the set difference ``thread_vocab - pre_thread_corpus``
        as a sorted list.

        Parameters
        ----------
        thread_vocab : set[str]
            Vocabulary extracted from the target thread.
        pre_thread_corpus : set[str]
            Vocabulary from all content that predates the thread.

        Returns
        -------
        list[str]
            Novel terms sorted alphabetically.
        """
        return sorted(thread_vocab - pre_thread_corpus)

    def track_term_spread(self, terms: list[str]) -> pd.DataFrame:
        """Track subsequent appearances of novel terms across threads.

        For each term, scans all posts in storage to find the earliest
        appearance (``first_appearance``), then finds all *other* threads
        (post_ids) where the term appears.  Computes ``adoption_count``
        and ``time_to_first_spread_hours``.

        Parameters
        ----------
        terms : list[str]
            Novel terms to track.

        Returns
        -------
        pd.DataFrame
            Columns: term, source_thread_id, first_appearance,
            adoption_count, adopting_thread_ids,
            time_to_first_spread_hours.
        """
        if not terms:
            return pd.DataFrame(columns=[
                "term", "source_thread_id", "first_appearance",
                "adoption_count", "adopting_thread_ids",
                "time_to_first_spread_hours",
            ])

        all_posts = self.storage.get_posts()
        all_comments = self.storage.get_comments()

        # Build a mapping: post_id → combined text (post body + comment bodies)
        post_texts: dict[str, str] = {}
        post_times: dict[str, datetime] = {}
        for p in all_posts:
            post_texts[p.post_id] = (p.body or "").lower()
            if p.created_at:
                post_times[p.post_id] = p.created_at

        # Append comment bodies to their parent post's text
        for c in all_comments:
            if c.post_id in post_texts:
                post_texts[c.post_id] += " " + (c.body or "").lower()
            if c.post_id not in post_times and c.created_at:
                post_times[c.post_id] = c.created_at

        rows: list[dict] = []
        for term in terms:
            term_lower = term.lower()
            # Find all posts containing this term
            containing_posts: list[tuple[str, datetime]] = []
            for pid, text in post_texts.items():
                if term_lower in text and pid in post_times:
                    containing_posts.append((pid, post_times[pid]))

            if not containing_posts:
                rows.append({
                    "term": term,
                    "source_thread_id": "",
                    "first_appearance": None,
                    "adoption_count": 0,
                    "adopting_thread_ids": [],
                    "time_to_first_spread_hours": None,
                })
                continue

            # Sort by time to find first appearance
            containing_posts.sort(key=lambda x: x[1])
            source_pid, first_time = containing_posts[0]

            # Adopting threads = all other threads containing the term
            adopting = [
                pid for pid, _ in containing_posts if pid != source_pid
            ]
            adoption_count = len(adopting)

            time_to_first_spread: Optional[float] = None
            if adoption_count > 0:
                first_spread_time = min(
                    t for pid, t in containing_posts if pid != source_pid
                )
                time_to_first_spread = (
                    (first_spread_time - first_time).total_seconds() / 3600.0
                )

            rows.append({
                "term": term,
                "source_thread_id": source_pid,
                "first_appearance": first_time,
                "adoption_count": adoption_count,
                "adopting_thread_ids": adopting,
                "time_to_first_spread_hours": time_to_first_spread,
            })

        return pd.DataFrame(rows)

    def find_most_replied_posts(self, submolt: str, n: int = 10) -> list[str]:
        """Find the *n* posts with the most replies in a submolt.

        A "reply" is any comment whose ``post_id`` matches the post.

        Parameters
        ----------
        submolt : str
            Submolt name to filter posts by.
        n : int
            Number of top posts to return.

        Returns
        -------
        list[str]
            Post IDs ordered by reply count descending.
        """
        posts = self.storage.get_posts(filters={"submolt": submolt})
        post_ids = {p.post_id for p in posts}

        if not post_ids:
            return []

        all_comments = self.storage.get_comments()
        reply_counts: dict[str, int] = defaultdict(int)
        for c in all_comments:
            if c.post_id in post_ids:
                reply_counts[c.post_id] += 1

        sorted_posts = sorted(
            reply_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [pid for pid, _ in sorted_posts[:n]]

    def write_outputs(self, terms_df: pd.DataFrame) -> None:
        """Write vocabulary emergence results to CSV.

        Parameters
        ----------
        terms_df : pd.DataFrame
            DataFrame produced by :meth:`track_term_spread`.
        """
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / "qualitative_vocabulary_emergence.csv"
        # Convert list columns to string for CSV serialisation
        df = terms_df.copy()
        if "adopting_thread_ids" in df.columns:
            df["adopting_thread_ids"] = df["adopting_thread_ids"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {out_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_bodies_for_post(self, post_id: str) -> list[str]:
        """Return the post body and all comment bodies for *post_id*."""
        bodies: list[str] = []

        posts = self.storage.get_posts()
        for p in posts:
            if p.post_id == post_id and p.body:
                bodies.append(p.body)

        comments = self.storage.get_comments(filters={"post_id": post_id})
        for c in comments:
            if c.body:
                bodies.append(c.body)

        return bodies

    def _tokenise(self, text: str) -> list[str]:
        """Lowercase and extract word tokens from *text*."""
        return self._TOKEN_RE.findall(text.lower())
