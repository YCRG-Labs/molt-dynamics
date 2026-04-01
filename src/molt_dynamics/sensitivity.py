"""Sensitivity analysis module for Molt Dynamics.

Provides:
- CoxSampleSensitivity: Tests Cox model stability across random samples (P1.1)
- IncidentSensitivity: Compares key estimates with/without Jan 31 incident (P1.4)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


class CoxSampleSensitivity:
    """Tests Cox model stability across random samples.

    Fits a Cox proportional-hazards model on multiple independent random
    sub-samples of the survival data and reports summary statistics on
    the resulting hazard ratios.
    """

    # The seed used for the published primary sample (seed index 0)
    PRIMARY_SAMPLE_SEED_OFFSET = 0

    def __init__(self, survival_data: pd.DataFrame, config: Config) -> None:
        """Initialise with pre-computed survival data.

        Args:
            survival_data: DataFrame produced by
                ``TemporalRoleAnalyzer.compute_covariates()`` containing at
                minimum the columns ``time_to_emergence_days``, ``emerged``,
                ``join_cohort``, ``initial_posting_cadence``,
                ``submolt_diversity_day3``, and ``early_reply``.
            config: Configuration (provides ``random_seed``, ``output_dir``).
        """
        self.survival_data = survival_data
        self.config = config

    def run_multi_sample(
        self,
        n_samples: int = 5,
        sample_size: int = 100,
    ) -> dict:
        """Fit Cox model on *n_samples* independent random sub-samples.

        For each sample a different random seed is used to draw
        *sample_size* rows (without replacement when possible) from
        ``self.survival_data``, then a Cox PH model is fitted.

        Returns:
            Dict with ``mean_hr``, ``sd_hr``, ``range_hr`` (each keyed
            by covariate name), ``per_sample_details`` (list of per-seed
            results), and ``primary_sample_seed``.  Also writes
            ``output/sensitivity_cox_samples.json``.
        """
        from lifelines import CoxPHFitter

        if self.survival_data.empty:
            logger.warning("No survival data for multi-sample sensitivity")
            return {}

        base_seed = self.config.random_seed
        per_sample_details: list[dict] = []

        for i in range(n_samples):
            seed = base_seed + i
            rng = np.random.RandomState(seed)

            n_available = len(self.survival_data)
            actual_size = min(sample_size, n_available)
            replace = actual_size > n_available
            indices = rng.choice(n_available, size=actual_size, replace=replace)
            sample_df = self.survival_data.iloc[indices].copy()

            try:
                hr_dict, ci_dict = self._fit_cox_on_sample(sample_df)
            except Exception as exc:
                logger.warning("Cox fit failed for seed %d: %s", seed, exc)
                per_sample_details.append({"seed": seed, "error": str(exc)})
                continue

            per_sample_details.append({
                "seed": seed,
                "sample_size": actual_size,
                "hazard_ratios": hr_dict,
                "confidence_intervals": ci_dict,
            })

        # Aggregate across successful fits
        successful = [d for d in per_sample_details if "hazard_ratios" in d]

        if not successful:
            logger.warning("All Cox fits failed in multi-sample sensitivity")
            return {"per_sample_details": per_sample_details, "error": "all_fits_failed"}

        # Collect HR values per covariate
        all_covariates = list(successful[0]["hazard_ratios"].keys())
        hr_arrays: dict[str, list[float]] = {c: [] for c in all_covariates}
        for detail in successful:
            for c in all_covariates:
                if c in detail["hazard_ratios"]:
                    hr_arrays[c].append(detail["hazard_ratios"][c])

        mean_hr: dict[str, float] = {}
        sd_hr: dict[str, float] = {}
        range_hr: dict[str, list[float]] = {}

        for c in all_covariates:
            vals = np.array(hr_arrays[c])
            mean_hr[c] = float(np.mean(vals))
            sd_hr[c] = float(np.std(vals, ddof=0))
            range_hr[c] = [float(np.min(vals)), float(np.max(vals))]

        primary_seed = base_seed + self.PRIMARY_SAMPLE_SEED_OFFSET

        result = {
            "mean_hr": mean_hr,
            "sd_hr": sd_hr,
            "range_hr": range_hr,
            "per_sample_details": per_sample_details,
            "primary_sample_seed": primary_seed,
            "n_samples": n_samples,
            "sample_size": sample_size,
        }

        # Write output
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "sensitivity_cox_samples.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info(
            "Wrote Cox sample sensitivity to %s",
            out_path / "sensitivity_cox_samples.json",
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_cox_on_sample(sample_df: pd.DataFrame) -> tuple[dict, dict]:
        """Fit a Cox PH model on a single sub-sample.

        Returns:
            Tuple of (hazard_ratios dict, confidence_intervals dict).
        """
        from lifelines import CoxPHFitter

        model_df = sample_df[
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
            model_df["join_cohort"], prefix="cohort", dtype=float,
        )
        ref_col = "cohort_day1-3"
        dummy_cols = [c for c in cohort_dummies.columns if c != ref_col]
        model_df = pd.concat([model_df, cohort_dummies[dummy_cols]], axis=1)
        model_df.drop(columns=["join_cohort"], inplace=True)

        model_df["early_reply"] = model_df["early_reply"].astype(float)
        model_df["time_to_emergence_days"] = model_df[
            "time_to_emergence_days"
        ].clip(lower=0.5)

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(
            model_df,
            duration_col="time_to_emergence_days",
            event_col="emerged",
        )

        summary = cph.summary
        hazard_ratios = {
            covar: float(summary.loc[covar, "exp(coef)"])
            for covar in summary.index
        }
        confidence_intervals = {
            covar: {
                "lower": float(summary.loc[covar, "exp(coef) lower 95%"]),
                "upper": float(summary.loc[covar, "exp(coef) upper 95%"]),
            }
            for covar in summary.index
        }
        return hazard_ratios, confidence_intervals


class IncidentSensitivity:
    """Compares key estimates with/without the Jan 31 incident window.

    Filters out posts and comments whose ``created_at`` falls within
    [incident_start, incident_end] and re-runs the five key metrics on
    both the full and filtered datasets, producing a side-by-side
    comparison table written to ``output/sensitivity_jan31_comparison.json``.
    """

    METRIC_NAMES = [
        "power_law_alpha",
        "logistic_beta1",
        "logistic_beta2",
        "cox_hazard_ratio",
        "cooperative_success_rate",
    ]

    def __init__(
        self,
        storage: "JSONStorage",
        network: "NetworkBuilder",
        config: Config,
    ) -> None:
        from datetime import datetime as _dt

        self.storage = storage
        self.network = network
        self.config = config
        self.incident_start = _dt(2026, 1, 31, 12, 0)
        self.incident_end = _dt(2026, 2, 1, 12, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_incident_window(self) -> "JSONStorage":
        """Return a filtered copy of storage excluding incident-window records.

        Posts and comments whose ``created_at`` falls within
        ``[incident_start, incident_end]`` (inclusive) are removed.  All
        other data (agents, interactions, submolts, memberships) is
        copied as-is so the returned object has the same interface as
        :class:`JSONStorage`.
        """
        from .storage import JSONStorage, parse_datetime

        filtered = JSONStorage.__new__(JSONStorage)
        filtered.config = self.config
        filtered.data_dir = self.storage.data_dir

        # Copy agents, submolts, memberships verbatim
        filtered._agents = dict(self.storage._agents)
        filtered._submolts = dict(self.storage._submolts)
        filtered._memberships = dict(self.storage._memberships)

        # Filter posts
        filtered._posts = {}
        for post_id, data in self.storage._posts.items():
            ts = parse_datetime(data.get("created_at"))
            if ts is not None and self.incident_start <= ts <= self.incident_end:
                continue
            filtered._posts[post_id] = data

        # Filter comments
        filtered._comments = {}
        for comment_id, data in self.storage._comments.items():
            ts = parse_datetime(data.get("created_at"))
            if ts is not None and self.incident_start <= ts <= self.incident_end:
                continue
            filtered._comments[comment_id] = data

        # Filter interactions by timestamp
        filtered._interactions = []
        for interaction in self.storage._interactions:
            ts = parse_datetime(interaction.get("timestamp"))
            if ts is not None and self.incident_start <= ts <= self.incident_end:
                continue
            filtered._interactions.append(interaction)

        # Rebuild author indexes
        filtered._posts_by_author = {}
        filtered._comments_by_author = {}
        filtered._build_author_indexes()

        return filtered

    def run_comparison(self) -> dict:
        """Run key analyses on both full and filtered data.

        Computes five metrics on each sample:

        * ``power_law_alpha`` – power-law exponent from cascade sizes
        * ``logistic_beta1`` – linear exposure coefficient
        * ``logistic_beta2`` – quadratic exposure coefficient
        * ``cox_hazard_ratio`` – Cox PH hazard ratio for exposure
        * ``cooperative_success_rate`` – fraction of collaborative events
          with quality > 0.5

        Returns:
            Dict with ``full`` and ``filtered`` sub-dicts, each containing
            the five metrics, plus metadata.  Also writes the result to
            ``output/sensitivity_jan31_comparison.json``.
        """
        filtered_storage = self.filter_incident_window()

        full_metrics = self._compute_metrics(self.storage, label="full")
        filtered_metrics = self._compute_metrics(filtered_storage, label="filtered")

        result = {
            "full": full_metrics,
            "filtered": filtered_metrics,
            "incident_window": {
                "start": self.incident_start.isoformat(),
                "end": self.incident_end.isoformat(),
            },
            "posts_removed": len(self.storage._posts) - len(filtered_storage._posts),
            "comments_removed": len(self.storage._comments) - len(filtered_storage._comments),
        }

        # Write output
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "sensitivity_jan31_comparison.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info(
            "Wrote Jan 31 sensitivity comparison to %s",
            out_path / "sensitivity_jan31_comparison.json",
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, storage: "JSONStorage", label: str = "") -> dict:
        """Compute the five key metrics on the given *storage*.

        Each metric is computed independently; if a particular analysis
        fails (e.g. insufficient data), the metric is reported as
        ``None`` with an accompanying error note.
        """
        metrics: dict = {}

        # 1. power_law_alpha — from cascade sizes
        try:
            metrics["power_law_alpha"] = self._compute_power_law_alpha(storage)
        except Exception as exc:
            logger.warning("power_law_alpha failed (%s): %s", label, exc)
            metrics["power_law_alpha"] = None

        # 2 & 3. logistic_beta1, logistic_beta2
        try:
            b1, b2 = self._compute_logistic_betas(storage)
            metrics["logistic_beta1"] = b1
            metrics["logistic_beta2"] = b2
        except Exception as exc:
            logger.warning("logistic betas failed (%s): %s", label, exc)
            metrics["logistic_beta1"] = None
            metrics["logistic_beta2"] = None

        # 4. cox_hazard_ratio
        try:
            metrics["cox_hazard_ratio"] = self._compute_cox_hazard_ratio(storage)
        except Exception as exc:
            logger.warning("cox_hazard_ratio failed (%s): %s", label, exc)
            metrics["cox_hazard_ratio"] = None

        # 5. cooperative_success_rate
        try:
            metrics["cooperative_success_rate"] = self._compute_cooperative_success_rate(storage)
        except Exception as exc:
            logger.warning("cooperative_success_rate failed (%s): %s", label, exc)
            metrics["cooperative_success_rate"] = None

        return metrics

    # -- Individual metric helpers ------------------------------------

    def _compute_power_law_alpha(self, storage: "JSONStorage") -> float | None:
        """Compute power-law alpha from cascade sizes in *storage*."""
        from .rq2_diffusion import CascadeIdentifier, CascadeAnalyzer

        identifier = CascadeIdentifier(storage, self.config)
        cascades = identifier.identify_meme_cascades()
        cascades += identifier.identify_skill_cascades()

        if len(cascades) < 10:
            return None

        analyzer = CascadeAnalyzer(cascades)
        result = analyzer.test_power_law()
        return result.get("alpha")

    def _compute_logistic_betas(
        self, storage: "JSONStorage"
    ) -> tuple[float | None, float | None]:
        """Compute logistic beta1 (linear) and beta2 (quadratic)."""
        from .rq2_diffusion import CascadeIdentifier, DiffusionModeler
        from .network import NetworkBuilder

        identifier = CascadeIdentifier(storage, self.config)
        cascades = identifier.identify_meme_cascades()
        cascades += identifier.identify_skill_cascades()

        if len(cascades) < 2:
            return None, None

        network = NetworkBuilder(storage)
        G = network.build_interaction_network()
        modeler = DiffusionModeler(cascades, G, self.config)
        result = modeler.fit_logistic_model()

        return result.get("beta_linear"), result.get("beta_quadratic")

    def _compute_cox_hazard_ratio(self, storage: "JSONStorage") -> float | None:
        """Compute Cox PH hazard ratio for exposure from cascade data."""
        from .rq2_diffusion import CascadeIdentifier, DiffusionModeler
        from .network import NetworkBuilder

        identifier = CascadeIdentifier(storage, self.config)
        cascades = identifier.identify_meme_cascades()
        cascades += identifier.identify_skill_cascades()

        if len(cascades) < 2:
            return None

        network = NetworkBuilder(storage)
        G = network.build_interaction_network()
        modeler = DiffusionModeler(cascades, G, self.config)
        result = modeler.fit_cox_hazards(max_cascades=100)

        return result.get("hazard_ratio")

    def _compute_cooperative_success_rate(self, storage: "JSONStorage") -> float | None:
        """Compute fraction of collaborative events with quality > 0.5."""
        from .rq3_collaboration import CollaborationIdentifier, SolutionAssessor

        identifier = CollaborationIdentifier(storage, self.config)
        events = identifier.identify_collaborative_events()

        if not events:
            return None

        assessor = SolutionAssessor(self.config)
        scored = 0
        successes = 0
        for event in events:
            if event.solution:
                result = assessor.assess_code_solution(event.solution)
                score = result.get("quality_score", 0.0)
                scored += 1
                if score > 0.5:
                    successes += 1

        if scored == 0:
            return None

        return float(successes / scored)
