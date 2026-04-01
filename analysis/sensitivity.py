#!/usr/bin/env python3
"""Sensitivity Analysis — Cox Sample Stability & January 31 Incident Robustness.

This script reproduces the sensitivity analyses from the paper:
"Molt Dynamics: Emergent Social Phenomena in Autonomous AI Agent Populations"

It performs:
  1. Cox model multi-sample stability (5 independent random samples of 100 cascades)
  2. January 31 incident window exclusion (Jan 31 12:00 – Feb 1 12:00 UTC)
  3. Side-by-side comparison of key estimates (full vs. incident-excluded)

Expected Inputs:
  - moltbook-observatory-archive/data/posts/*.parquet
  - moltbook-observatory-archive/data/comments/*.parquet
  - moltbook-observatory-archive/data/agents/*.parquet
  - config/default.yaml  (random seed, pipeline parameters)

Expected Outputs:
  - output/sensitivity_cox_samples.json       — mean HR, SD, range across 5 samples
  - output/sensitivity_jan31_comparison.json   — full vs filtered: alpha, beta1, beta2,
                                                 cox HR, cooperative success rate

Dependencies:
  - pandas, numpy, scipy
  - lifelines       (Cox PH model)
  - molt_dynamics    (core pipeline: storage, diffusion, roles, collaboration modules)

Usage:
  python analysis/sensitivity.py --dataset-path moltbook-observatory-archive
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis: Cox stability + Jan 31 robustness")
    parser.add_argument("--dataset-path", default="moltbook-observatory-archive",
                        help="Path to the Observatory Archive dataset")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for analysis outputs")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of random samples for Cox stability test")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Cascade sample size per Cox model fit")
    args = parser.parse_args()

    print(f"Sensitivity analysis — dataset: {args.dataset_path}")
    print(f"Cox stability: {args.n_samples} samples × {args.sample_size} cascades")
    print("TODO: Wire up molt_dynamics.sensitivity pipeline")
    print("See src/molt_dynamics/sensitivity.py for implementation")


if __name__ == "__main__":
    main()
