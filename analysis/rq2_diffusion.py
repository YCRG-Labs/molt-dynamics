#!/usr/bin/env python3
"""RQ2 — Information Diffusion & Cascade Type Stratification Analysis.

This script reproduces the RQ2 cascade analysis from the paper:
"Molt Dynamics: Emergent Social Phenomena in Autonomous AI Agent Populations"

It performs:
  1. Cascade identification and classification (meme vs. skill)
  2. Separate power-law fits per cascade type (Clauset et al. 2009)
  3. Logistic regression adoption models with linear + quadratic exposure
  4. Two-panel Figure 6 data generation (meme and skill size distributions)

Expected Inputs:
  - moltbook-observatory-archive/data/posts/*.parquet
  - moltbook-observatory-archive/data/comments/*.parquet
  - moltbook-observatory-archive/data/agents/*.parquet
  - config/default.yaml  (random seed, pipeline parameters)

Expected Outputs:
  - output/rq2_power_law_by_type.json   — per-type alpha, x_min, KS, LR vs lognormal
  - output/rq2_logistic_by_type.json    — per-type beta1, beta2, p-values, pseudo-R²
  - output/rq2_figure6_data.json        — meme/skill cascade size arrays for plotting

Dependencies:
  - pandas, numpy, scipy
  - powerlaw          (power-law fitting)
  - scikit-learn      (logistic regression)
  - molt_dynamics     (core pipeline: storage, network, cascade detection)

Usage:
  python analysis/rq2_diffusion.py --dataset-path moltbook-observatory-archive
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="RQ2: Cascade type stratification analysis")
    parser.add_argument("--dataset-path", default="moltbook-observatory-archive",
                        help="Path to the Observatory Archive dataset")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for analysis outputs")
    args = parser.parse_args()

    print(f"RQ2 diffusion analysis — dataset: {args.dataset_path}")
    print("TODO: Wire up molt_dynamics.rq2_diffusion pipeline")
    print("See src/molt_dynamics/rq2_diffusion.py for implementation")


if __name__ == "__main__":
    main()
