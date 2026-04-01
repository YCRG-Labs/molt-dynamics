# Molt Dynamics: Emergent Social Phenomena in Autonomous AI Agent Populations

Code and data pointers for the COLM submission.

#### Brandon Yee <sup>1</sup> Krishna Sharma <sup>2</sup>

<sup>1</sup> Management Sciences Lab, Yee Collins Research Group
<sup>2</sup> Hoover Institute, Stanford University

## Repository Structure

```
├── analysis/                  # Standalone analysis scripts (entry points)
│   ├── rq2_diffusion.py       # RQ2: Cascade type stratification & diffusion
│   ├── rq1_roles.py           # RQ1: Temporal role emergence & survival
│   ├── rq3_cooperation.py     # RQ3: Complexity-matched baseline & cooperation
│   ├── qualitative_events.py  # Qualitative: Crustafarianism trace & vocabulary
│   └── sensitivity.py         # Sensitivity: Cox stability & Jan 31 robustness
├── src/molt_dynamics/         # Core analysis pipeline (Python package)
│   ├── rq1_roles.py           # Role clustering, PCA, Cox PH, Kaplan-Meier
│   ├── rq2_diffusion.py       # Cascade detection, power-law, logistic model
│   ├── rq3_collaboration.py   # Collaboration ID, complexity matching, baselines
│   ├── qualitative_events.py  # Propagation trees, vocabulary emergence
│   ├── sensitivity.py         # Incident robustness, Cox sample sensitivity
│   └── ...                    # Storage, config, models, network, features
├── data/                      # Data directory (see data/README.md)
├── paper/                     # LaTeX manuscript and bibliography
├── config/                    # Pipeline configuration (default.yaml)
├── output/                    # Generated analysis outputs (JSON/CSV)
└── tests/                     # Unit and property-based tests (pytest + hypothesis)
```

## Dataset

**Source**: [SimulaMet/moltbook-observatory-archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)

The dataset includes daily parquet files (Jan–Mar 2026) covering agents, posts,
comments, submolts, snapshots, and word-frequency tables.

```bash
git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Full Pipeline

```bash
python -m molt_dynamics.main --dataset-path moltbook-observatory-archive
```

### 3. Run Individual Analyses

```bash
# Standalone analysis scripts
python analysis/rq2_diffusion.py --dataset-path moltbook-observatory-archive
python analysis/rq1_roles.py     --dataset-path moltbook-observatory-archive
python analysis/rq3_cooperation.py --dataset-path moltbook-observatory-archive
python analysis/qualitative_events.py --dataset-path moltbook-observatory-archive
python analysis/sensitivity.py   --dataset-path moltbook-observatory-archive

# Or via the package modules directly
python -m molt_dynamics.rq1_roles
python -m molt_dynamics.rq2_diffusion
python -m molt_dynamics.rq3_collaboration
```

## Analysis Overview

| Script | Research Question | Key Outputs |
|--------|------------------|-------------|
| `rq2_diffusion.py` | RQ2: Do cascades show saturating contagion? | Power-law fits by type, logistic model coefficients, Figure 6 data |
| `rq1_roles.py` | RQ1: What predicts hub emergence? | Cox hazard ratios, KM curves, PCA scree, UMAP projection |
| `rq3_cooperation.py` | RQ3: Is cooperation costly? | Complexity-matched Cohen's d, bootstrap CIs |
| `qualitative_events.py` | Qualitative: Crustafarianism & vocabulary | Propagation tree, cascade comparison, vocabulary emergence |
| `sensitivity.py` | Robustness checks | Cox multi-sample stability, Jan 31 incident comparison |

## Tests

```bash
pytest tests/ -v
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/) to
validate 21 correctness properties across randomly generated inputs.

## License

See [LICENSE](LICENSE).
