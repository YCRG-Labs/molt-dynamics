Code Associated With:

## Benchmarking Emergent Coordination in Large-Scale LLM Populations: An Evaluation Framework on the MoltBook Archive

## Dataset

**Source**: [SimulaMet/moltbook-observatory-archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)

The dataset includes:
- **Agents**: AI agent profiles with karma, follower counts, and metadata
- **Posts**: Posts created by agents with scores and comment counts
- **Comments**: Post comments with scores and parent relationships
- **Submolts**: Community metadata and subscriber statistics
- **Snapshots**: Periodic global observatory metrics
- **Word Frequency**: Hourly word frequency statistics

## Quick Start

### 1. Clone the Dataset

```bash
git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the Package

```bash
pip install -e .
```

This installs the `molt_dynamics` package in development mode.

### 4. Load the Data

```bash
# Load all data
python scripts/load_dataset.py --dataset-path moltbook-observatory-archive

# Load with limits
python scripts/load_dataset.py --dataset-path moltbook-observatory-archive --max-posts 10000 --max-comments 5000
```

The data will be loaded into `output/data/` as JSON files for analysis.

### 4. Run Analysis

```bash
# Run complete analysis pipeline (loads data + analyzes)
python -m molt_dynamics.main --dataset-path moltbook-observatory-archive

# Or load data first, then analyze
python scripts/load_dataset.py --dataset-path moltbook-observatory-archive
python -m molt_dynamics.main --skip-loading

# Run specific research questions
python -m molt_dynamics.rq1_roles      # Agent roles
python -m molt_dynamics.rq2_diffusion  # Information diffusion
python -m molt_dynamics.rq3_collaboration  # Collaboration
python -m molt_dynamics.rq4_phase      # Phase transitions
```
