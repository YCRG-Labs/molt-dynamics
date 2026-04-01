# Data

## MoltBook Observatory Archive

The analysis uses the MoltBook Observatory Archive, a daily parquet dataset
spanning late January to late March 2026 containing agent, post, comment,
submolt, snapshot, and word-frequency tables.

### Source

**HuggingFace**: [SimulaMet/moltbook-observatory-archive](https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive)

### Download

```bash
git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
```

Or use the HuggingFace `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("SimulaMet/moltbook-observatory-archive", "agents")
```

### Structure

```
moltbook-observatory-archive/data/
├── agents/       # Daily agent profile snapshots (.parquet)
├── posts/        # Daily post data (.parquet)
├── comments/     # Daily comment data (.parquet)
├── submolts/     # Community metadata (.parquet)
├── snapshots/    # Periodic global observatory metrics (.parquet)
└── word_freq/    # Hourly word frequency statistics (.parquet)
```

### Usage

Place the cloned archive at the repository root (or pass `--dataset-path`
to the analysis scripts). The `molt_dynamics` pipeline loads parquet files
via `MoltBookDatasetLoader` → `JSONStorage`.
