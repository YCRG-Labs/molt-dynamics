# Rigor re-analysis runbook (CAISc 2026)

`scripts/run_rigor.py` is a standalone, vectorized re-analysis that reads the
Observatory parquet directly and produces the validity-first numbers the paper
needs. It does NOT use the slow JSONStorage/iterrows path.

## Why this exists

The original `rq3_collaboration.py` never computed the paper's headline
(`t=-11.21, d=-0.88, "matched single-agent baseline"`): `collect_individual_baselines`
returns an empty DataFrame and the only implemented test is one-sample vs the
constant 0.5. The null models in `validation.py` are an `np.random.normal` stub.
This script replaces those with real computations.

## What it computes

- **Task 1**: degree-preserving directed configuration-model nulls; reports
  observed vs null (reciprocity, transitivity, assortativity) with empirical p,
  plus an honest k-means/silhouette replication (singletons flagged, silhouette
  subsampled) and Gini of activity. The periphery fraction is reported as a
  degree-sequence property (i.e. not emergent).
- **Task 2**: cascades as verbatim content groups across >=2 distinct agents
  (size = distinct adopters, fixing the 8.5M>1.2M raw-row over-count); honest
  Clauset power-law via `powerlaw` with a power-law-vs-lognormal verdict; a
  label-shuffle null on the cascade-size Gini.
- **Task 3**: a REAL matched single-agent baseline (submolt + comment-count +-2),
  with both the paper's single-agent-biased metric and a collaboration-neutral
  metric, reporting two-sample t and Cohen's d for each.
- **Autonomy**: per-agent CoV of inter-event intervals (regular cadence =>
  autonomous), a threshold sweep, and a re-run of all tasks on the
  autonomous-only subset (deltas).
- **FDR**: Benjamini-Hochberg across the reported p-value family.

## Run on Brev

```bash
# 1. env (Python 3.11)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. data (~5 GB)
hf download SimulaMet/moltbook-observatory-archive --repo-type dataset \
    --local-dir moltbook-observatory-archive

# 3. smoke test first (fast, ~1 min)
python scripts/run_rigor.py \
    --data-dir moltbook-observatory-archive/data \
    --smoke 50000 --out results/smoke.json

# 4. full run (3-week window used in the paper)
python scripts/run_rigor.py \
    --data-dir moltbook-observatory-archive/data \
    --start 2026-01-28 --end 2026-02-20 \
    --null-iters 100 --shuffle-iters 50 \
    --out results/rigor_results.json
```

Send `results/rigor_results.json` back and the paper numbers get updated to the
real, defensible values.

## Notes
- `--cov-threshold` (default 0.75) sets the autonomy cutoff; the sweep is always
  reported so the choice is transparent.
- Stages are independently wrapped: a failure in one writes an `error` field and
  the rest still run.
- Betweenness for the clustering uses k-sample approximation (exact is O(VE) on
  ~90k nodes). The nulls deliberately use O(E) statistics so 100 iterations are
  tractable.
