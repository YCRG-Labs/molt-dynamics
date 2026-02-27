import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import molt_dynamics.rq1_roles as rq1_roles
from molt_dynamics.config import Config
from molt_dynamics.rq1_roles import RoleAnalyzer


def _features_df(n_agents: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "agent_id": [f"agent_{i}" for i in range(n_agents)],
            "in_degree": rng.normal(size=n_agents),
            "out_degree": rng.normal(size=n_agents),
            "betweenness": rng.normal(size=n_agents),
            "clustering_coefficient": rng.normal(size=n_agents),
            "pagerank": rng.normal(size=n_agents),
        }
    )


def test_tsne_uses_n_iter_when_max_iter_unsupported(monkeypatch):
    seen = {}

    class DummyTSNE:
        def __init__(self, n_components, perplexity, n_iter, random_state):
            seen["kwargs"] = {
                "n_components": n_components,
                "perplexity": perplexity,
                "n_iter": n_iter,
                "random_state": random_state,
            }

        def fit_transform(self, X):
            return np.zeros((X.shape[0], 2))

    monkeypatch.setattr(rq1_roles, "TSNE", DummyTSNE)

    analyzer = RoleAnalyzer(_features_df(), Config())
    emb = analyzer.compute_tsne_embedding(feature_set="network", perplexity=5, max_iter=123)

    assert emb.shape == (10, 2)
    assert seen["kwargs"]["n_iter"] == 123
    assert "max_iter" not in seen["kwargs"]


def test_tsne_uses_max_iter_when_supported(monkeypatch):
    seen = {}

    class DummyTSNE:
        def __init__(self, n_components, perplexity, max_iter, random_state):
            seen["kwargs"] = {
                "n_components": n_components,
                "perplexity": perplexity,
                "max_iter": max_iter,
                "random_state": random_state,
            }

        def fit_transform(self, X):
            return np.zeros((X.shape[0], 2))

    monkeypatch.setattr(rq1_roles, "TSNE", DummyTSNE)

    analyzer = RoleAnalyzer(_features_df(), Config())
    emb = analyzer.compute_tsne_embedding(feature_set="network", perplexity=5, max_iter=456)

    assert emb.shape == (10, 2)
    assert seen["kwargs"]["max_iter"] == 456
    assert "n_iter" not in seen["kwargs"]

