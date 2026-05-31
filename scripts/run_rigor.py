"""Validity-first rigor pipeline for the MoltBook analysis (CAISc 2026).

Standalone, vectorized re-analysis that reads the Observatory parquet files
directly (bypassing the slow JSONStorage/iterrows path) and produces the
rigorous numbers the paper needs:

  Task 1  structural roles + DEGREE-PRESERVING NULL MODELS
          (which structure exceeds what the degree sequence forces?)
  Task 2  information diffusion with HONEST Clauset power-law reporting
          (power-law vs lognormal verdict) + a timestamp-shuffle null
  Task 3  a REAL matched single-agent baseline (the paper's headline
          d=-0.88 has no generating code) + a collaboration-neutral metric
  Autonomy  a coefficient-of-variation (CoV) heartbeat filter with a
          threshold sweep; key stats re-run on the autonomous-only subset
  FDR     Benjamini-Hochberg across the reported p-value family

Everything is wrapped per-stage so a partial failure still writes results.

Usage (on Brev):
    python scripts/run_rigor.py \
        --data-dir moltbook-observatory-archive/data \
        --start 2026-01-28 --end 2026-02-20 \
        --out results/rigor_results.json

Smoke test first (fast, sample of posts):
    python scripts/run_rigor.py --data-dir ... --smoke 50000 --out results/smoke.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rigor")

SEED = 42
np.random.seed(SEED)

TECHNICAL_KEYWORDS = {
    "error", "bug", "fix", "issue", "problem", "help", "solution", "solve",
    "debug", "crash", "exception", "fail", "broken", "stuck", "how to",
    "implement", "code", "function", "method", "class", "api", "library",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _find_parquet(data_dir: str, table: str) -> str | None:
    """Locate a table's parquet (single consolidated file or first partition)."""
    cands = [
        os.path.join(data_dir, table, f"{table}.parquet"),
        os.path.join(data_dir, f"{table}.parquet"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    import glob
    hits = sorted(glob.glob(os.path.join(data_dir, table, "*.parquet")))
    if hits:
        return hits[0] if len(hits) == 1 else None  # signal multi below
    return None


def load_table(data_dir: str, table: str, columns: list[str]) -> pd.DataFrame:
    """Load a table, concatenating partitions if needed, keeping only `columns`."""
    import glob
    folder = os.path.join(data_dir, table)
    files = []
    single = os.path.join(folder, f"{table}.parquet")
    flat = os.path.join(data_dir, f"{table}.parquet")
    if os.path.exists(single):
        files = [single]
    elif os.path.exists(flat):
        files = [flat]
    elif os.path.isdir(folder):
        files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet for table '{table}' under {data_dir}")
    frames = []
    for f in files:
        avail = set(pd.read_parquet(f, engine="pyarrow", columns=None).columns) \
            if False else None  # avoid double read; rely on try below
        try:
            frames.append(pd.read_parquet(f, columns=columns))
        except Exception:
            df = pd.read_parquet(f)
            keep = [c for c in columns if c in df.columns]
            frames.append(df[keep])
    df = pd.concat(frames, ignore_index=True)
    log.info("loaded %s: %d rows from %d file(s)", table, len(df), len(files))
    return df


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def prepare(data_dir: str, start: str, end: str, smoke: int | None):
    """Load + dedup + window posts/comments/agents."""
    posts = load_table(data_dir, "posts",
                        ["id", "agent_id", "submolt", "title", "content",
                         "score", "comment_count", "created_at", "dump_date"])
    comments = load_table(data_dir, "comments",
                          ["id", "post_id", "agent_id", "parent_id", "content",
                           "score", "created_at", "dump_date"])

    # Dedup incremental dumps: keep last-seen row per id
    for name, df in (("posts", posts), ("comments", comments)):
        if "dump_date" in df.columns:
            df.sort_values("dump_date", inplace=True, kind="stable")
        before = len(df)
        df.drop_duplicates(subset="id", keep="last", inplace=True)
        log.info("%s dedup: %d -> %d", name, before, len(df))

    posts["created_at"] = _to_dt(posts["created_at"])
    comments["created_at"] = _to_dt(comments["created_at"])

    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    posts = posts[(posts["created_at"] >= lo) & (posts["created_at"] < hi)].copy()
    comments = comments[(comments["created_at"] >= lo) & (comments["created_at"] < hi)].copy()
    log.info("after window [%s,%s): posts=%d comments=%d", start, end, len(posts), len(comments))

    if smoke:
        posts = posts.sort_values("created_at").head(smoke).copy()
        keep_ids = set(posts["id"])
        comments = comments[comments["post_id"].isin(keep_ids)].copy()
        log.info("SMOKE: posts=%d comments=%d", len(posts), len(comments))

    for df in (posts, comments):
        df["agent_id"] = df["agent_id"].astype(str)
    return posts, comments


# --------------------------------------------------------------------------- #
# Interaction network (vectorized reply edges)
# --------------------------------------------------------------------------- #
def build_edges(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """commenter -> (parent comment author | post author). Weighted edge list."""
    post_author = posts.set_index("id")["agent_id"]
    comment_author = comments.set_index("id")["agent_id"]

    c = comments.copy()
    c["parent_id"] = c["parent_id"].astype("object")
    is_comment_reply = c["parent_id"].notna() & c["parent_id"].astype(str).str.startswith("t1_")

    tgt = pd.Series(index=c.index, dtype="object")
    # replies to comments -> parent comment author
    tgt.loc[is_comment_reply] = c.loc[is_comment_reply, "parent_id"].map(comment_author)
    # everything else -> post author
    tgt.loc[~is_comment_reply] = c.loc[~is_comment_reply, "post_id"].map(post_author)

    e = pd.DataFrame({"src": c["agent_id"].values, "dst": tgt.values})
    e = e.dropna()
    e = e[e["src"] != e["dst"]]
    edges = e.groupby(["src", "dst"]).size().reset_index(name="weight")
    log.info("edges: %d directed pairs over %d interactions",
             len(edges), int(edges["weight"].sum()))
    return edges


def build_graph(edges: pd.DataFrame):
    import networkx as nx
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges[["src", "dst", "weight"]].itertuples(index=False, name=None))
    log.info("graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# --------------------------------------------------------------------------- #
# Autonomy (coefficient-of-variation heartbeat filter)
# --------------------------------------------------------------------------- #
def autonomy_cov(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """Per-agent CoV of inter-event intervals. Autonomous agents fire on a
    regular cadence (heartbeat) => LOW CoV; human-steered => irregular/bursty."""
    ev = pd.concat([
        posts[["agent_id", "created_at"]],
        comments[["agent_id", "created_at"]],
    ], ignore_index=True).dropna()
    ev = ev.sort_values(["agent_id", "created_at"])
    out = []
    for aid, g in ev.groupby("agent_id", sort=False):
        ts = g["created_at"].values
        if len(ts) < 5:           # need enough events to estimate cadence
            continue
        dt = np.diff(ts).astype("timedelta64[s]").astype(float)
        dt = dt[dt > 0]
        if len(dt) < 4:
            continue
        m = dt.mean()
        if m <= 0:
            continue
        out.append((aid, len(ts), m / 3600.0, dt.std() / m))
    df = pd.DataFrame(out, columns=["agent_id", "n_events", "mean_gap_hours", "cov"])
    log.info("autonomy: scored %d agents (>=5 events)", len(df))
    return df


# --------------------------------------------------------------------------- #
# Task 1: structure + degree-preserving null models
# --------------------------------------------------------------------------- #
def task1(G, autonomous_ids: set | None = None) -> dict:
    import networkx as nx
    res = {}
    n = G.number_of_nodes()
    res["n_nodes"] = n
    res["n_edges"] = G.number_of_edges()
    if n < 10:
        return {"result": "insufficient_nodes", **res}

    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    tot = {x: indeg.get(x, 0) + outdeg.get(x, 0) for x in G.nodes()}
    deg_arr = np.array(list(tot.values()))

    # Honest "periphery": agents with total degree <= 1. This is a property of
    # the degree sequence, so it should be ~identical under a degree null --
    # demonstrating it is NOT emergent coordination.
    res["periphery_frac_deg_le_1"] = float((deg_arr <= 1).mean())
    res["gini_total_degree"] = _gini(deg_arr)

    # Observed structural statistics
    def stats_of(H):
        Hu = H.to_undirected()
        return {
            "reciprocity": float(nx.reciprocity(H)) if H.number_of_edges() else 0.0,
            "transitivity": float(nx.transitivity(Hu)),
            "assortativity": _safe(lambda: nx.degree_assortativity_coefficient(H)),
        }

    obs = stats_of(G)
    res["observed"] = obs
    return res  # nulls attached separately (expensive); see task1_nulls


def task1_nulls(G, n_iter: int) -> dict:
    """Degree-preserving directed configuration-model null."""
    import networkx as nx
    din = [d for _, d in G.in_degree()]
    dout = [d for _, d in G.out_degree()]
    keys = ["reciprocity", "transitivity", "assortativity"]
    samples = {k: [] for k in keys}
    for it in range(n_iter):
        try:
            Gc = nx.directed_configuration_model(din, dout, seed=SEED + it)
            Gc = nx.DiGraph(Gc)                       # collapse multi-edges
            Gc.remove_edges_from(nx.selfloop_edges(Gc))
            Gcu = Gc.to_undirected()
            samples["reciprocity"].append(nx.reciprocity(Gc))
            samples["transitivity"].append(nx.transitivity(Gcu))
            samples["assortativity"].append(_safe(lambda: nx.degree_assortativity_coefficient(Gc)))
        except Exception as e:
            log.warning("null iter %d failed: %s", it, e)
    out = {}
    obs = task1(G)["observed"]
    for k in keys:
        arr = np.array([x for x in samples[k] if x is not None and np.isfinite(x)])
        if len(arr) < 3:
            out[k] = {"observed": obs[k], "null_n": len(arr)}
            continue
        mu, sd = float(arr.mean()), float(arr.std(ddof=1))
        z = (obs[k] - mu) / sd if sd > 0 else 0.0
        p = float((np.abs(arr - mu) >= abs(obs[k] - mu)).mean())  # two-sided empirical
        out[k] = {"observed": float(obs[k]), "null_mean": mu, "null_std": sd,
                  "z": float(z), "emp_p": p, "null_n": len(arr)}
    return out


def task1_clusters(G, sample_silhouette: int = 10000) -> dict:
    """Replicate the k-means role clustering, honestly (drop singletons,
    subsample silhouette, report cluster sizes)."""
    import networkx as nx
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    nodes = list(G.nodes())
    n = len(nodes)
    if n < 50:
        return {"result": "insufficient_nodes"}
    indeg = dict(G.in_degree()); outdeg = dict(G.out_degree())
    pr = nx.pagerank(G, alpha=0.85)
    # approximate betweenness (exact is O(VE); k-sample keeps it tractable)
    k = min(1000, n)
    btw = nx.betweenness_centrality(G, k=k, seed=SEED)
    clu = nx.clustering(G.to_undirected())
    X = np.array([[indeg[v], outdeg[v], btw[v], clu[v], pr[v]] for v in nodes], dtype=float)
    Xs = StandardScaler().fit_transform(X)

    rng = np.random.RandomState(SEED)
    samp = rng.choice(n, size=min(sample_silhouette, n), replace=False)
    best = None
    for kk in range(3, 9):
        km = KMeans(n_clusters=kk, random_state=SEED, n_init=10).fit(Xs)
        sil = silhouette_score(Xs[samp], km.labels_[samp])
        sizes = np.bincount(km.labels_).tolist()
        n_singletons = int(sum(1 for s in sizes if s == 1))
        rec = {"k": kk, "silhouette": float(sil), "sizes": sizes,
               "n_singletons": n_singletons,
               "largest_frac": float(max(sizes) / n)}
        if best is None or sil > best["silhouette"]:
            best = rec
    best["note"] = ("silhouette is dominated by the large low-degree periphery; "
                    "singleton clusters are degree outliers, not roles")
    return best


# --------------------------------------------------------------------------- #
# Task 2: cascades + honest power-law + timestamp-shuffle null
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
def _norm(t: str) -> str:
    return _WS.sub(" ", str(t).lower().strip())


def cascades_from_text(posts: pd.DataFrame, min_len: int = 12) -> pd.DataFrame:
    """Verbatim/near-verbatim content groups across >=2 distinct agents.
    Cascade size = number of DISTINCT adopting agents (fixes the 8.5M>1.2M
    raw-row over-count)."""
    p = posts[["agent_id", "content", "created_at"]].dropna().copy()
    p["norm"] = p["content"].map(_norm)
    p = p[p["norm"].str.len() >= min_len]
    grp = p.groupby("norm")
    agg = grp.agg(n_agents=("agent_id", "nunique"),
                  n_posts=("agent_id", "size"),
                  seed_time=("created_at", "min"))
    casc = agg[agg["n_agents"] >= 2].reset_index()
    log.info("cascades: %d content groups with >=2 agents", len(casc))
    return casc


def powerlaw_report(sizes: np.ndarray) -> dict:
    import powerlaw
    sizes = np.asarray(sizes)
    sizes = sizes[sizes > 0]
    if len(sizes) < 50:
        return {"result": "insufficient", "n": int(len(sizes))}
    fit = powerlaw.Fit(sizes, discrete=True, verbose=False)
    out = {
        "n": int(len(sizes)),
        "alpha": float(fit.power_law.alpha),
        "xmin": float(fit.power_law.xmin),
        "n_tail": int((sizes >= fit.power_law.xmin).sum()),
        "frac_mass_in_tail": float((sizes >= fit.power_law.xmin).mean()),
    }
    for alt in ("lognormal", "exponential", "truncated_power_law"):
        try:
            R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
            out[alt] = {"R": float(R), "p": float(p),
                        "powerlaw_favored": bool(R > 0 and p < 0.05)}
        except Exception as e:
            out[alt] = {"error": str(e)}
    ln = out.get("lognormal", {})
    if ln and not ln.get("powerlaw_favored", False):
        out["verdict"] = ("heavy-tailed; power-law NOT distinguishable from "
                          "lognormal (do not claim power-law)")
    else:
        out["verdict"] = "power-law favored over lognormal"
    return out


def task2(posts: pd.DataFrame, n_shuffle: int = 20) -> dict:
    res = {}
    casc = cascades_from_text(posts)
    res["n_cascades"] = int(len(casc))
    if len(casc) < 50:
        res["result"] = "too_few_cascades"
        return res
    sizes = casc["n_agents"].values
    res["total_adopting_agents"] = int(sizes.sum())
    res["powerlaw"] = _safe_dict(lambda: powerlaw_report(sizes))

    # Timestamp-shuffle null: does cascade SIZE structure exceed chance grouping?
    # Compare observed Gini of cascade sizes to sizes under shuffled agent labels.
    obs_gini = _gini(sizes)
    null_ginis = []
    p = posts[["agent_id", "content"]].dropna().copy()
    p["norm"] = p["content"].map(_norm)
    p = p[p["norm"].str.len() >= 12]
    agents = p["agent_id"].values
    rng = np.random.RandomState(SEED)
    for _ in range(n_shuffle):
        shuffled = rng.permutation(agents)
        tmp = pd.DataFrame({"agent_id": shuffled, "norm": p["norm"].values})
        g = tmp.groupby("norm")["agent_id"].nunique()
        g = g[g >= 2].values
        if len(g):
            null_ginis.append(_gini(g))
    if null_ginis:
        arr = np.array(null_ginis)
        res["size_gini_vs_null"] = {
            "observed": float(obs_gini), "null_mean": float(arr.mean()),
            "null_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "emp_p": float((arr >= obs_gini).mean()),
        }
    return res


# --------------------------------------------------------------------------- #
# Task 3: real matched single-agent baseline + neutral metric
# --------------------------------------------------------------------------- #
def _has_kw(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in TECHNICAL_KEYWORDS)


def _biased_quality(code: str) -> float:
    """The paper's metric: rewards a single clean code-bearing post."""
    has_code = bool(re.search(r"```|`[^`]+`", code))
    has_comments = bool(re.search(r"#.*|//.*|/\*.*\*/", code))
    has_tests = bool(re.search(r"test|assert|expect", code.lower()))
    syntax = _balanced(code)
    return 0.3 * has_code + 0.2 * has_comments + 0.3 * has_tests + 0.2 * syntax


def _balanced(code: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for ch in code:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return False
    return len(stack) == 0


def _neutral_quality(texts: list[str]) -> float:
    """Collaboration-neutral: thread reaches a quality artifact regardless of
    whether it is concentrated in one post or distributed across many.
    Uses the MAX over contributions (so distribution is not penalized) plus a
    thread-level resolution signal."""
    joined = "\n".join(texts)
    best_code = max((_biased_quality(t) for t in texts), default=0.0)
    resolved = bool(re.search(r"\b(works|fixed|solved|thanks|got it|that did it|resolved)\b",
                              joined.lower()))
    has_code_anywhere = bool(re.search(r"```|`[^`]+`", joined))
    return float(np.clip(0.6 * best_code + 0.25 * resolved + 0.15 * has_code_anywhere, 0, 1))


def task3(posts: pd.DataFrame, comments: pd.DataFrame) -> dict:
    from scipy import stats
    res = {}
    # group comments by post
    cby = comments.groupby("post_id")
    pmeta = posts.set_index("id")
    collab, single = [], []
    for pid, g in cby:
        if pid not in pmeta.index:
            continue
        post = pmeta.loc[pid]
        if isinstance(post, pd.DataFrame):
            post = post.iloc[0]
        bodies = [str(post.get("title", "")), str(post.get("content", ""))] + \
                 g["content"].astype(str).tolist()
        text = " ".join(bodies)
        n_comments = len(g)
        participants = set(g["agent_id"]) | {str(post["agent_id"])}
        if n_comments < 5 or not _has_kw(text):
            continue
        dur_min = (g["created_at"].max() - g["created_at"].min()).total_seconds() / 60.0
        rec = {"pid": pid, "submolt": post.get("submolt", ""),
               "n_comments": n_comments, "n_part": len(participants),
               "biased": _biased_quality(text), "neutral": _neutral_quality(bodies)}
        if len(participants) >= 3 and dur_min >= 30:
            collab.append(rec)
        elif len(participants) == 1:
            single.append(rec)
    res["n_collab"] = len(collab)
    res["n_single"] = len(single)
    if len(collab) < 5 or len(single) < 5:
        res["result"] = "insufficient"
        return res

    cdf, sdf = pd.DataFrame(collab), pd.DataFrame(single)
    # match each collab event to single-agent threads by submolt + n_comments +-2
    matched_idx = []
    for _, r in cdf.iterrows():
        m = sdf[(sdf["submolt"] == r["submolt"]) &
                (sdf["n_comments"].between(r["n_comments"] - 2, r["n_comments"] + 2))]
        if len(m):
            matched_idx.append(m.sample(min(5, len(m)), random_state=SEED).index)
    if matched_idx:
        mi = np.unique(np.concatenate([np.array(x) for x in matched_idx]))
        base = sdf.loc[mi]
    else:
        base = sdf
    res["n_matched_single"] = int(len(base))

    for metric in ("biased", "neutral"):
        a, b = cdf[metric].values, base[metric].values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = _cohens_d(a, b)
        res[metric] = {
            "collab_mean": float(np.mean(a)), "single_mean": float(np.mean(b)),
            "t": float(t), "p": float(p), "cohens_d": float(d),
            "collab_better": bool(np.mean(a) > np.mean(b)),
        }
    res["note"] = ("'biased' replicates the paper's single-agent-favoring metric; "
                   "'neutral' is collaboration-neutral. Compare the two d's.")
    return res


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _cohens_d(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def _safe(fn):
    try:
        v = fn()
        return float(v) if v is not None and np.isfinite(v) else None
    except Exception:
        return None


def _safe_dict(fn):
    try:
        return fn()
    except Exception as e:
        return {"error": str(e)}


def fdr_bh(pvals: dict) -> dict:
    """Benjamini-Hochberg across a flat dict {label: p}."""
    items = [(k, v) for k, v in pvals.items() if isinstance(v, (int, float)) and np.isfinite(v)]
    if not items:
        return {}
    labels, ps = zip(*items)
    try:
        from statsmodels.stats.multitest import multipletests
        rej, q, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")
        return {labels[i]: {"p": float(ps[i]), "q": float(q[i]), "sig_fdr": bool(rej[i])}
                for i in range(len(labels))}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="path to .../moltbook-observatory-archive/data")
    ap.add_argument("--start", default="2026-01-28")
    ap.add_argument("--end", default="2026-02-20")
    ap.add_argument("--out", default="results/rigor_results.json")
    ap.add_argument("--smoke", type=int, default=None, help="sample N posts for a fast dry run")
    ap.add_argument("--null-iters", type=int, default=50)
    ap.add_argument("--shuffle-iters", type=int, default=20)
    ap.add_argument("--cov-threshold", type=float, default=0.75,
                    help="agents with CoV below this are 'autonomous' (regular cadence)")
    args = ap.parse_args()

    t0 = time.time()
    R = {"config": vars(args), "seed": SEED}

    posts, comments = prepare(args.data_dir, args.start, args.end, args.smoke)
    R["counts"] = {"posts": len(posts), "comments": len(comments),
                   "unique_agents": int(pd.concat([posts["agent_id"], comments["agent_id"]]).nunique())}

    edges = build_edges(posts, comments)
    G = build_graph(edges)

    # ---- autonomy ----
    cov = _stage("autonomy_cov", lambda: autonomy_cov(posts, comments))
    autonomous = set()
    if isinstance(cov, pd.DataFrame) and len(cov):
        thr = args.cov_threshold
        autonomous = set(cov.loc[cov["cov"] <= thr, "agent_id"])
        sweep = {}
        for t in (0.3, 0.5, 0.75, 1.0, 1.5):
            sweep[str(t)] = int((cov["cov"] <= t).sum())
        R["autonomy"] = {
            "n_scored": int(len(cov)),
            "cov_threshold": thr,
            "n_autonomous": int(len(autonomous)),
            "frac_autonomous": float(len(autonomous) / max(1, len(cov))),
            "cov_quantiles": {q: float(cov["cov"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "threshold_sweep_counts": sweep,
        }

    # ---- task 1 ----
    R["task1"] = _stage("task1", lambda: task1(G, autonomous))
    R["task1_nulls"] = _stage("task1_nulls", lambda: task1_nulls(G, args.null_iters))
    R["task1_clusters"] = _stage("task1_clusters", lambda: task1_clusters(G))

    # ---- task 2 ----
    R["task2"] = _stage("task2", lambda: task2(posts, args.shuffle_iters))

    # ---- task 3 ----
    R["task3"] = _stage("task3", lambda: task3(posts, comments))

    # ---- autonomous-only re-run (deltas) ----
    if autonomous:
        ap_ = posts[posts["agent_id"].isin(autonomous)].copy()
        ac_ = comments[comments["agent_id"].isin(autonomous)].copy()
        R["autonomous_only"] = {
            "posts": len(ap_), "comments": len(ac_),
            "task1": _stage("auto.task1", lambda: task1(build_graph(build_edges(ap_, ac_)))),
            "task2": _stage("auto.task2", lambda: task2(ap_, args.shuffle_iters)),
            "task3": _stage("auto.task3", lambda: task3(ap_, ac_)),
        }

    # ---- FDR across reported p-values ----
    pvals = {}
    for k in ("reciprocity", "transitivity", "assortativity"):
        v = R.get("task1_nulls", {}).get(k, {})
        if "emp_p" in v:
            pvals[f"t1.{k}"] = v["emp_p"]
    for m in ("biased", "neutral"):
        v = R.get("task3", {}).get(m, {})
        if "p" in v:
            pvals[f"t3.{m}"] = v["p"]
    R["fdr_bh"] = fdr_bh(pvals)

    R["runtime_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    log.info("wrote %s (%.1fs)", args.out, R["runtime_sec"])
    print(json.dumps({k: R[k] for k in ("counts", "autonomy", "task1_nulls", "task2", "task3", "fdr_bh") if k in R},
                     indent=2, default=str))


def _stage(name, fn):
    try:
        t = time.time()
        out = fn()
        log.info("stage %s done (%.1fs)", name, time.time() - t)
        return out
    except Exception as e:
        import traceback
        log.error("stage %s FAILED: %s", name, e)
        return {"error": str(e), "traceback": traceback.format_exc()[-1500:]}


if __name__ == "__main__":
    main()
