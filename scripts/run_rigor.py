"""Validity-first rigor pipeline for the MoltBook analysis (CAISc 2026).

FAST standalone re-analysis. Reads the Observatory parquet directly with
polars (predicate pushdown + streaming), builds the interaction graph with
igraph (C core), and parallelizes the null models with joblib.

Produces the rigorous numbers the paper needs:

  Task 1  structural roles + DEGREE-PRESERVING NULL MODELS (igraph config model)
  Task 2  cascades (size = distinct adopters) + HONEST Clauset power-law verdict
          + a label-shuffle null on the cascade-size Gini
  Task 3  a REAL matched single-agent baseline (the paper's d=-0.88 has no
          generating code) with the paper's biased metric AND a
          collaboration-neutral metric
  Autonomy  per-agent CoV heartbeat filter + threshold sweep + autonomous-only
          re-run of every task
  FDR     Benjamini-Hochberg across the reported p-value family

GPU (optional, Brev): wrap with `python -m cudf.pandas scripts/run_rigor.py ...`
and/or `NX_CUGRAPH_AUTOCONFIG=True`. Not required; the polars+igraph CPU path is
already minutes, not hours.

Usage:
    python scripts/run_rigor.py --data-dir moltbook-observatory-archive/data \
        --start 2026-01-28 --end 2026-02-20 --out results/rigor_results.json
Smoke first:
    python scripts/run_rigor.py --data-dir ... --smoke 60000 --out results/smoke.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

import numpy as np
import polars as pl
import igraph as ig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)], datefmt="%H:%M:%S")
log = logging.getLogger("rigor")

SEED = 42
np.random.seed(SEED)

TECHNICAL_KEYWORDS = [
    "error", "bug", "fix", "issue", "problem", "help", "solution", "solve",
    "debug", "crash", "exception", "fail", "broken", "stuck", "how to",
    "implement", "code", "function", "method", "class", "api", "library",
]
_KW_RE = re.compile("|".join(re.escape(k) for k in TECHNICAL_KEYWORDS))


# --------------------------------------------------------------------------- #
def _ca_naive_expr(dtype):
    """created_at -> tz-naive Datetime[us], whether stored as string or datetime
    (files mix us/ns precision and tz, so normalize before any concat)."""
    if dtype == pl.String:
        return (pl.col("created_at")
                .str.to_datetime(strict=False, time_unit="us", time_zone="UTC")
                .dt.replace_time_zone(None))
    e = pl.col("created_at").dt.cast_time_unit("us")
    if getattr(dtype, "time_zone", None) is not None:
        e = e.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return e


def _scan(data_dir, table, cols, lo, hi):
    import glob
    folder = os.path.join(data_dir, table)
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        flat = os.path.join(data_dir, f"{table}.parquet")
        files = [flat] if os.path.exists(flat) else []
    if not files:
        raise FileNotFoundError(f"no parquet for {table} under {data_dir}")
    win = (pl.col("created_at") >= lo) & (pl.col("created_at") < hi)
    lfs = []
    for f in files:
        lf = pl.scan_parquet(f).select(cols)
        dtype = lf.collect_schema()["created_at"]
        lfs.append(lf.with_columns(_ca_naive_expr(dtype).alias("created_at")).filter(win))
    lf = pl.concat(lfs, how="vertical_relaxed")
    if "dump_date" in cols:
        lf = lf.sort("dump_date").unique(subset="id", keep="last")
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def prepare(data_dir, start, end, smoke):
    from datetime import datetime, timedelta
    lo = datetime.fromisoformat(start)
    hi = datetime.fromisoformat(end) + timedelta(days=1)
    posts = _scan(data_dir, "posts",
                  ["id", "agent_id", "submolt", "title", "content", "created_at", "dump_date"], lo, hi)
    comments = _scan(data_dir, "comments",
                     ["id", "post_id", "agent_id", "parent_id", "content", "created_at", "dump_date"], lo, hi)
    posts = posts.with_columns(pl.col("agent_id").cast(pl.Utf8))
    comments = comments.with_columns(pl.col("agent_id").cast(pl.Utf8))
    log.info("window posts=%d comments=%d", posts.height, comments.height)
    if smoke:
        posts = posts.sort("created_at").head(smoke)
        keep = set(posts["id"].to_list())
        comments = comments.filter(pl.col("post_id").is_in(keep))
        log.info("SMOKE posts=%d comments=%d", posts.height, comments.height)
    return posts, comments


def build_edges(posts, comments):
    post_auth = posts.select(["id", "agent_id"]).rename({"id": "post_id", "agent_id": "p_auth"})
    com_auth = comments.select(["id", "agent_id"]).rename({"id": "parent_id", "agent_id": "c_auth"})
    c = (comments.select(["agent_id", "post_id", "parent_id"]).rename({"agent_id": "src"})
         .join(post_auth, on="post_id", how="left")
         .join(com_auth, on="parent_id", how="left")
         .with_columns(pl.coalesce(["c_auth", "p_auth"]).alias("dst"))
         .drop_nulls("dst").filter(pl.col("src") != pl.col("dst")))
    edges = c.group_by(["src", "dst"]).len().rename({"len": "weight"})
    log.info("edges=%d interactions=%d", edges.height, int(edges["weight"].sum()) if edges.height else 0)
    return edges


def build_graph(edges):
    src = edges["src"].to_list(); dst = edges["dst"].to_list(); w = edges["weight"].to_list()
    names = list(dict.fromkeys(src + dst))
    idx = {n: i for i, n in enumerate(names)}
    g = ig.Graph(directed=True)
    g.add_vertices(len(names)); g.vs["name"] = names
    g.add_edges([(idx[a], idx[b]) for a, b in zip(src, dst)])
    g.es["weight"] = w
    log.info("graph n=%d m=%d", g.vcount(), g.ecount())
    return g


def autonomy_cov(posts, comments):
    import pandas as pd
    ev = pl.concat([posts.select(["agent_id", "created_at"]),
                    comments.select(["agent_id", "created_at"])]).drop_nulls().sort(["agent_id", "created_at"])
    g = (ev.group_by("agent_id")
         .agg(n_events=pl.len(),
              gaps=pl.col("created_at").diff().dt.total_seconds().drop_nulls())
         .filter(pl.col("n_events") >= 5))
    rows = []
    for aid, n, gaps in zip(g["agent_id"].to_list(), g["n_events"].to_list(), g["gaps"].to_list()):
        a = np.array([x for x in gaps if x and x > 0], dtype=float)
        if len(a) < 4:
            continue
        m = a.mean()
        if m > 0:
            rows.append((aid, int(n), m / 3600.0, a.std() / m))
    df = pd.DataFrame(rows, columns=["agent_id", "n_events", "mean_gap_hours", "cov"])
    log.info("autonomy: scored %d agents", len(df))
    return df


def _g_stats(g):
    return {"reciprocity": float(g.reciprocity()) if g.ecount() else 0.0,
            "transitivity": float(g.transitivity_undirected(mode="zero")),
            "assortativity": _f(lambda: g.assortativity_degree(directed=True))}


def task1(g):
    indeg = np.array(g.indegree()); outdeg = np.array(g.outdegree()); tot = indeg + outdeg
    return {"n_nodes": g.vcount(), "n_edges": g.ecount(),
            "periphery_frac_deg_le_1": float((tot <= 1).mean()),
            "gini_total_degree": _gini(tot), "observed": _g_stats(g)}


def _null_worker(outdeg, indeg, seed):
    gc = ig.Graph.Degree_Sequence(list(outdeg), list(indeg), method="configuration").simplify()
    return (float(gc.reciprocity()) if gc.ecount() else 0.0,
            float(gc.transitivity_undirected(mode="zero")),
            _f(lambda: gc.assortativity_degree(directed=True)))


def task1_nulls(g, n_iter, n_jobs):
    outdeg, indeg = g.outdegree(), g.indegree()
    try:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_null_worker)(outdeg, indeg, SEED + i) for i in range(n_iter))
    except Exception as e:
        log.warning("joblib unavailable (%s); serial", e)
        res = [_null_worker(outdeg, indeg, SEED + i) for i in range(n_iter)]
    obs = _g_stats(g); keys = ["reciprocity", "transitivity", "assortativity"]; out = {}
    for j, k in enumerate(keys):
        a = np.array([r[j] for r in res if r[j] is not None and np.isfinite(r[j])])
        if len(a) < 3:
            out[k] = {"observed": obs[k], "null_n": int(len(a))}; continue
        mu, sd = float(a.mean()), float(a.std(ddof=1))
        out[k] = {"observed": float(obs[k]), "null_mean": mu, "null_std": sd,
                  "z": float((obs[k] - mu) / sd) if sd > 0 else 0.0,
                  "emp_p": float((np.abs(a - mu) >= abs(obs[k] - mu)).mean()), "null_n": int(len(a))}
    return out


def task1_clusters(g, sample_silhouette=10000, btw_cutoff=4):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    n = g.vcount()
    if n < 50:
        return {"result": "insufficient_nodes"}
    pr = np.array(g.pagerank(weights="weight"))
    btw = np.array(g.betweenness(cutoff=btw_cutoff))
    clu = np.array(g.transitivity_local_undirected(mode="zero"))
    X = np.column_stack([g.indegree(), g.outdegree(), btw, clu, pr]).astype(float)
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(SEED); samp = rng.choice(n, size=min(sample_silhouette, n), replace=False)
    best = None
    for kk in range(3, 9):
        km = KMeans(n_clusters=kk, random_state=SEED, n_init=10).fit(Xs)
        sil = float(silhouette_score(Xs[samp], km.labels_[samp]))
        sizes = np.bincount(km.labels_).tolist()
        rec = {"k": kk, "silhouette": sil, "sizes": sizes,
               "n_singletons": int(sum(1 for s in sizes if s == 1)), "largest_frac": float(max(sizes) / n)}
        if best is None or sil > best["silhouette"]:
            best = rec
    best["note"] = "silhouette dominated by low-degree periphery; singletons are degree outliers, not roles"
    return best


def _cascade_sizes(posts, min_len=12):
    p = (posts.select(["agent_id", "content"]).drop_nulls()
         .with_columns(pl.col("content").str.to_lowercase().str.replace_all(r"\s+", " ")
                       .str.strip_chars().alias("norm"))
         .filter(pl.col("norm").str.len_chars() >= min_len))
    grp = p.group_by("norm").agg(n_agents=pl.col("agent_id").n_unique(), n_posts=pl.len())
    return grp.filter(pl.col("n_agents") >= 2), p


def task2(posts, n_shuffle=20):
    casc, p = _cascade_sizes(posts)
    res = {"n_cascades": int(casc.height)}
    if casc.height < 50:
        res["result"] = "too_few_cascades"; return res
    sizes = casc["n_agents"].to_numpy()
    res["total_adopting_agents"] = int(sizes.sum())
    res["powerlaw"] = _safe(lambda: powerlaw_report(sizes))
    obs = _gini(sizes); agents = p["agent_id"].to_numpy(); norms = p["norm"]
    rng = np.random.RandomState(SEED); nulls = []
    for _ in range(n_shuffle):
        tmp = pl.DataFrame({"agent_id": rng.permutation(agents), "norm": norms})
        v = tmp.group_by("norm").agg(na=pl.col("agent_id").n_unique()).filter(pl.col("na") >= 2)["na"].to_numpy()
        if len(v):
            nulls.append(_gini(v))
    if nulls:
        a = np.array(nulls)
        res["size_gini_vs_null"] = {"observed": float(obs), "null_mean": float(a.mean()),
                                    "null_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                                    "emp_p": float((a >= obs).mean())}
    return res


def powerlaw_report(sizes):
    import powerlaw
    sizes = np.asarray(sizes); sizes = sizes[sizes > 0]
    if len(sizes) < 50:
        return {"result": "insufficient", "n": int(len(sizes))}
    fit = powerlaw.Fit(sizes, discrete=True, verbose=False)
    out = {"n": int(len(sizes)), "alpha": float(fit.power_law.alpha), "xmin": float(fit.power_law.xmin),
           "n_tail": int((sizes >= fit.power_law.xmin).sum()),
           "frac_mass_in_tail": float((sizes >= fit.power_law.xmin).mean())}
    for alt in ("lognormal", "exponential", "truncated_power_law"):
        try:
            R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
            out[alt] = {"R": float(R), "p": float(p), "powerlaw_favored": bool(R > 0 and p < 0.05)}
        except Exception as e:
            out[alt] = {"error": str(e)}
    ln = out.get("lognormal", {})
    out["verdict"] = ("heavy-tailed; power-law NOT distinguishable from lognormal (do not claim power-law)"
                      if not ln.get("powerlaw_favored") else "power-law favored over lognormal")
    return out


def _balanced(code):
    pairs = {"(": ")", "[": "]", "{": "}"}; stack = []
    for ch in code:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return False
    return not stack


def _biased_quality(code):
    has_code = bool(re.search(r"```|`[^`]+`", code))
    has_comments = bool(re.search(r"#.*|//.*|/\*.*\*/", code))
    has_tests = bool(re.search(r"test|assert|expect", code.lower()))
    return 0.3 * has_code + 0.2 * has_comments + 0.3 * has_tests + 0.2 * _balanced(code)


def _neutral_quality(texts):
    joined = "\n".join(texts)
    best = max((_biased_quality(t) for t in texts), default=0.0)
    resolved = bool(re.search(r"\b(works|fixed|solved|thanks|got it|resolved)\b", joined.lower()))
    code_any = bool(re.search(r"```|`[^`]+`", joined))
    return float(np.clip(0.6 * best + 0.25 * resolved + 0.15 * code_any, 0, 1))


def task3(posts, comments):
    from scipy import stats
    import pandas as pd
    pmeta = posts.select(["id", "agent_id", "submolt", "title", "content"]).rename(
        {"id": "post_id", "agent_id": "post_author", "content": "post_content"})
    th = (comments.group_by("post_id").agg(
            n_comments=pl.len(), commenters=pl.col("agent_id"), bodies=pl.col("content"),
            tmin=pl.col("created_at").min(), tmax=pl.col("created_at").max())
          .join(pmeta, on="post_id", how="inner")
          .with_columns(((pl.col("tmax") - pl.col("tmin")).dt.total_seconds() / 60).alias("dur_min"))
          .filter(pl.col("n_comments") >= 5))
    collab, single = [], []
    for r in th.iter_rows(named=True):
        bodies = [str(b) for b in (r["bodies"] or [])]
        head = [str(r["title"] or ""), str(r["post_content"] or "")]
        text = " ".join(head + bodies)
        if not _KW_RE.search(text.lower()):
            continue
        parts = set(r["commenters"] or []) | {r["post_author"]}
        rec = {"submolt": r["submolt"], "n_comments": r["n_comments"],
               "biased": _biased_quality(text), "neutral": _neutral_quality(head + bodies)}
        if len(parts) >= 3 and (r["dur_min"] or 0) >= 30:
            collab.append(rec)
        elif len(parts) == 1:
            single.append(rec)
    res = {"n_collab": len(collab), "n_single": len(single)}
    if len(collab) < 5 or len(single) < 5:
        res["result"] = "insufficient"; return res
    cdf, sdf = pd.DataFrame(collab), pd.DataFrame(single)
    idxs = []
    for _, rr in cdf.iterrows():
        m = sdf[(sdf["submolt"] == rr["submolt"]) &
                (sdf["n_comments"].between(rr["n_comments"] - 2, rr["n_comments"] + 2))]
        if len(m):
            idxs.append(m.sample(min(5, len(m)), random_state=SEED).index.to_numpy())
    base = sdf.loc[np.unique(np.concatenate(idxs))] if idxs else sdf
    res["n_matched_single"] = int(len(base))
    for metric in ("biased", "neutral"):
        a, b = cdf[metric].to_numpy(), base[metric].to_numpy()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        res[metric] = {"collab_mean": float(a.mean()), "single_mean": float(b.mean()),
                       "t": float(t), "p": float(p), "cohens_d": float(_cohens_d(a, b)),
                       "collab_better": bool(a.mean() > b.mean())}
    res["note"] = "'biased'=paper's single-agent-favoring metric; 'neutral'=collaboration-neutral"
    return res


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b); n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def _f(fn):
    try:
        v = fn(); return float(v) if v is not None and np.isfinite(v) else None
    except Exception:
        return None


def _safe(fn):
    try:
        return fn()
    except Exception as e:
        return {"error": str(e)}


def fdr_bh(pvals):
    items = [(k, v) for k, v in pvals.items() if isinstance(v, (int, float)) and np.isfinite(v)]
    if not items:
        return {}
    labels, ps = zip(*items)
    try:
        from statsmodels.stats.multitest import multipletests
        rej, q, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")
        return {labels[i]: {"p": float(ps[i]), "q": float(q[i]), "sig_fdr": bool(rej[i])} for i in range(len(labels))}
    except Exception as e:
        return {"error": str(e)}


def _stage(name, fn):
    t = time.time()
    try:
        out = fn(); log.info("stage %s done (%.1fs)", name, time.time() - t); return out
    except Exception as e:
        import traceback
        log.error("stage %s FAILED: %s", name, e)
        return {"error": str(e), "traceback": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", default="2026-01-28")
    ap.add_argument("--end", default="2026-02-20")
    ap.add_argument("--out", default="results/rigor_results.json")
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--null-iters", type=int, default=100)
    ap.add_argument("--shuffle-iters", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--cov-threshold", type=float, default=0.75)
    args = ap.parse_args()

    t0 = time.time()
    R = {"config": vars(args), "seed": SEED}
    posts, comments = prepare(args.data_dir, args.start, args.end, args.smoke)
    R["counts"] = {"posts": posts.height, "comments": comments.height,
                   "unique_agents": int(pl.concat([posts["agent_id"], comments["agent_id"]]).n_unique())}
    edges = build_edges(posts, comments)
    g = build_graph(edges)

    cov = _stage("autonomy_cov", lambda: autonomy_cov(posts, comments))
    autonomous = set()
    try:
        if hasattr(cov, "empty") and not cov.empty:
            autonomous = set(cov.loc[cov["cov"] <= args.cov_threshold, "agent_id"])
            R["autonomy"] = {"n_scored": int(len(cov)), "cov_threshold": args.cov_threshold,
                             "n_autonomous": int(len(autonomous)),
                             "frac_autonomous": float(len(autonomous) / max(1, len(cov))),
                             "cov_quantiles": {str(q): float(cov["cov"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                             "threshold_sweep": {str(t): int((cov["cov"] <= t).sum()) for t in (0.3, 0.5, 0.75, 1.0, 1.5)}}
    except Exception as e:
        log.warning("autonomy summary failed: %s", e)

    R["task1"] = _stage("task1", lambda: task1(g))
    R["task1_nulls"] = _stage("task1_nulls", lambda: task1_nulls(g, args.null_iters, args.n_jobs))
    R["task1_clusters"] = _stage("task1_clusters", lambda: task1_clusters(g))
    R["task2"] = _stage("task2", lambda: task2(posts, args.shuffle_iters))
    R["task3"] = _stage("task3", lambda: task3(posts, comments))

    if autonomous:
        ap_ = posts.filter(pl.col("agent_id").is_in(autonomous))
        ac_ = comments.filter(pl.col("agent_id").is_in(autonomous))
        R["autonomous_only"] = {"posts": ap_.height, "comments": ac_.height,
                                "task1": _stage("auto.task1", lambda: task1(build_graph(build_edges(ap_, ac_)))),
                                "task2": _stage("auto.task2", lambda: task2(ap_, args.shuffle_iters)),
                                "task3": _stage("auto.task3", lambda: task3(ap_, ac_))}

    pv = {}
    for k in ("reciprocity", "transitivity", "assortativity"):
        v = R.get("task1_nulls", {}).get(k, {})
        if "emp_p" in v:
            pv[f"t1.{k}"] = v["emp_p"]
    for m in ("biased", "neutral"):
        v = R.get("task3", {}).get(m, {})
        if "p" in v:
            pv[f"t3.{m}"] = v["p"]
    R["fdr_bh"] = fdr_bh(pv)

    R["runtime_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    log.info("wrote %s (%.1fs total)", args.out, R["runtime_sec"])
    print(json.dumps({k: R[k] for k in ("counts", "autonomy", "task1", "task1_nulls", "task2", "task3", "fdr_bh") if k in R},
                     indent=2, default=str)[:3500])


if __name__ == "__main__":
    main()
