"""Statistical analysis of the benchmark runs.

Reads the raw CSVs written by scripts/run_benchmarks.py and produces:
  * results/tables/*.csv and *.md  - per-configuration statistics, speedup,
    efficiency, OpenMP overhead, pinning contrast, noisy-vs-quiet comparison,
    flagged observations and data-quality anomalies;
  * results/figures/*.png and *.pdf - time, speedup, efficiency, GFLOP/s,
    pinning contrast and background-load comparison;
  * vectorisation_factorial.* (optional study): the 2x2 factorial
    {scalar, simd build} x {seq, omp(p)} from two runs measured back to back.

Statistics use only measured repetitions (is_warmup == 0) whose status is OK.
No row is deleted or modified. Every row with a non-OK status, a thread-count
mismatch or an affinity failure is listed in anomalies.csv together with its
reason and whether it entered the statistics (in_statistics); configurations
with no OK repetition at all are listed in missing_configs.csv, and
configurations with fewer repetitions than expected are flagged
incomplete_reps.

Definitions (all from medians of the 10 repetitions of a configuration):
  S(p)   = T_seq / T_omp(p)           speedup against the sequential kernel
  E(p)   = S(p) / p                   parallel efficiency
  GFLOP/s = 2 n^3 / T / 1e9
  overhead = T_omp(1) / T_seq          cost of the OpenMP machinery at p = 1
  S range = [Q1_seq / Q3_omp, Q3_seq / Q1_omp]  (interquartile-based bounds)

Flags:
  superlinear      S > p, p > 1 (superlinear_robust if the lower bound > p)
  slowdown         S < 1, p > 1 (slowdown_robust if the upper bound < 1)
  high_dispersion  IQR / median > 0.10 for the omp or the seq configuration
  incomplete_reps  fewer OK repetitions than expected for omp or seq
  oversubscribed   p > logical processors
At p = 1, S is the inverse of the OpenMP overhead and is reported in
omp_overhead instead of being flagged.

Vectorisation factorial (per n and p, medians; bounds as for S):
  V      = T_scalar / T_simd           vectorisation speedup (seq and omp(p))
  S_var  = T_seq,var / T_omp,var(p)     parallel speedup within each build
  S_comb = T_seq,scalar / T_omp,simd(p) both techniques combined
  interaction = S_comb / (V_seq * S_scalar)  (1 = the effects multiply)
If extra runs (--vcheck) re-measure some omp configurations, V_omp_runs_lo/hi
give the range of V over all runs and unstable_across_runs marks
configurations where that range contains 1 (the sign of the effect is not
reproducible); vectorisation_reproducibility lists the per-run medians.

Usage:
    python scripts/analyze.py [--primary CSV] [--unpinned CSV] [--noisy CSV]
                              [--simd CSV --scalar-ctrl CSV]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

KEYS = ["method", "n", "p_requested"]
HIGH_DISPERSION = 0.10  # IQR / median above which a configuration is flagged
REQUIRED_COLUMNS = {
    "run_id", "n", "method", "p_requested", "p_actual", "physical_cores",
    "logical_procs", "pin_mode", "affinity_ok", "rep", "is_warmup", "status",
    "time_s", "peak_rss_bytes", "theoretical_bytes", "checksum",
}


# ---------------------------------------------------------------- loading

def load_run(path, required=True):
    """Read one merged raw CSV and check it has the expected columns.

    A missing optional file returns None (with a warning) instead of aborting.
    """
    if not path.is_file():
        if required:
            sys.exit(f"error: {path} not found")
        print(f"warning: {path} not found, skipping", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"error: {path} lacks columns {sorted(missing)}")
    return df


def measured(df):
    """Rows that enter the statistics: timed repetitions with status OK."""
    return df[(df["is_warmup"] == 0) & (df["status"] == "OK")]


def anomalies(df, label):
    """Rows that are not clean measurements (kept, never deleted).

    in_statistics tells whether the row still enters the statistics (only
    status == OK measured rows do), so a mismatch or affinity problem on an
    OK row is visible rather than silently averaged in.
    """
    checks = ((df["status"] != "OK", "status!=OK"),
              (df["p_actual"] != df["p_requested"], "p_actual!=p_requested"),
              (df["affinity_ok"] != 1, "affinity_ok!=1"))
    reasons = pd.Series([";".join(why for mask, why in checks if mask.loc[i])
                         for i in df.index], index=df.index, dtype=object)
    bad = df[reasons != ""]
    out = bad[["run_id", "method", "n", "p_requested", "p_actual", "pin_mode",
               "rep", "is_warmup", "status", "affinity_ok", "time_s"]].copy()
    out.insert(0, "dataset", label)
    out["in_statistics"] = (bad["is_warmup"] == 0) & (bad["status"] == "OK")
    out["reason"] = reasons[bad.index]
    return out


def missing_configs(df, label):
    """Configurations present in the raw CSV with no measured OK repetition."""
    timed = df[df["is_warmup"] == 0]
    present = timed.groupby(KEYS).size().index
    ok = measured(df).groupby(KEYS).size().index
    out = pd.DataFrame(list(present.difference(ok)), columns=KEYS)
    out.insert(0, "dataset", label)
    return out


def checksum_issues(df, label):
    """Every method and p must produce the same checksum for a given n."""
    m = measured(df)
    counts = m.groupby("n")["checksum"].nunique()
    return [f"{label}: n={n} has {c} distinct checksums"
            for n, c in counts.items() if c != 1]


# ---------------------------------------------------------------- statistics

def config_stats(df):
    """Median / quartiles / IQR and derived metrics per configuration."""
    m = measured(df)
    g = m.groupby(KEYS)
    t = g["time_s"]
    st = pd.DataFrame({
        "reps": t.size(),
        "median_s": t.median(),
        "q1_s": t.quantile(0.25),
        "q3_s": t.quantile(0.75),
        "min_s": t.min(),
        "max_s": t.max(),
        "mean_s": t.mean(),
        "std_s": t.std(ddof=1),
        "peak_rss_bytes": g["peak_rss_bytes"].max(),
        "theoretical_bytes": g["theoretical_bytes"].first(),
        "logical_procs": g["logical_procs"].first(),
        "physical_cores": g["physical_cores"].first(),
    }).reset_index()
    st["iqr_s"] = st["q3_s"] - st["q1_s"]
    st["iqr_rel"] = st["iqr_s"] / st["median_s"]
    st["max_over_min"] = st["max_s"] / st["min_s"]
    st["gflops"] = 2.0 * st["n"].astype(float) ** 3 / st["median_s"] / 1e9
    st["rss_over_theoretical"] = st["peak_rss_bytes"] / st["theoretical_bytes"]
    return st


def speedup_table(st):
    """S(p), E(p) and interquartile bounds of S for every omp configuration."""
    seq = st[st["method"] == "seq"].set_index("n")
    omp = st[st["method"] == "omp"].copy()
    omp = omp[omp["n"].isin(seq.index)]
    s = seq.loc[omp["n"]]
    omp["t_seq_median_s"] = s["median_s"].to_numpy()
    omp["speedup"] = s["median_s"].to_numpy() / omp["median_s"]
    omp["speedup_lo"] = s["q1_s"].to_numpy() / omp["q3_s"]
    omp["speedup_hi"] = s["q3_s"].to_numpy() / omp["q1_s"]
    omp["efficiency"] = omp["speedup"] / omp["p_requested"]
    # Self-relative speedup: T_omp(1) / T_omp(p) (scalability of the omp code).
    t1 = omp[omp["p_requested"] == 1].set_index("n")["median_s"]
    omp["speedup_vs_omp1"] = omp["n"].map(t1) / omp["median_s"]
    omp["seq_iqr_rel"] = s["iqr_rel"].to_numpy()
    omp["seq_reps"] = s["reps"].to_numpy()
    omp["oversubscribed"] = omp["p_requested"] > omp["logical_procs"]
    expected_reps = st["reps"].max()

    flags = []
    for r in omp.itertuples():
        f = []
        # At p = 1, S is just T_seq / T_omp(1) (see omp_overhead), so neither
        # superlinearity nor slowdown is flagged there.
        if r.p_requested > 1 and r.speedup > r.p_requested:
            f.append("superlinear_robust" if r.speedup_lo > r.p_requested
                     else "superlinear")
        if r.p_requested > 1 and r.speedup < 1.0:
            f.append("slowdown_robust" if r.speedup_hi < 1.0 else "slowdown")
        if max(r.iqr_rel, r.seq_iqr_rel) > HIGH_DISPERSION:
            f.append("high_dispersion")
        if min(r.reps, r.seq_reps) < expected_reps:
            f.append("incomplete_reps")
        if r.oversubscribed:
            f.append("oversubscribed")
        flags.append(";".join(f))
    omp["flags"] = flags
    cols = ["n", "p_requested", "reps", "seq_reps", "t_seq_median_s", "median_s",
            "q1_s", "q3_s", "iqr_rel", "speedup", "speedup_lo", "speedup_hi", "efficiency",
            "speedup_vs_omp1", "gflops", "flags"]
    return omp[cols].sort_values(["n", "p_requested"]).reset_index(drop=True)


def overhead_table(st):
    """T_omp(1) / T_seq per n, with interquartile bounds."""
    seq = st[st["method"] == "seq"].set_index("n")
    omp1 = st[(st["method"] == "omp") & (st["p_requested"] == 1)].set_index("n")
    common = seq.index.intersection(omp1.index)
    out = pd.DataFrame({
        "n": common,
        "t_seq_median_s": seq.loc[common, "median_s"].to_numpy(),
        "t_omp1_median_s": omp1.loc[common, "median_s"].to_numpy(),
    })
    out["overhead_ratio"] = out["t_omp1_median_s"] / out["t_seq_median_s"]
    out["ratio_lo"] = omp1.loc[common, "q1_s"].to_numpy() / seq.loc[common, "q3_s"].to_numpy()
    out["ratio_hi"] = omp1.loc[common, "q3_s"].to_numpy() / seq.loc[common, "q1_s"].to_numpy()
    out["ratio_range_includes_1"] = (out["ratio_lo"] <= 1.0) & (out["ratio_hi"] >= 1.0)
    return out


def naive_table(st):
    """Cost of the i-j-k reference relative to the i-k-j sequential kernel."""
    seq = st[st["method"] == "seq"].set_index("n")
    nai = st[st["method"] == "naive"].set_index("n")
    common = seq.index.intersection(nai.index)
    out = pd.DataFrame({
        "n": common,
        "t_naive_median_s": nai.loc[common, "median_s"].to_numpy(),
        "t_seq_median_s": seq.loc[common, "median_s"].to_numpy(),
        "gflops_naive": nai.loc[common, "gflops"].to_numpy(),
        "gflops_seq": seq.loc[common, "gflops"].to_numpy(),
    })
    out["naive_over_seq"] = out["t_naive_median_s"] / out["t_seq_median_s"]
    return out


def payoff_table(sp):
    """Per n: where parallelism starts to pay off and where it peaks."""
    rows = []
    for n, grp in sp.groupby("n"):
        grp = grp.sort_values("p_requested")
        best = grp.loc[grp["speedup"].idxmax()]
        robust = grp[(grp["p_requested"] > 1) & (grp["speedup_lo"] > 1.0)]
        near = grp[grp["speedup"] >= 0.9 * best["speedup"]]
        rows.append({
            "n": n,
            "min_p_robust_speedup_gt_1": int(robust["p_requested"].min()) if len(robust) else None,
            "best_p": int(best["p_requested"]),
            "best_speedup": best["speedup"],
            "efficiency_at_best_p": best["efficiency"],
            "min_p_within_90pct_of_best": int(near["p_requested"].min()),
        })
    return pd.DataFrame(rows)


def compare_table(st_a, st_b, label_a, label_b, method="omp"):
    """Median and IQR/median of two runs side by side for common configs."""
    a = st_a[st_a["method"] == method].set_index(["n", "p_requested"])
    b = st_b[st_b["method"] == method].set_index(["n", "p_requested"])
    common = a.index.intersection(b.index)
    out = pd.DataFrame({
        f"median_s_{label_a}": a.loc[common, "median_s"],
        f"q1_s_{label_a}": a.loc[common, "q1_s"],
        f"q3_s_{label_a}": a.loc[common, "q3_s"],
        f"iqr_rel_{label_a}": a.loc[common, "iqr_rel"],
        f"median_s_{label_b}": b.loc[common, "median_s"],
        f"q1_s_{label_b}": b.loc[common, "q1_s"],
        f"q3_s_{label_b}": b.loc[common, "q3_s"],
        f"iqr_rel_{label_b}": b.loc[common, "iqr_rel"],
    })
    out[f"ratio_{label_b}_over_{label_a}"] = (out[f"median_s_{label_b}"]
                                              / out[f"median_s_{label_a}"])
    return out.reset_index().sort_values(["n", "p_requested"])


def factorial_table(st_scalar, st_simd):
    """2x2 factorial {scalar, simd} x {seq, omp(p)} for every common (n, p)."""
    def cell(st, method):
        return st[st["method"] == method].set_index(["n", "p_requested"])

    seq_sc = cell(st_scalar, "seq").droplevel("p_requested")
    seq_si = cell(st_simd, "seq").droplevel("p_requested")
    omp_sc, omp_si = cell(st_scalar, "omp"), cell(st_simd, "omp")
    rows = []
    for n, p in omp_sc.index.intersection(omp_si.index).sort_values():
        if n not in seq_sc.index or n not in seq_si.index:
            continue
        a, b = omp_sc.loc[(n, p)], omp_si.loc[(n, p)]
        sa, sb = seq_sc.loc[n], seq_si.loc[n]
        v_seq = sa["median_s"] / sb["median_s"]
        s_scalar = sa["median_s"] / a["median_s"]
        s_comb = sa["median_s"] / b["median_s"]
        flags = []
        v_omp_lo, v_omp_hi = a["q1_s"] / b["q3_s"], a["q3_s"] / b["q1_s"]
        if v_omp_hi < 1.0:
            flags.append("simd_slower_robust")
        elif a["median_s"] / b["median_s"] < 1.0:
            flags.append("simd_slower")
        if max(a["iqr_rel"], b["iqr_rel"], sa["iqr_rel"], sb["iqr_rel"]) > HIGH_DISPERSION:
            flags.append("high_dispersion")
        rows.append({
            "n": n, "p": p,
            "t_seq_scalar_s": sa["median_s"], "t_seq_simd_s": sb["median_s"],
            "t_omp_scalar_s": a["median_s"], "t_omp_simd_s": b["median_s"],
            "V_seq": v_seq,
            "V_seq_lo": sa["q1_s"] / sb["q3_s"], "V_seq_hi": sa["q3_s"] / sb["q1_s"],
            "V_omp": a["median_s"] / b["median_s"],
            "V_omp_lo": v_omp_lo, "V_omp_hi": v_omp_hi,
            "S_scalar": s_scalar,
            "S_simd": sb["median_s"] / b["median_s"],
            "S_comb": s_comb,
            "interaction": s_comb / (v_seq * s_scalar),
            "gflops_omp_scalar": a["gflops"], "gflops_omp_simd": b["gflops"],
            "flags": ";".join(flags),
        })
    return pd.DataFrame(rows)


def reproducibility_table(runs):
    """Median / quartiles of every omp configuration in several runs, by build.

    runs: list of (label, DataFrame). The build is taken from the CSV's
    compiler_flags column (-march=native -> simd, otherwise scalar).
    """
    frames = []
    for label, df in runs:
        m = measured(df)
        m = m[m["method"] == "omp"]
        build = np.where(m["compiler_flags"].str.contains("-march=native"), "simd", "scalar")
        g = m.assign(build=build).groupby(["build", "n", "p_requested"])["time_s"]
        st = pd.DataFrame({"median_s": g.median(), "q1_s": g.quantile(0.25),
                           "q3_s": g.quantile(0.75)}).reset_index()
        st.insert(0, "run", label)
        frames.append(st)
    out = pd.concat(frames, ignore_index=True)
    # Spread of the medians across runs, relative to their mean, per build/config.
    spread = out.groupby(["build", "n", "p_requested"])["median_s"].agg(
        lambda s: (s.max() - s.min()) / s.mean())
    out["between_run_spread"] = out.set_index(["build", "n", "p_requested"]).index.map(spread)
    return out.sort_values(["n", "p_requested", "build", "run"]).reset_index(drop=True)


# ---------------------------------------------------------------- output

def fmt(value):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if value == 0 or 1e-3 <= abs(value) < 1e5:
            return f"{value:.4g}"
        return f"{value:.3e}"
    if value is None:
        return ""
    return str(value)


def write_table(df, name, title, source):
    """Write <name>.csv (full precision) and <name>.md (4 significant digits)."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    lines = [f"# {title}", "", f"Source: `{source}`", "",
             "| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    (TABLE_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=200)
    fig.savefig(FIGURE_DIR / f"{name}.pdf")
    plt.close(fig)


def mark_cores(ax, physical, logical):
    ax.axvline(physical, color="grey", ls=":", lw=1)
    ax.axvline(logical, color="grey", ls="--", lw=1)
    ax.text(physical, ax.get_ylim()[1], f" {physical} cores", va="top",
            fontsize=7, color="grey")
    ax.text(logical, ax.get_ylim()[1], f" {logical} LPs", va="top",
            fontsize=7, color="grey")


def fig_speedup(sp, physical, logical, name, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ps = sorted(sp["p_requested"].unique())
    ax.plot(ps, ps, color="black", lw=0.8, ls="-", label="ideal S = p")
    for n, grp in sp.groupby("n"):
        grp = grp.sort_values("p_requested")
        err = [grp["speedup"] - grp["speedup_lo"], grp["speedup_hi"] - grp["speedup"]]
        ax.errorbar(grp["p_requested"], grp["speedup"], yerr=err, marker="o",
                    ms=3, capsize=2, lw=1, label=f"n = {n}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    # The y range covers the ideal line up to the number of logical processors
    # and every upper bound; the ideal line beyond that leaves the plot.
    ymax = max(logical, sp["speedup_hi"].max()) * 1.05
    ax.set_ylim(0, ymax)
    ax.set_xlabel("threads p")
    ax.set_ylabel("speedup S(p) = T_seq / T_omp(p)")
    ax.set_title(title, fontsize=10)
    mark_cores(ax, physical, logical)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
    fig.text(0.01, 0.005, "bars: [Q1_seq/Q3_omp, Q3_seq/Q1_omp]; ideal line clipped "
             "at the top of the axis", fontsize=6.5, color="grey")
    ax.grid(alpha=0.3)
    save_figure(fig, name)


def fig_efficiency(sp, physical, logical):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n, grp in sp.groupby("n"):
        grp = grp.sort_values("p_requested")
        ax.plot(grp["p_requested"], grp["efficiency"], marker="o", ms=3, lw=1,
                label=f"n = {n}")
    ps = sorted(sp["p_requested"].unique())
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("threads p")
    ax.set_ylabel("efficiency E(p) = S(p) / p")
    ax.set_title("Parallel efficiency (pinned, median of 10 reps)", fontsize=10)
    mark_cores(ax, physical, logical)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    save_figure(fig, "efficiency_vs_p")


def fig_time(st):
    """Median time vs n for seq, naive and omp at every p (log-log)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in ("naive", "seq"):
        d = st[st["method"] == method].sort_values("n")
        ax.errorbar(d["n"], d["median_s"], yerr=[d["median_s"] - d["q1_s"],
                    d["q3_s"] - d["median_s"]], marker="s", ms=4, lw=1.4,
                    capsize=2, label=method)
    omp = st[st["method"] == "omp"]
    for p, d in omp.groupby("p_requested"):
        d = d.sort_values("n")
        ax.errorbar(d["n"], d["median_s"], yerr=[d["median_s"] - d["q1_s"],
                    d["q3_s"] - d["median_s"]], marker="o", ms=2.5, lw=0.8,
                    capsize=1.5, alpha=0.85, label=f"omp p={p}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ns = sorted(st["n"].unique())
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("median time [s] (bars: Q1-Q3)")
    ax.set_title("Execution time (pinned)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    save_figure(fig, "time_vs_n")


def fig_time_vs_p(st):
    """Median omp time vs p, one panel per n, with the seq time as reference."""
    ns = sorted(st[st["method"] == "omp"]["n"].unique())
    cols = 3
    rows = int(np.ceil(len(ns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 2.8 * rows), squeeze=False)
    for ax, n in zip(axes.flat, ns):
        d = st[(st["method"] == "omp") & (st["n"] == n)].sort_values("p_requested")
        ax.errorbar(d["p_requested"], d["median_s"], yerr=[d["median_s"] - d["q1_s"],
                    d["q3_s"] - d["median_s"]], marker="o", ms=3, capsize=2, lw=1,
                    label="omp")
        seq = st[(st["method"] == "seq") & (st["n"] == n)]
        if len(seq):
            ax.axhline(seq["median_s"].iloc[0], color="red", ls="--", lw=0.8, label="seq")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(d["p_requested"])
        ax.set_xticklabels([str(p) for p in d["p_requested"]], fontsize=7)
        ax.set_title(f"n = {n}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    for ax in axes.flat[len(ns):]:
        ax.set_visible(False)
    axes[0][0].legend(fontsize=7)
    fig.supxlabel("threads p", fontsize=9)
    fig.supylabel("median time [s] (bars: Q1-Q3)", fontsize=9)
    save_figure(fig, "time_vs_p")


def fig_gflops(st, physical, logical):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    omp = st[st["method"] == "omp"]
    for n, d in omp.groupby("n"):
        d = d.sort_values("p_requested")
        ax.plot(d["p_requested"], d["gflops"], marker="o", ms=3, lw=1, label=f"n = {n}")
    ps = sorted(omp["p_requested"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("threads p")
    ax.set_ylabel("GFLOP/s (2n^3 / median time)")
    ax.set_title("Achieved floating-point throughput (omp, pinned)", fontsize=10)
    mark_cores(ax, physical, logical)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    save_figure(fig, "gflops_vs_p")


def fig_compare(cmp, label_a, label_b, name, title):
    """Grouped bars of median time for two runs, one group per (n, p)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(cmp))
    w = 0.38
    for off, label in ((-w / 2, label_a), (w / 2, label_b)):
        med = cmp[f"median_s_{label}"]
        err = [med - cmp[f"q1_s_{label}"], cmp[f"q3_s_{label}"] - med]
        ax.bar(x + off, med, w, yerr=err, capsize=2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}\np={p}" for n, p in zip(cmp["n"], cmp["p_requested"])],
                       fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("median time [s] (bars: Q1-Q3)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    save_figure(fig, name)


def fig_factorial(fac):
    """(a) vectorisation speedup vs n; (b) GFLOP/s of the four factorial cells."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ns = sorted(fac["n"].unique())
    first = fac[fac["p"] == fac["p"].min()].sort_values("n")
    ax1.errorbar(first["n"], first["V_seq"], yerr=[first["V_seq"] - first["V_seq_lo"],
                 first["V_seq_hi"] - first["V_seq"]], marker="s", ms=4, capsize=2,
                 lw=1.4, label="seq")
    for i, (p, d) in enumerate(fac.groupby("p"), start=1):
        d = d.sort_values("n")
        ax1.errorbar(d["n"], d["V_omp"], yerr=[d["V_omp"] - d["V_omp_lo"],
                     d["V_omp_hi"] - d["V_omp"]], marker="o", ms=3, capsize=2, lw=1,
                     color=f"C{i}", label=f"omp p={p}")
        if "V_omp_runs_lo" in d:
            # Range over repeated runs, drawn slightly to the right of the point.
            r = d.dropna(subset=["V_omp_runs_lo"])
            ax1.vlines(r["n"] * 1.04, r["V_omp_runs_lo"], r["V_omp_runs_hi"],
                       color=f"C{i}", ls=":", lw=2)
    if "V_omp_runs_lo" in fac and fac["V_omp_runs_lo"].notna().any():
        ax1.plot([], [], color="grey", ls=":", lw=2, label="range over repeated runs")
    ax1.axhline(1.0, color="black", lw=0.8)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_xlabel("matrix size n")
    ax1.set_ylabel("V = T_scalar / T_simd")
    ax1.set_title("(a) Vectorisation speedup (bars: interquartile bounds)", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    seq = first.sort_values("n")
    ax2.plot(seq["n"], 2.0 * seq["n"].astype(float) ** 3 / seq["t_seq_scalar_s"] / 1e9,
             marker="s", ms=4, lw=1.2, color="C0", ls="--", label="seq scalar")
    ax2.plot(seq["n"], 2.0 * seq["n"].astype(float) ** 3 / seq["t_seq_simd_s"] / 1e9,
             marker="s", ms=4, lw=1.2, color="C0", label="seq simd")
    for i, (p, d) in enumerate(fac.groupby("p"), start=1):
        d = d.sort_values("n")
        ax2.plot(d["n"], d["gflops_omp_scalar"], marker="o", ms=3, lw=1, color=f"C{i}",
                 ls="--", label=f"omp p={p} scalar")
        ax2.plot(d["n"], d["gflops_omp_simd"], marker="o", ms=3, lw=1, color=f"C{i}",
                 label=f"omp p={p} simd")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("matrix size n")
    ax2.set_ylabel("GFLOP/s (2n^3 / median time)")
    ax2.set_title("(b) Throughput of the factorial cells", fontsize=9)
    ax2.legend(fontsize=6.5, ncol=2)
    ax2.grid(alpha=0.3)
    save_figure(fig, "vectorisation_factorial")


# ---------------------------------------------------------------- main

def rel(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--primary", type=Path, default=RAW_DIR / "bench_full_quiet.csv",
                    help="pinned full sweep used as primary data")
    ap.add_argument("--unpinned", type=Path, default=RAW_DIR / "bench_unpinned_quiet.csv",
                    help="unpinned contrast run (optional; '' to skip)")
    ap.add_argument("--noisy", type=Path, default=RAW_DIR / "bench_full.csv",
                    help="sweep taken under background load (optional; '' to skip)")
    ap.add_argument("--simd", type=Path, default=RAW_DIR / "bench_simd_quiet2.csv",
                    help="simd-build run of the vectorisation study (optional; '' to skip)")
    ap.add_argument("--scalar-ctrl", type=Path,
                    default=RAW_DIR / "bench_scalar_ctrl_quiet2.csv",
                    help="scalar control run measured with --simd (optional; '' to skip)")
    ap.add_argument("--vcheck", type=Path, nargs="*",
                    default=sorted(RAW_DIR.glob("bench_vcheck_*.csv")),
                    help="extra simd/scalar runs for the reproducibility check "
                         "(default: results/raw/bench_vcheck_*.csv)")
    args = ap.parse_args()

    primary = load_run(args.primary)
    datasets = {"primary": (args.primary, primary)}
    stale = {"unpinned": "pinning_contrast.*",
             "noisy": "noisy_vs_quiet.*, speedup_vs_p_noisy.*",
             "simd": "vectorisation_factorial.*",
             "scalar_ctrl": "vectorisation_factorial.*"}
    for label, path in (("unpinned", args.unpinned), ("noisy", args.noisy),
                        ("simd", args.simd), ("scalar_ctrl", args.scalar_ctrl)):
        df = load_run(path, required=False) if str(path) not in ("", ".") else None
        if df is None:
            print(f"warning: no {label} dataset; outputs from earlier runs "
                  f"({stale[label]}) were not regenerated and may be stale",
                  file=sys.stderr)
        else:
            datasets[label] = (path, df)
    for path in args.vcheck:
        df = load_run(path, required=False)
        if df is not None:
            datasets[f"vcheck:{path.stem.removeprefix('bench_')}"] = (path, df)
    sources = ", ".join(rel(p) for p, _ in datasets.values())

    # Data quality: anomalous rows, configurations without OK data, checksums.
    anom = pd.concat([anomalies(df, lab) for lab, (_, df) in datasets.items()],
                     ignore_index=True)
    write_table(anom, "anomalies", "Anomalous rows (kept in raw data; see in_statistics)",
                sources)
    missing = pd.concat([missing_configs(df, lab) for lab, (_, df) in datasets.items()],
                        ignore_index=True)
    write_table(missing, "missing_configs",
                "Configurations with no measured OK repetition (absent from statistics)",
                sources)
    issues = [i for lab, (_, df) in datasets.items() for i in checksum_issues(df, lab)]
    if "simd" in datasets and "scalar_ctrl" in datasets:
        # Both builds round identically (no FMA contraction), so checksums must match.
        cs = [measured(datasets[k][1]).groupby("n")["checksum"].first()
              for k in ("scalar_ctrl", "simd")]
        for n in cs[0].index.intersection(cs[1].index):
            if cs[0][n] != cs[1][n]:
                issues.append(f"simd vs scalar_ctrl: n={n} checksums differ")

    physical = int(primary["physical_cores"].iloc[0])
    logical = int(primary["logical_procs"].iloc[0])
    src = rel(args.primary)

    st = config_stats(primary)
    sp = speedup_table(st)
    ov = overhead_table(st)
    nv = naive_table(st)
    po = payoff_table(sp)

    stats_cols = ["method", "n", "p_requested", "reps", "median_s", "q1_s", "q3_s",
                  "iqr_s", "iqr_rel", "min_s", "max_s", "max_over_min", "mean_s",
                  "std_s", "gflops", "peak_rss_bytes", "theoretical_bytes",
                  "rss_over_theoretical"]
    write_table(st[stats_cols].sort_values(KEYS), "stats_per_config",
                "Per-configuration statistics (measured reps, status OK)", src)
    write_table(sp, "speedup_efficiency", "Speedup and efficiency (omp vs seq, medians)", src)
    write_table(ov, "omp_overhead", "OpenMP overhead at p = 1: T_omp(1) / T_seq", src)
    write_table(nv, "naive_vs_seq", "Naive i-j-k vs sequential i-k-j", src)
    write_table(po, "parallel_payoff", "When parallelism pays off (per n)", src)
    flagged = sp[sp["flags"] != ""][["n", "p_requested", "speedup", "speedup_lo",
                                      "speedup_hi", "efficiency", "iqr_rel", "flags"]]
    write_table(flagged, "flags", "Flagged configurations (superlinear, slowdown, "
                "high dispersion, oversubscribed)", src)

    # Wide speedup matrix (n x p), handy for the report.
    wide = sp.pivot(index="n", columns="p_requested", values="speedup")
    wide.columns = [f"p={c}" for c in wide.columns]
    write_table(wide.reset_index(), "speedup_matrix", "Speedup S(p) matrix (rows n, columns p)", src)

    fig_speedup(sp, physical, logical, "speedup_vs_p",
                "Speedup vs threads (pinned; bars: interquartile bounds)")
    fig_efficiency(sp, physical, logical)
    fig_time(st)
    fig_time_vs_p(st)
    fig_gflops(st, physical, logical)

    if "unpinned" in datasets:
        path_u, df_u = datasets["unpinned"]
        cmp = compare_table(st, config_stats(df_u), "pinned", "unpinned")
        write_table(cmp, "pinning_contrast", "Pinned (cores) vs unpinned, omp",
                    f"{src}, {rel(path_u)}")
        fig_compare(cmp, "pinned", "unpinned", "pinning_contrast",
                    "Thread pinning contrast (omp, quiet machine)")

    if "noisy" in datasets:
        path_n, df_n = datasets["noisy"]
        st_n = config_stats(df_n)
        frames = [compare_table(st, st_n, "quiet", "noisy", method=m).assign(method=m)
                  for m in ("seq", "omp", "naive")]
        cmpn = pd.concat(frames, ignore_index=True)
        cmpn = cmpn[["method"] + [c for c in cmpn.columns if c != "method"]]
        write_table(cmpn, "noisy_vs_quiet",
                    "Background-load run vs quiet run (same configurations)",
                    f"{src}, {rel(path_n)}")
        sp_n = speedup_table(st_n)
        fig_speedup(sp_n, physical, logical, "speedup_vs_p_noisy",
                    "Speedup under background load (run 'full', not primary data)")

    fac = None
    if "simd" in datasets and "scalar_ctrl" in datasets:
        (path_si, df_si), (path_sc, df_sc) = datasets["simd"], datasets["scalar_ctrl"]
        fac = factorial_table(config_stats(df_sc), config_stats(df_si))
        fac_src = f"{rel(path_sc)}, {rel(path_si)}"

        checks = [(lab.split(":", 1)[1], df) for lab, (_, df) in datasets.items()
                  if lab.startswith("vcheck:")]
        if checks:
            runs = [(path_si.stem.removeprefix("bench_"), df_si),
                    (path_sc.stem.removeprefix("bench_"), df_sc)] + checks
            rep = reproducibility_table(runs)
            # Only the configurations that the extra runs re-measured.
            keys = set().union(*(set(zip(measured(df)["n"], measured(df)["p_requested"]))
                                 for _, df in checks))
            rep = rep[[k in keys for k in zip(rep["n"], rep["p_requested"])]]
            write_table(rep, "vectorisation_reproducibility",
                        "Vectorisation study: omp medians across repeated runs",
                        ", ".join(rel(p) for lab, (p, _) in datasets.items()
                                  if lab in ("simd", "scalar_ctrl") or lab.startswith("vcheck:")))
            # Range of V_omp over all runs: [min scalar / max simd, max scalar / min simd].
            med = rep.groupby(["n", "p_requested", "build"])["median_s"].agg(["min", "max"])
            v_runs = {}
            for (n, p), g in med.groupby(level=["n", "p_requested"]):
                g = g.droplevel(["n", "p_requested"])
                if {"scalar", "simd"} <= set(g.index):
                    v_runs[(n, p)] = (g.loc["scalar", "min"] / g.loc["simd", "max"],
                                      g.loc["scalar", "max"] / g.loc["simd", "min"])
            fac["V_omp_runs_lo"] = [v_runs.get((n, p), (np.nan, np.nan))[0]
                                    for n, p in zip(fac["n"], fac["p"])]
            fac["V_omp_runs_hi"] = [v_runs.get((n, p), (np.nan, np.nan))[1]
                                    for n, p in zip(fac["n"], fac["p"])]
            unstable = (fac["V_omp_runs_lo"] < 1.0) & (fac["V_omp_runs_hi"] > 1.0)
            fac.loc[unstable, "flags"] = [";".join(filter(None, [f, "unstable_across_runs"]))
                                          for f in fac.loc[unstable, "flags"]]
            fac_src += ", " + ", ".join(rel(p) for lab, (p, _) in datasets.items()
                                        if lab.startswith("vcheck:"))

        write_table(fac, "vectorisation_factorial",
                    "Vectorisation 2x2 factorial (scalar vs simd build, seq vs omp)", fac_src)
        fig_factorial(fac)

    # Console summary.
    print(f"primary: {src}  ({len(measured(primary))} measured OK rows, "
          f"{physical} cores / {logical} LPs)")
    print(f"anomalous rows: {len(anom)} "
          f"({int(anom['in_statistics'].sum())} of them enter the statistics)")
    print(f"configurations without OK data: {len(missing)}")
    for r in missing.itertuples():
        print(f"  {r.dataset}: {r.method} n={r.n} p={r.p_requested}")
    print("checksum issues: " + ("; ".join(issues) if issues else "none"))
    print(f"flagged omp configurations: {len(flagged)}")
    for r in flagged.itertuples():
        print(f"  n={r.n:5d} p={r.p_requested:2d} S={r.speedup:6.2f} "
              f"[{r.speedup_lo:.2f}, {r.speedup_hi:.2f}]  {r.flags}")
    if fac is not None:
        print("vectorisation factorial (V = T_scalar / T_simd):")
        for r in fac.itertuples():
            print(f"  n={r.n:5d} p={r.p:2d} V_seq={r.V_seq:5.2f} V_omp={r.V_omp:5.2f} "
                  f"[{r.V_omp_lo:.2f}, {r.V_omp_hi:.2f}] S_comb={r.S_comb:6.2f}  {r.flags}")
    print(f"tables  -> {rel(TABLE_DIR)}")
    print(f"figures -> {rel(FIGURE_DIR)}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
