"""Run the benchmark sweep (n x p x method) with build/bench.exe.

Every configuration (method, n, p) runs in its own bench process, which writes
one CSV row per repetition into its own part file
(results/raw/bench_<tag>_parts/cfg_<index>_<method>_n<n>_p<p>.csv, created by
bench.exe itself). At the end the runner concatenates all parts, in plan order,
into results/raw/bench_<tag>.csv; the part files are kept as well. (One file
per process also avoids a Windows protection on this machine that prevents the
unsigned bench.exe from modifying files created by another program.)
This runner:
  - sets the threading environment for every process (OMP_DYNAMIC=false,
    OMP_NUM_THREADS=p, OMP_PROC_BIND, OMP_PLACES, OPENBLAS_NUM_THREADS=1,
    MKL_NUM_THREADS=1) and records it in a metadata JSON file;
  - visits n in ascending order and, within each n, runs the (method, p)
    configurations in a random order drawn from a recorded seed (blocked
    randomisation; ascending n is needed for the time-budget rule below);
  - enforces the time budget: bench.exe marks a repetition slower than the
    budget as TIMEOUT and stops; the runner additionally kills a process that
    exceeds a hard wall-clock limit and records a TIMEOUT row itself. After a
    TIMEOUT, the same (method, p) is not run for larger n: a
    SKIPPED_AFTER_TIMEOUT row is recorded instead (no time is invented);
  - never deletes or filters measurements.

Usage:
    python scripts/run_benchmarks.py --quick                       # smoke sweep
    python scripts/run_benchmarks.py                               # full sweep
    python scripts/run_benchmarks.py --dry-run --estimate-from results/raw/<quick>.csv
"""

import argparse
import csv
import datetime as dt
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "build" / ("bench.exe" if os.name == "nt" else "bench")
RAW_DIR = ROOT / "results" / "raw"

FULL_SIZES = [128, 256, 512, 1024, 2048, 3072]
FULL_THREADS = [1, 2, 4, 6, 8, 12, 16, 32]
QUICK_SIZES = [128, 256, 512, 1024]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke sweep: n=128..1024, reps=3, warmup=1")
    ap.add_argument("--sizes", type=int, nargs="+")
    ap.add_argument("--threads", type=int, nargs="+", default=FULL_THREADS)
    ap.add_argument("--methods", nargs="+", default=["seq", "omp", "naive"],
                    choices=["seq", "omp", "naive"])
    ap.add_argument("--naive-max-n", type=int, default=1024,
                    help="largest n for the optional slow naive reference")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--time-budget", type=float, default=300.0,
                    help="seconds per repetition")
    ap.add_argument("--mem-budget", type=int, default=2 << 30, help="bytes")
    ap.add_argument("--pin", choices=["none", "cores"], default="cores")
    ap.add_argument("--proc-bind", default="close", help="OMP_PROC_BIND value")
    ap.add_argument("--places", default="cores", help="OMP_PLACES value")
    ap.add_argument("--order-seed", type=int, default=20251113,
                    help="seed of the randomised configuration order")
    ap.add_argument("--seed", type=int, default=20250301, help="input matrix seed")
    ap.add_argument("--input-dir", default=str(ROOT / "data"))
    ap.add_argument("--bench-exe", type=Path, default=BENCH,
                    help="benchmark executable (e.g. build/simd/bench.exe for the "
                         "vectorised variant)")
    ap.add_argument("--tag", default=None, help="name of the run (default: timestamp)")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                    help="output directory for the raw CSV, metadata and log")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (and estimate) without running")
    ap.add_argument("--estimate-from", type=Path,
                    help="raw CSV used to extrapolate the duration (T ~ n^3)")
    args = ap.parse_args()

    if args.sizes is None:
        args.sizes = QUICK_SIZES if args.quick else FULL_SIZES
    if args.reps is None:
        args.reps = 3 if args.quick else 10
    if args.warmup is None:
        args.warmup = 1 if args.quick else 2
    if not args.quick and args.reps < 5:
        ap.error("the protocol requires at least 5 measured repetitions")
    args.sizes = sorted(set(args.sizes))
    return args


def build_plan(args):
    """Returns the ordered list of (n, method, p) and the per-n shuffles."""
    rng = random.Random(args.order_seed)
    plan = []
    for n in args.sizes:
        block = []
        for method in args.methods:
            if method == "omp":
                block += [(n, "omp", p) for p in args.threads]
            elif method == "naive":
                if n <= args.naive_max_n:
                    block.append((n, "naive", 1))
            else:
                block.append((n, method, 1))
        rng.shuffle(block)
        plan += block
    return plan


def load_estimates(path):
    """Median measured time per (method, p, n) from an earlier raw CSV."""
    import statistics
    times = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "OK" or row["is_warmup"] != "0":
                continue
            key = (row["method"], int(row["p_requested"]), int(row["n"]))
            times.setdefault(key, []).append(float(row["time_s"]))
    return {k: statistics.median(v) for k, v in times.items()}


def estimate_seconds(plan, args, medians):
    """Extrapolates each configuration from its largest measured n with T ~ n^3."""
    total, missing = 0.0, []
    per_config = []
    for n, method, p in plan:
        known = [(kn, t) for (m, kp, kn), t in medians.items() if m == method and kp == p]
        if not known:
            missing.append((n, method, p))
            continue
        n0, t0 = max(known)
        t_rep = t0 * (n / n0) ** 3
        cost = t_rep * (args.reps + args.warmup) + 0.3  # + process start-up
        per_config.append((n, method, p, t_rep, cost))
        total += cost
    return total, per_config, missing


def write_marker_row(path, header, base):
    """Writes a part file with one row produced by the runner (TIMEOUT kill / skip)."""
    row = {h: "" for h in header}
    row.update(base)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def merge_parts(parts_dir, out_csv, header):
    """Concatenates every part file (plan order) into out_csv. Returns the row count."""
    def key(path):
        return (int(path.name.split("_")[1]), path.name)  # cfg_<index>_...
    expected = ",".join(header)
    rows = 0
    with open(out_csv, "w", newline="") as out:
        out.write(expected + "\n")
        for part in sorted(parts_dir.glob("cfg_*.csv"), key=key):
            lines = part.read_text().splitlines()
            if not lines:
                continue
            if lines[0] != expected:
                raise RuntimeError(f"{part.name}: header differs from bench.exe's schema")
            for line in lines[1:]:
                if line:
                    out.write(line + "\n")
                    rows += 1
    return rows


def main():
    args = parse_args()
    plan = build_plan(args)
    tag = args.tag or ("quick_" if args.quick else "full_") + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.raw_dir / f"bench_{tag}.csv"
    parts_dir = args.raw_dir / f"bench_{tag}_parts"
    if not args.dry_run and parts_dir.exists() and any(parts_dir.glob("cfg_*.csv")):
        sys.exit(f"tag already used ({parts_dir}): choose another --tag")
    # bench.exe uses the classic Win32 file API, limited to MAX_PATH (260) characters.
    longest = max((len(str(parts_dir / f"cfg_{i:03d}_{m}_n{n}_p{p}_timeout.csv"))
                   for i, (n, m, p) in enumerate(plan)), default=0)
    if os.name == "nt" and longest >= 255:
        sys.exit(f"part file paths would reach {longest} characters (Windows MAX_PATH is 260): "
                 f"use a shorter --raw-dir or --tag")
    meta_path = args.raw_dir / f"bench_{tag}_meta.json"
    log_path = args.raw_dir / f"bench_{tag}_stderr.log"

    print(f"Plan: {len(plan)} configurations, sizes={args.sizes}, threads={args.threads}, "
          f"methods={args.methods}, reps={args.reps}, warmup={args.warmup}, "
          f"order seed={args.order_seed}")
    if args.estimate_from:
        medians = load_estimates(args.estimate_from)
        total, per_config, missing = estimate_seconds(plan, args, medians)
        print(f"Estimated duration: {total / 60:.1f} min ({total:.0f} s) extrapolated from "
              f"{args.estimate_from.name} with T ~ n^3")
        by_n = {}
        for n, _, _, _, cost in per_config:
            by_n[n] = by_n.get(n, 0.0) + cost
        for n in sorted(by_n):
            print(f"  n={n:<5} ~{by_n[n] / 60:6.1f} min")
        slow = [c for c in per_config if c[3] > args.time_budget]
        for n, m, p, t_rep, _ in slow:
            print(f"  predicted TIMEOUT: {m} p={p} n={n} (~{t_rep:.0f} s per repetition)")
        if missing:
            print(f"  no estimate for {len(missing)} configurations: {missing[:5]}...")
    if args.dry_run:
        for i, (n, m, p) in enumerate(plan):
            print(f"  [{i:3d}] n={n:<5} {m:<5} p={p}")
        return 0

    bench = args.bench_exe.resolve()
    if not bench.exists():
        sys.exit(f"{bench} not found; build first")
    parts_dir.mkdir(parents=True, exist_ok=True)
    Path(args.input_dir).mkdir(parents=True, exist_ok=True)

    base_env = dict(os.environ)
    base_env.update({
        "OMP_DYNAMIC": "false",
        "OMP_PROC_BIND": args.proc_bind,
        "OMP_PLACES": args.places,
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    git_rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    meta = {
        "tag": tag,
        "start_local": dt.datetime.now().isoformat(timespec="seconds"),
        "git_revision": git_rev,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {k: base_env[k] for k in
                        ("OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
                         "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "notes": [
            "OMP_NUM_THREADS is set to p for every process.",
            "libgomp (MinGW-w64) ignores OMP_PROC_BIND/OMP_PLACES on Windows; the "
            "proc_bind/places CSV columns are the requested values only. The affinity "
            "actually applied is given by the pin_mode and pinned_lps columns "
            "(bench.exe --pin, SetThreadAffinityMask).",
            "Order: ascending n; (method, p) shuffled within each n with order_seed.",
        ],
        "plan": [{"index": i, "n": n, "method": m, "p": p} for i, (n, m, p) in enumerate(plan)],
        "events": [],
    }

    # The CSV schema comes from bench.exe itself, so marker rows always match it.
    header = subprocess.run([str(bench), "--print-csv-header"], capture_output=True,
                            text=True, check=True).stdout.strip().split(",")
    logical = os.cpu_count() or 0

    # Hard wall-clock limit per process: every repetition at the budget + slack.
    hard_limit = args.time_budget * (args.reps + args.warmup) + 120
    timed_out = set()  # (method, p) that hit the budget
    t_start = time.monotonic()

    log = open(log_path, "a", encoding="utf-8")
    try:
        for i, (n, method, p) in enumerate(plan):
            common = {"run_id": tag, "config_index": i, "n": n, "method": method,
                      "dtype": "double", "p_requested": p, "seed": args.seed,
                      "pin_mode": args.pin, "proc_bind": args.proc_bind,
                      "places": args.places, "time_budget_s": args.time_budget,
                      "logical_procs": logical, "oversubscribed": int(logical > 0 and p > logical)}
            part = parts_dir / f"cfg_{i:03d}_{method}_n{n}_p{p}.csv"
            if (method, p) in timed_out:
                print(f"[{i + 1}/{len(plan)}] n={n} {method} p={p}: SKIPPED_AFTER_TIMEOUT")
                write_marker_row(part, header, {**common, "status": "SKIPPED_AFTER_TIMEOUT",
                                                   "timestamp": utc_now()})
                meta["events"].append({"index": i, "event": "SKIPPED_AFTER_TIMEOUT"})
                continue

            env = dict(base_env, OMP_NUM_THREADS=str(p))
            cmd = [str(bench), "--method", method, "--n", str(n), "--threads", str(p),
                   "--reps", str(args.reps), "--warmup", str(args.warmup),
                   "--seed", str(args.seed), "--pin", args.pin,
                   "--input-dir", args.input_dir, "--time-budget", str(args.time_budget),
                   "--mem-budget", str(args.mem_budget), "--out", str(part),
                   "--run-id", tag, "--config-index", str(i)]
            t0 = time.monotonic()
            log.write(f"=== [{i}] {' '.join(cmd)}\n")
            log.flush()
            try:
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=log, timeout=hard_limit)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = "KILLED"
                write_marker_row(part.with_name(part.stem + "_killed.csv"), header, {**common, "status": "TIMEOUT",
                                                   "timestamp": utc_now()})
            wall = time.monotonic() - t0
            if rc not in (0, 3, "KILLED"):
                # Failure must be visible in the CSV, not only in the log/metadata.
                write_marker_row(part.with_name(part.stem + "_error.csv"), header,
                                 {**common, "status": f"ERROR_RC{rc}", "timestamp": utc_now()})
            if rc == 3 and count_data_rows(part) == 0:
                write_marker_row(part.with_name(part.stem + "_timeout.csv"), header,
                                 {**common, "status": "TIMEOUT", "timestamp": utc_now()})
            if rc in (3, "KILLED"):
                timed_out.add((method, p))
            status = {0: "ok", 3: "TIMEOUT", "KILLED": "TIMEOUT (killed)"}.get(rc, f"ERROR rc={rc}")
            meta["events"].append({"index": i, "returncode": rc, "wall_s": round(wall, 3)})
            print(f"[{i + 1}/{len(plan)}] n={n:<5} {method:<5} p={p:<3} {status:<8} "
                  f"{wall:7.2f} s  (elapsed {(time.monotonic() - t_start) / 60:.1f} min)")
    except KeyboardInterrupt:
        meta["events"].append({"event": "INTERRUPTED_BY_USER"})
        print("Interrupted: merging the parts measured so far.")
    finally:
        log.close()
        # Always merge and record metadata, even after an interruption.
        rows = merge_parts(parts_dir, out_csv, header)
        meta["end_local"] = dt.datetime.now().isoformat(timespec="seconds")
        meta["total_wall_s"] = round(time.monotonic() - t_start, 1)
        meta["merged_rows"] = rows
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Raw CSV: {out_csv} ({rows} rows)\nParts: {parts_dir}\n"
              f"Metadata: {meta_path}\nStderr log: {log_path}")
        report_anomalies(out_csv)
    return 0


def count_data_rows(path):
    """Number of data rows (excluding the header) in a part file; 0 if missing."""
    if not path.exists():
        return 0
    return max(0, sum(1 for line in path.read_text().splitlines() if line) - 1)


def report_anomalies(out_csv):
    """Warns about rows whose status is not OK / TIMEOUT / SKIPPED_*."""
    with open(out_csv, newline="") as f:
        statuses = [row["status"] for row in csv.DictReader(f)]
    bad = [s for s in statuses
           if s not in ("OK", "TIMEOUT") and not s.startswith("SKIPPED")]
    if bad:
        counts = {s: bad.count(s) for s in sorted(set(bad))}
        print(f"WARNING: {len(bad)} anomalous rows in {out_csv.name}: {counts}")
    else:
        print("All rows have status OK, TIMEOUT or SKIPPED_*.")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    sys.exit(main())
