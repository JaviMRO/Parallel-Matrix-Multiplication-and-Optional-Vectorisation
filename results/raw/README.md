# Raw benchmark runs

Every file here is kept exactly as produced by `scripts/run_benchmarks.py` /
`build/bench.exe`; no row has been deleted or edited. Each run has a merged CSV
(`bench_<tag>.csv`), the per-configuration part files (`bench_<tag>_parts/`),
the run metadata (`bench_<tag>_meta.json`) and the captured stderr
(`bench_<tag>_stderr.log`).

| Tag | Purpose | Pinning | Conditions | Used as primary data? |
|---|---|---|---|---|
| `quick_smoke` | Harness smoke test (n = 128..1024, 3 reps, 1 warm-up) | cores | normal desktop use | No (harness check only) |
| `full` | Full sweep, first attempt | cores | **heavy background load** (see below) | No — kept as an interference example |
| `unpinned` | Unpinned contrast, first attempt | none | **heavy background load** | No — kept as an interference example |
| `full_quiet` | Full sweep (n = 128..3072, p = 1..32, 2 warm-ups + 10 reps) | cores | background applications closed | **Yes** |
| `unpinned_quiet` | Unpinned contrast (omp, n = 1024, 2048; p = 4, 8, 16, 32) | none | background applications closed | **Yes** (pinning contrast) |

## Annotation: background load during `full` and `unpinned`

During the first attempt, interactive applications (web browser, mail client,
music player, editor, peripheral software) were consuming roughly 1.3-2 logical
processors. The effect is visible in the data: e.g. `seq`, n = 1024 has
repetitions ranging from 0.62 s to 4.36 s, whereas the same configuration in
`full_quiet` ranges from 0.318 s to 0.365 s. Those applications were closed
and both sweeps were repeated as `full_quiet` / `unpinned_quiet`. The
background load measured immediately before the quiet runs is recorded in
`background_load_quiet.txt`.

## Annotation: outliers in `full_quiet`

`omp`, n = 128, p = 16 has a max/min ratio of ~238 (IQR/median ~34): with all
16 logical processors pinned, any remaining background activity preempts one
pinned thread for a scheduler quantum, which is orders of magnitude longer than
the ~0.1 ms kernel. The median is unaffected; the rows are kept unchanged.
