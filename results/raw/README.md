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
| `simd_quiet` | Vectorisation study, simd build (`build/simd/bench.exe`; seq/omp, n = 512..3072, p = 1, 8, 16) | cores | **web browser left open, heavy load** (see below) | No — contaminated, kept as recorded |
| `scalar_ctrl_quiet` | Vectorisation study, scalar control (same configurations, `build/bench.exe`) | cores | **web browser left open, heavy load** | No — contaminated, kept as recorded |
| `simd_quiet2` | Vectorisation study, simd build (repeat of `simd_quiet`) | cores | background applications closed (`background_load_simd2.txt`) | **Yes** (vectorisation factorial) |
| `scalar_ctrl_quiet2` | Vectorisation study, scalar control (repeat of `scalar_ctrl_quiet`), run right after `simd_quiet2` | cores | background applications closed | **Yes** (vectorisation factorial) |
| `vcheck_simd_a`, `vcheck_scalar_a`, `vcheck_simd_b`, `vcheck_scalar_b` | Reproducibility check (omp, n = 2048, 3072; p = 8, 16), simd and scalar alternated a, a, b, b | cores | background applications closed | **Yes** (reproducibility of the vectorisation effect) |

The `_quiet` suffix of the two vectorisation runs reflects the intended
protocol; the conditions actually met are described below.

## Annotation: background load during `full` and `unpinned`

During the first attempt, interactive applications (web browser, mail client,
music player, editor, peripheral software) were consuming roughly 1.3-2 logical
processors. The effect is visible in the data: e.g. `seq`, n = 1024 has
repetitions ranging from 0.62 s to 4.36 s, whereas the same configuration in
`full_quiet` ranges from 0.318 s to 0.365 s. Those applications were closed
and both sweeps were repeated as `full_quiet` / `unpinned_quiet`. The
background load measured immediately before the quiet runs is recorded in
`background_load_quiet.txt`.

## Annotation: background load during `simd_quiet` and `scalar_ctrl_quiet`

The web browser could not be closed for these runs. Immediately before them,
background processes consumed 14.74 CPU-seconds per 5 s (about 3 logical
processors; one browser process alone 8.8), see `background_load_simd.txt`.
The load clearly varied during the runs: the scalar control is 1.0-5.5x slower
than the same configurations in `full_quiet` (e.g. `seq`, n = 3072: median
24.99 s vs 12.24 s), and within the scalar run itself `seq` n = 3072 (24.99 s)
is twice as slow as `omp` p = 1 n = 3072 (12.74 s), although both execute the
same kernel and agree within 1% in `full_quiet`. Differences of this size
reflect when each configuration ran, not the code. Checksums are consistent (one per
n, identical between the two builds), so the rows are valid measurements of a
loaded machine, but they are not used to quantify the effect of
vectorisation.

## Annotation: reproducibility check of the vectorisation study

In `simd_quiet2`, omp n = 2048 p = 8 was slower with the simd build than with
the scalar one (median 0.802 s vs 0.627 s, interquartile bounds of the ratio
entirely below 1). Because this contradicted the other configurations, both
builds were re-measured twice, alternating (`vcheck_*`). The simd medians for
that configuration were 0.555 s and 0.452 s, against 0.624 s and 0.597 s for
the scalar build, so the slowdown did not reproduce. All rows are kept;
`scripts/analyze.py` reports the per-run medians in
`results/tables/vectorisation_reproducibility.*` and flags such configurations
`unstable_across_runs` in `vectorisation_factorial.*`.

## Annotation: outliers in `full_quiet`

`omp`, n = 128, p = 16 has a max/min ratio of ~238 (IQR/median ~34): with all
16 logical processors pinned, any remaining background activity preempts one
pinned thread for a scheduler quantum, which is orders of magnitude longer than
the ~0.1 ms kernel. The median is unaffected; the rows are kept unchanged.
