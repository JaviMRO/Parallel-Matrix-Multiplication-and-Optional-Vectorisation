# Parallel Matrix Multiplication and Optional Vectorisation

Big Data – Assignment 3. Study of how the number of OpenMP threads `p` affects the
performance of dense, shared-memory matrix multiplication `C = A·B` (`double`,
row-major, contiguous 1D arrays), and of when parallelisation pays off. An
optional part measures auto-vectorisation (AVX2) as a 2×2 factorial with
threading.

The report is in `report/report.md`. Every number in it comes from the CSV
files in `results/raw/` through `scripts/analyze.py`.

## Contents

1. [Repository structure](#repository-structure)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Kernels and correctness](#kernels-and-correctness)
5. [Benchmark protocol](#benchmark-protocol)
6. [Reproducing the experiments](#reproducing-the-experiments)
7. [Which data feed which result](#which-data-feed-which-result)
8. [Environment-specific notes](#environment-specific-notes)
9. [Packaging the materials](#packaging-the-materials)

## Repository structure

```
.
├── src/
│   ├── matmul.h            kernel contract + shared i-k-j row kernel (static inline)
│   ├── matmul_seq.c        sequential kernel (refuses to compile with -fopenmp)
│   ├── matmul_omp.c        OpenMP kernel: parallel for schedule(static) over rows of C
│   ├── matmul_naive.c      naive i-j-k reference (slow; correctness/contrast only)
│   ├── inputs.c/.h         SplitMix64 generator, aligned allocation, MATMULv1 binary I/O
│   ├── timing.c/.h         monotonic timer (QueryPerformanceCounter), peak memory
│   ├── affinity.c/.h       CPU topology query and per-thread pinning (Windows API)
│   ├── bench.c             benchmark harness: one (method, n, p) configuration -> CSV rows
│   └── test.c              correctness tests (+ --dump mode used by validate_ref.py)
├── scripts/
│   ├── collect_env.ps1         hardware/software environment -> results/environment.txt
│   ├── run_benchmarks.py       sweep runner (randomised order, budgets, metadata)
│   ├── validate_ref.py         independent check against numpy (single-threaded BLAS)
│   ├── analyze.py              statistics, tables and figures
│   └── collect_vec_evidence.ps1 vectoriser reports, disassembly, scalar-vs-simd bitwise check
├── data/                   generated input matrices (*.bin, git-ignored, regenerated from the seed)
├── results/
│   ├── environment.txt
│   ├── correctness/        test logs, numpy validation, ThreadSanitizer note
│   ├── raw/                raw per-repetition CSVs + metadata (never filtered; see raw/README.md)
│   ├── tables/             CSV + Markdown tables written by analyze.py
│   ├── figures/            PNG + PDF figures written by analyze.py
│   └── vectorisation_evidence/  optional SIMD part: compiler reports, disassembly, checks
├── report/report.md        the report
├── build.ps1               build script actually used (Windows PowerShell)
├── Makefile                mirror of build.ps1 for GNU make (not tested: make unavailable here)
└── README.md
```

`build/` (binaries) is git-ignored and recreated by `build.ps1`.

## Requirements

Measured machine (full details in `results/environment.txt`):

| Item | Value |
|---|---|
| CPU | Intel Core i7-10700K, 8 cores / 16 logical processors (SMT), 3.8 GHz base |
| Caches | L1d 32 KiB and L2 256 KiB per core, L3 16 MiB shared |
| RAM | 15.85 GiB |
| OS | Windows 11 Pro 10.0.26300, power plan "High performance" |
| Compiler | GCC 14.2.0, MSYS2 UCRT64, OpenMP 4.5 (`libgomp`) |
| Python | 3.10.2, numpy 2.2.3, pandas 2.3.3, matplotlib 3.10.0 |

To reproduce you need GCC with OpenMP (the scripts assume MSYS2 UCRT64 in
`C:\msys64\ucrt64\bin`; pass `-GccDir` to change it) and Python 3.10+ with
numpy, pandas and matplotlib. The Windows-specific parts are thread pinning
(`affinity.c`) and the PowerShell scripts; on POSIX the kernels, tests and
benchmark build with the Makefile but pinning is not implemented (`--pin none`
only).

## Quick start

All commands are run from the repository root in Windows PowerShell.

```powershell
# 1. Build (scalar variant, used for the main experiment)
powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1

# 2. Correctness (test.exe prints to stdout; the committed log was saved with *>)
build\test.exe *> results\correctness\test_output.txt
python scripts/validate_ref.py

# 3. A short smoke sweep (n <= 1024, 3 reps), then the analysis of the committed data.
#    --raw-dir keeps the smoke output out of the repository; without it the files
#    are written to results/raw/ and leave the git working tree dirty.
python scripts/run_benchmarks.py --quick --tag my_smoke --raw-dir "$env:TEMP\smoke_raw"
python scripts/analyze.py
```

## Kernels and correctness

- **Loop order i-k-j.** For every row `i` of C, `C[i][:]` is zeroed and then
  updated with `A[i][k] * B[k][:]` for `k = 0..n-1`. The innermost loop walks
  rows of B and C contiguously (stride 1). The sequential and OpenMP versions
  call the *same* `static inline` row function (`matmul.h`), so they perform
  exactly the same floating-point operations in the same order.
- **Parallelisation.** `#pragma omp parallel for schedule(static)` over the rows
  of C. Each thread writes a disjoint block of rows; A and B are read-only, so
  there is no data race and no reduction.
- **Compiler flags (main experiment).** `-O3 -fno-tree-vectorize -std=c11`
  (no `-march`), identical for every unit; `-fopenmp` only where OpenMP is used.
  Executables are linked with `-static` so that they do not depend on the MSYS2
  DLL directory being on `PATH`.
- **Inputs.** A and B hold values in [0, 1) from a SplitMix64 stream (default
  seed `20250301`); the same seed always produces bit-identical matrices.
- **Tests** (`src/test.c`; log in `results/correctness/test_output.txt`):
  hand-computed 2×2/3×3 cases, identity and zero matrices, random matrices
  with n = 1, 7, 64, 100, 257 against a long-double reference
  (`|x − y| ≤ atol + rtol·|y|`, `rtol = atol = 1e-12`), the naive kernel, the
  OpenMP kernel for p = 1, 2, 3, 4, 8, 16, 32, the actual team size, input
  determinism, file round-trips and n = 0. Result: 150 checks, 0 failures; the
  OpenMP result is bitwise identical to the sequential one for every n and p.
- **Independent reference** (`scripts/validate_ref.py`; log in
  `results/correctness/validate_ref.txt`): numpy `A @ B` with OpenBLAS forced
  to one thread, for n = 64, 257, 512. The script also re-implements the
  generator in Python and checks that the inputs are bit-identical.
- **Data races.** ThreadSanitizer is not available with MinGW-w64 (libtsan is
  not shipped; see `results/correctness/tsan_unavailable.txt`). The evidence
  against races is the disjoint-rows design plus the bitwise equality of
  sequential and OpenMP results for many n and p.

## Benchmark protocol

`build\bench.exe` measures one configuration and writes one CSV row per
repetition; `scripts/run_benchmarks.py` drives the sweep.

- **Timed region.** Only the kernel call, measured with
  `QueryPerformanceCounter`. Input generation, allocation, zeroing, the
  checksum of C (which consumes the result) and all checks run outside it.
- **Repetitions.** 2 warm-ups (recorded, flagged `is_warmup = 1`) + 10 measured
  repetitions per configuration. Statistics use the median and the
  interquartile range of the 10 measured repetitions.
- **Thread control.** `OMP_DYNAMIC=false`, `OMP_NUM_THREADS=p`, and
  `omp_set_dynamic(0)`. Before every repetition the team size is checked
  inside a parallel region (`p_actual`) and a mismatch is recorded as status
  `P_MISMATCH`.
- **Pinning.** libgomp on Windows ignores `OMP_PROC_BIND` / `OMP_PLACES`
  ("Affinity not supported on this configuration"), so `bench.exe --pin cores`
  pins each OpenMP thread with `SetThreadAffinityMask`. The order is one
  logical processor per physical core first (0, 2, …, 14), then the SMT
  siblings (1, 3, …, 15); for p > 16 the order wraps around. Before every
  repetition the harness checks that every thread is the same OS thread and
  runs on its processor (`affinity_ok`).
- **Order.** Sizes run in ascending order; within each n the (method, p)
  configurations are shuffled with a recorded seed (`--order-seed 20251113`).
- **Budgets.** 300 s per repetition (`TIMEOUT`, after which larger n for that
  method/p are skipped) and 2 GiB for the three matrices
  (`SKIPPED_MEMORY_BUDGET`). No budget was reached in the reported runs.
- **Recorded per row.** Timestamp, configuration, requested and actual threads,
  topology, pinning, status, time, peak working set, theoretical memory
  (3·8·n² bytes), checksum, seed, compiler and flags (32 columns; header from
  `build\bench.exe --print-csv-header`).
- **Integrity checks done by analyze.py.** Rows with a non-OK status, a thread
  mismatch or an affinity failure are listed in `results/tables/anomalies.*`
  (none in the reported data); every method and p must give the same checksum
  for a given n.

## Reproducing the experiments

Close other applications and do not use the machine while a sweep runs:
background load visibly distorts the results (see `results/raw/README.md`).

```powershell
# Environment description
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_env.ps1

# Main sweep: n = 128..3072, p = 1,2,4,6,8,12,16,32, seq/omp/naive, pinned (~13 min)
python scripts/run_benchmarks.py --tag full_quiet

# Pinning contrast (~35 s)
python scripts/run_benchmarks.py --tag unpinned_quiet --pin none --methods omp `
    --sizes 1024 2048 --threads 4 8 16 32

# Optional vectorisation study (~13 min): simd build + scalar control, back to back
powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1 -Variant simd
python scripts/run_benchmarks.py --tag simd_quiet2 --bench-exe build/simd/bench.exe `
    --methods seq omp --sizes 512 1024 2048 3072 --threads 1 8 16
python scripts/run_benchmarks.py --tag scalar_ctrl_quiet2 `
    --methods seq omp --sizes 512 1024 2048 3072 --threads 1 8 16

# Vectorisation evidence (run in the current session, see the notes below)
& .\scripts\collect_vec_evidence.ps1
build\simd\test.exe
python scripts/validate_ref.py --test-exe build/simd/test.exe `
    --log results/vectorisation_evidence/validate_ref_simd.txt

# Analysis: tables and figures from the committed raw data
python scripts/analyze.py
```

Useful runner options: `--dry-run` (print the plan),
`--estimate-from results/raw/bench_full_quiet.csv` (predict the duration from
an earlier run), `--reps`, `--warmup`, `--time-budget`, `--mem-budget`,
`--naive-max-n` (default 1024). A tag can only be used once, so earlier data
are never overwritten.

`analyze.py` reads `bench_full_quiet.csv` (primary), `bench_unpinned_quiet.csv`,
`bench_full.csv` (background-load example), `bench_simd_quiet2.csv`,
`bench_scalar_ctrl_quiet2.csv` and `bench_vcheck_*.csv` by default; every input
can be changed or skipped (`''`) with its option (`--help`).

## Which data feed which result

| Result | Raw data | Tables / figures |
|---|---|---|
| Time, speedup, efficiency, GFLOP/s vs p and n | `bench_full_quiet.csv` | `stats_per_config`, `speedup_efficiency`, `speedup_matrix`, `parallel_payoff`, `flags`; `time_vs_n`, `time_vs_p`, `speedup_vs_p`, `efficiency_vs_p`, `gflops_vs_p` |
| OpenMP overhead at p = 1 | `bench_full_quiet.csv` | `omp_overhead` |
| Naive i-j-k vs i-k-j | `bench_full_quiet.csv` | `naive_vs_seq` |
| Pinned vs unpinned | `bench_full_quiet.csv`, `bench_unpinned_quiet.csv` | `pinning_contrast` (table and figure) |
| Effect of background load | `bench_full.csv` vs `bench_full_quiet.csv` | `noisy_vs_quiet`, `speedup_vs_p_noisy` |
| Vectorisation factorial | `bench_simd_quiet2.csv`, `bench_scalar_ctrl_quiet2.csv`, `bench_vcheck_*.csv` | `vectorisation_factorial` (table and figure), `vectorisation_reproducibility` |
| Vectorisation evidence | — | `results/vectorisation_evidence/summary.txt` and the files next to it |

`results/raw/README.md` lists every run, the conditions it was measured under
and the anomalies observed. Runs taken under background load are kept but not
used as primary data.

## Environment-specific notes

- **No GNU make.** The MSYS2 installation on the measured machine could not
  install `make`, so `build.ps1` is the build script actually used. The
  `Makefile` mirrors it (`make`, `make VARIANT=simd`) but has not been run.
- **Two build variants.** `build.ps1 -Variant scalar` (default, `build/`,
  `-O3 -fno-tree-vectorize`) and `-Variant simd` (`build/simd/`,
  `-O3 -march=native`, AVX2 on this CPU). With `-std=c11` gcc does not contract
  `a*b+c` into FMA (0 FMA instructions in the simd objects), so both variants
  produce bitwise-identical results; `collect_vec_evidence.ps1` checks this.
- **Windows Controlled Folder Access.** On the measured machine, programs that
  are not trusted by Windows may create files under the Desktop folder but not
  modify files created by another program, and a nested `powershell -File`
  process could not overwrite its own earlier output. For this reason
  `bench.exe` writes one part file per configuration that it creates itself
  (`bench_<tag>_parts/`), which the runner merges, and
  `collect_vec_evidence.ps1` is run with `&` in the current PowerShell session.
- **Path length.** `bench.exe` uses the classic Win32 file API (260-character
  limit). The runner refuses to start if an output path would be too long; keep
  the repository in a short enough directory.
- **Thread affinity.** See the benchmark protocol: pinning is done by
  `bench.exe`, and the `proc_bind` / `places` CSV columns only record the
  requested environment values.

## Packaging the materials

```powershell
python scripts/make_pdf.py                      # report/report.md -> report/report.pdf
python scripts/make_zip.py --surname <Surname>
```

`make_pdf.py` renders the Markdown with mistune and prints it to PDF with
Microsoft Edge (or Chrome) in headless mode; no other tools are needed.

Creates `Assignment03_<Surname>_Materials.zip` in the repository root with
every file tracked by git (sources, scripts, build files, raw data, tables,
figures, evidence, README and report) and nothing else: no binaries, no
`build/`, no generated `data/*.bin`. Commit before packaging; the script refuses
to run with uncommitted changes unless `--allow-dirty` is given.
