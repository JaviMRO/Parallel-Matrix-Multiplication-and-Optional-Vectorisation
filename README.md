# Parallel Matrix Multiplication and Optional Vectorisation

Big Data – Assignment 3. Study of how the number of OpenMP threads `p` affects the
performance of dense, shared-memory matrix multiplication `C = A·B` (`double`,
row-major, contiguous 1D arrays), and of when parallelisation pays off.

> This README is a skeleton. Full build, test, benchmark and analysis instructions
> are added once the corresponding parts of the project exist.

## Repository structure

```
.
├── src/                       C sources: kernels, input generator, timing, benchmark, tests
├── scripts/                   collect_env.ps1, benchmark runner, analysis and validation scripts
├── data/                      generated input matrices (*.bin are git-ignored; regenerated from a fixed seed)
├── results/
│   ├── environment.txt        hardware/software environment (scripts/collect_env.ps1)
│   ├── raw/                   raw per-repetition CSV measurements (never filtered)
│   ├── figures/               PNG + PDF figures
│   ├── tables/                CSV + Markdown summary tables
│   └── vectorisation_evidence/ compiler vectorisation reports and assembly excerpts (optional part)
├── report/                    report.md (source of the PDF report)
├── build/                     compiler outputs (git-ignored)
├── build.ps1 / Makefile       build scripts
└── README.md
```

## Toolchain

- Windows 11, GCC 14.2.0 (MSYS2 UCRT64) with OpenMP 4.5 (`libgomp`).
- Python 3.10 with numpy, pandas and matplotlib (validation and analysis).

## Environment capture

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_env.ps1
```

Writes `results/environment.txt`.

## Correctness testing

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1
build\test.exe                      # unit tests; log in results/correctness/test_output.txt
python scripts/validate_ref.py      # numpy reference (OPENBLAS_NUM_THREADS=1); log in results/correctness/validate_ref.txt
```

- Tolerance: `|x - y| <= atol + rtol*|y|` with `rtol = atol = 1e-12`. Bitwise
  equality between the sequential and OpenMP results is checked and reported
  separately (it holds, because both use the same per-element summation order).
- ThreadSanitizer (`-fsanitize=thread`, tsan) is **not available** with
  MinGW-w64 / MSYS2 UCRT64: libtsan is not shipped and the link fails (see
  `results/correctness/tsan_unavailable.txt`). Data races were therefore not
  checked with a dynamic tool. Supporting evidence: the disjoint-rows design of
  `matmul_omp.c` (`schedule(static)` over rows of C; A and B read-only) and the
  bitwise equality of OpenMP and sequential results for n = 1, 7, 64, 100, 257
  and p = 1, 2, 3, 4, 8, 16, 32.
