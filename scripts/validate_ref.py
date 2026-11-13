"""Validate the C kernels against an independent reference (numpy).

For each n, build/test.exe --dump writes A, B, C_seq and C_omp (MATMULv1
binary files) into data/. This script then:
  1. regenerates A and B with a Python re-implementation of SplitMix64 and
     checks they are bit-identical to the C generator's output;
  2. compares C_seq and C_omp with numpy's A @ B (OpenBLAS, forced to one
     thread) using rtol = atol = 1e-12;
  3. checks that C_omp is bitwise identical to C_seq.

numpy's BLAS uses a different summation order (blocking, possibly FMA), so
exact equality with numpy is NOT expected; only the tolerance check is.

Note on data races: ThreadSanitizer (-fsanitize=thread / tsan) is not
available with MinGW-w64 / MSYS2 UCRT64 (libtsan is not shipped), so races were
not checked dynamically. Supporting evidence is the disjoint-rows design of
matmul_omp.c plus the bitwise equality C_omp == C_seq checked here and in
src/test.c for many n and p.

Usage:
    python scripts/validate_ref.py [--sizes 64 257 512] [--threads 8]
Writes a log to results/correctness/validate_ref.txt. Exit code 0 = pass.
"""

import os

# Must be set before numpy is imported so the BLAS library starts single-threaded.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST_EXE = ROOT / "build" / ("test.exe" if os.name == "nt" else "test")
DATA_DIR = ROOT / "data"
LOG_FILE = ROOT / "results" / "correctness" / "validate_ref.txt"

DEFAULT_SEED = 20250301  # MATMUL_DEFAULT_SEED in src/inputs.h
RTOL = 1e-12
ATOL = 1e-12
MAGIC = b"MATMULv1"
HEADER = struct.Struct("<8sQQQ")  # magic, rows, cols, seed

GAMMA = np.uint64(0x9E3779B97F4A7C15)
MIX1 = np.uint64(0xBF58476D1CE4E5B9)
MIX2 = np.uint64(0x94D049BB133111EB)


def load_matrix(path):
    """Reads a MATMULv1 file; returns (matrix, seed)."""
    data = path.read_bytes()
    magic, rows, cols, seed = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path} is not a MATMULv1 file")
    values = np.frombuffer(data, dtype="<f8", offset=HEADER.size)
    if rows != cols:
        raise ValueError(f"{path}: matrix is {rows}x{cols}, expected square")
    if values.size != rows * cols:
        raise ValueError(f"{path}: expected {rows * cols} values, found {values.size}")
    return values.reshape(rows, cols), seed


def splitmix64_doubles(seed, count):
    """Vectorised SplitMix64 -> doubles in [0, 1); same stream as src/inputs.c."""
    with np.errstate(over="ignore"):
        k = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(seed) + k * GAMMA  # state after k increments (mod 2^64)
        z = (z ^ (z >> np.uint64(30))) * MIX1
        z = (z ^ (z >> np.uint64(27))) * MIX2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


def blas_threads():
    """Number of BLAS threads numpy will use, if threadpoolctl is available."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return "unknown (threadpoolctl not installed); OPENBLAS_NUM_THREADS=1 set before import"
    info = [f"{p.get('internal_api')}={p.get('num_threads')}" for p in threadpool_info()]
    return ", ".join(info) or "no BLAS pool detected"


def validate(n, seed, threads, log):
    paths = {
        "A": DATA_DIR / f"A_n{n}_s{seed}.bin",
        "B": DATA_DIR / f"B_n{n}_s{seed}.bin",
        "C_seq": DATA_DIR / f"C_seq_n{n}_s{seed}.bin",
        "C_omp": DATA_DIR / f"C_omp_p{threads}_n{n}_s{seed}.bin",
    }
    try:
        return _validate_dumped(n, seed, threads, paths, log)
    finally:
        # Remove dumped files even if the dump or a check failed part-way.
        for path in paths.values():
            path.unlink(missing_ok=True)


def _validate_dumped(n, seed, threads, paths, log):
    proc = subprocess.run(
        [str(TEST_EXE), "--dump", str(n), str(seed), str(DATA_DIR), str(threads)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log(f"n={n}: test.exe --dump failed ({proc.returncode}): {proc.stderr.strip()}")
        return False

    matrices = {}
    for name, path in paths.items():
        M, file_seed = load_matrix(path)
        if M.shape != (n, n) or file_seed != seed:
            log(f"n={n}: {path.name} has shape {M.shape} and seed {file_seed}, "
                f"expected ({n}, {n}) and seed {seed}")
            return False
        matrices[name] = M
    A, B, C_seq, C_omp = (matrices[k] for k in ("A", "B", "C_seq", "C_omp"))

    stream = splitmix64_doubles(seed, 2 * n * n)
    inputs_ok = (np.array_equal(A.ravel(), stream[: n * n])
                 and np.array_equal(B.ravel(), stream[n * n:]))

    ref = A @ B
    ok = inputs_ok
    log(f"n={n:<5} inputs bit-identical to Python SplitMix64: {'yes' if inputs_ok else 'NO'}")
    for name, C in (("C_seq", C_seq), (f"C_omp(p={threads})", C_omp)):
        abs_err = np.abs(C - ref)
        rel_err = abs_err / np.where(ref != 0.0, np.abs(ref), 1.0)
        close = bool(np.allclose(C, ref, rtol=RTOL, atol=ATOL))
        exact = bool(np.array_equal(C, ref))
        ok &= close
        log(f"        {name:<12} vs numpy: max abs err {abs_err.max():.3e}, "
            f"max rel err {rel_err.max():.3e}, allclose(rtol=atol=1e-12): "
            f"{'yes' if close else 'NO'}, bitwise equal to numpy: {'yes' if exact else 'no'}")
    same = C_seq.tobytes() == C_omp.tobytes()
    ok &= same
    log(f"        C_omp(p={threads}) bitwise identical to C_seq: {'yes' if same else 'NO'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 257, 512])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not TEST_EXE.exists():
        sys.exit(f"{TEST_EXE} not found; build first (build.ps1 or make)")
    DATA_DIR.mkdir(exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    def log(text):
        print(text)
        lines.append(text)

    log(f"Reference validation against numpy {np.__version__} "
        f"(rtol={RTOL:g}, atol={ATOL:g}, seed={args.seed})")
    log(f"BLAS threads: {blas_threads()}")
    results = [validate(n, args.seed, args.threads, log) for n in args.sizes]
    passed = all(results)
    log("VALIDATION PASSED" if passed else "VALIDATION FAILED")
    LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
