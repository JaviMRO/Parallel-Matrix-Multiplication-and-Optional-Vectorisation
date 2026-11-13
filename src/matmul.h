#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

/*
 * Dense matrix multiplication C = A * B for square n x n matrices.
 *
 * Common contract for every implementation:
 *   - Element type: double (IEEE-754 binary64).
 *   - Layout: row-major, contiguous 1D arrays; element (r, c) is M[r * n + c].
 *   - C is overwritten: every kernel sets C to zero itself before
 *     accumulating, so the caller does not need to clear it.
 *   - C must not alias A or B (A and B may alias each other).
 *   - n >= 0; n == 0 is a no-op.
 */

/*
 * Computes row i of C with the i-k-j loop order:
 *   C[i][:] = 0;  for k: C[i][:] += A[i][k] * B[k][:]
 * The innermost loop walks B and C contiguously (unit stride).
 *
 * This is the single kernel body shared by matmul_seq and matmul_omp, so both
 * versions execute exactly the same arithmetic in the same order. For every
 * element C[i][j] the products are summed over k = 0, 1, ..., n-1, starting
 * from 0.0, independently of how rows are distributed among threads.
 */
static inline void matmul_row_ikj(const double *restrict A,
                                  const double *restrict B,
                                  double *restrict C,
                                  size_t n, size_t i)
{
    double *restrict Ci = C + i * n;
    const double *restrict Ai = A + i * n;

    for (size_t j = 0; j < n; ++j)
        Ci[j] = 0.0;

    for (size_t k = 0; k < n; ++k) {
        const double aik = Ai[k];
        const double *restrict Bk = B + k * n;
        for (size_t j = 0; j < n; ++j)
            Ci[j] += aik * Bk[j];
    }
}

/* Sequential baseline (i-k-j). Compiled without -fopenmp. */
void matmul_seq(const double *A, const double *B, double *C, int n);

/*
 * OpenMP version: the same i-k-j kernel with the i loop shared among threads
 * (schedule(static)). The number of threads is taken from the OpenMP runtime
 * (omp_set_num_threads / OMP_NUM_THREADS); the caller controls it.
 */
void matmul_omp(const double *A, const double *B, double *C, int n);

/*
 * Optional slow reference: textbook i-j-k order with a scalar accumulator.
 * The inner loop reads B with stride n (cache-unfriendly). It is NOT the
 * baseline; it is only reported as an additional, clearly separated reference.
 */
void matmul_naive_ijk(const double *A, const double *B, double *C, int n);

#endif /* MATMUL_H */
