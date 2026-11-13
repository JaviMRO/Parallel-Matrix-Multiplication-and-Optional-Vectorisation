#include "matmul.h"

#ifndef _OPENMP
#error "matmul_omp.c must be compiled WITH -fopenmp"
#endif

void matmul_omp(const double *A, const double *B, double *C, int n)
{
    const size_t N = (size_t)(n > 0 ? n : 0);

    /*
     * Why there are no data races:
     *   - The loop variable i is private to each thread (OpenMP loop rule), and
     *     every variable declared inside matmul_row_ikj (j, k, aik, Ci, Ai, Bk)
     *     is a local of the calling thread, hence private as well.
     *   - schedule(static) gives each thread a disjoint set of rows i, and a
     *     thread only writes C[i * N .. i * N + N - 1] for its own rows, so no
     *     two threads ever write the same element of C.
     *   - A and B are only read; N is read-only.
     * No reduction, atomic or critical section is needed. The implicit barrier
     * at the end of the parallel loop guarantees C is complete on return.
     */
    #pragma omp parallel for schedule(static) default(none) shared(A, B, C, N)
    for (size_t i = 0; i < N; ++i)
        matmul_row_ikj(A, B, C, N, i);
}
