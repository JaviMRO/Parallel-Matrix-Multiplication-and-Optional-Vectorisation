#include "matmul.h"

/*
 * The sequential baseline must not contain any active OpenMP construct, so
 * this translation unit is compiled without -fopenmp. Fail loudly otherwise.
 */
#ifdef _OPENMP
#error "matmul_seq.c must be compiled WITHOUT -fopenmp"
#endif

void matmul_seq(const double *A, const double *B, double *C, int n)
{
    const size_t N = (size_t)(n > 0 ? n : 0);

    for (size_t i = 0; i < N; ++i)
        matmul_row_ikj(A, B, C, N, i);
}
