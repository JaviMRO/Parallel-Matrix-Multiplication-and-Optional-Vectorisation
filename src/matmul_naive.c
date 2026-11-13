#include "matmul.h"

/*
 * Optional slow reference (NOT the baseline): i-j-k order, sequential.
 * Each C[i][j] is accumulated in a scalar starting from 0.0 over
 * k = 0, 1, ..., n-1, i.e. the same summation order as the i-k-j kernel.
 */
#ifdef _OPENMP
#error "matmul_naive.c must be compiled WITHOUT -fopenmp"
#endif

void matmul_naive_ijk(const double *A, const double *B, double *C, int n)
{
    const size_t N = (size_t)(n > 0 ? n : 0);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}
