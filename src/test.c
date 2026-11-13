/*
 * Correctness tests for the matrix-multiplication kernels.
 *
 * Usage:
 *   test.exe                         run the full test suite (exit code 0 = pass)
 *   test.exe --dump N SEED DIR [P]   write A, B, C_seq and C_omp (P threads,
 *                                    default 4) as MATMULv1 files into DIR,
 *                                    for scripts/validate_ref.py
 *
 * Tolerance for floating-point comparisons: |x - y| <= ATOL + RTOL * |y| with
 * RTOL = ATOL = 1e-12. Bitwise equality between the sequential and the OpenMP
 * kernels is checked and reported separately (it is expected, because both
 * execute the same per-element summation order).
 *
 * Data-race checking (ThreadSanitizer): -fsanitize=thread cannot be used with
 * the MinGW-w64 / MSYS2 UCRT64 toolchain because libtsan is not shipped and the
 * link fails (see results/correctness/tsan_unavailable.txt). Data races were
 * therefore NOT verified with a dynamic tool. Supporting evidence instead:
 * the design argument in matmul_omp.c (schedule(static) assigns disjoint rows
 * of C; A and B are read-only) and the bitwise equality of the OpenMP and
 * sequential results for n = 1, 7, 64, 100, 257 and p = 1, 2, 3, 4, 8, 16, 32.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inputs.h"
#include "matmul.h"

#define RTOL 1e-12
#define ATOL 1e-12

static int g_checks = 0;
static int g_failures = 0;

static void check(int condition, const char *what)
{
    ++g_checks;
    if (!condition) {
        ++g_failures;
        printf("  FAIL: %s\n", what);
    }
}

/* Largest violation ratio |x-y| / (ATOL + RTOL*|y|); <= 1 means within tolerance. */
static double tolerance_ratio(const double *x, const double *y, size_t count)
{
    double worst = 0.0;
    for (size_t q = 0; q < count; ++q) {
        if (!isfinite(x[q]) || !isfinite(y[q]))
            return INFINITY;
        double diff = fabs(x[q] - y[q]);
        double ratio = diff / (ATOL + RTOL * fabs(y[q]));
        if (ratio > worst)
            worst = ratio;
    }
    return worst;
}

static double max_rel_error(const double *x, const double *y, size_t count)
{
    double worst = 0.0;
    for (size_t q = 0; q < count; ++q) {
        double denom = fabs(y[q]) > 0.0 ? fabs(y[q]) : 1.0;
        double rel = fabs(x[q] - y[q]) / denom;
        if (rel > worst)
            worst = rel;
    }
    return worst;
}

static int bitwise_equal(const double *x, const double *y, size_t count)
{
    return memcmp(x, y, count * sizeof(double)) == 0;
}

static void fill_value(double *M, size_t count, double value)
{
    for (size_t q = 0; q < count; ++q)
        M[q] = value;
}

static void fill_identity(double *M, int n)
{
    fill_value(M, (size_t)n * (size_t)n, 0.0);
    for (int i = 0; i < n; ++i)
        M[(size_t)i * (size_t)n + (size_t)i] = 1.0;
}

/* Independent reference: i-j-k with long double (x87 80-bit) accumulation. */
static void matmul_reference_ld(const double *A, const double *B, double *C, int n)
{
    const size_t N = (size_t)n;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            long double sum = 0.0L;
            for (size_t k = 0; k < N; ++k)
                sum += (long double)A[i * N + k] * (long double)B[k * N + j];
            C[i * N + j] = (double)sum;
        }
}

/* Number of threads that actually run a parallel region after omp_set_num_threads(p). */
static int actual_threads(int p)
{
    int team = 0;
    omp_set_num_threads(p);
    #pragma omp parallel default(none) shared(team)
    {
        #pragma omp single
        team = omp_get_num_threads();
    }
    return team;
}

static const int THREADS[] = {1, 2, 3, 4, 8, 16, 32};
#define N_THREADS ((int)(sizeof THREADS / sizeof THREADS[0]))

/* ------------------------------------------------------------------ tests */

static void test_known_small(void)
{
    printf("[known] 2x2 and 3x3 with hand-computed results\n");
    const double a2[4] = {1, 2, 3, 4}, b2[4] = {5, 6, 7, 8}, e2[4] = {19, 22, 43, 50};
    const double a3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const double b3[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    const double e3[9] = {30, 24, 18, 84, 69, 54, 138, 114, 90};
    double c[9];

    matmul_seq(a2, b2, c, 2);
    check(bitwise_equal(c, e2, 4), "2x2 seq");
    for (int t = 0; t < N_THREADS; ++t) {
        omp_set_num_threads(THREADS[t]);
        fill_value(c, 4, NAN);
        matmul_omp(a2, b2, c, 2);
        check(bitwise_equal(c, e2, 4), "2x2 omp");
    }
    matmul_seq(a3, b3, c, 3);
    check(bitwise_equal(c, e3, 9), "3x3 seq");
    for (int t = 0; t < N_THREADS; ++t) {
        omp_set_num_threads(THREADS[t]);
        fill_value(c, 9, NAN);
        matmul_omp(a3, b3, c, 3);
        check(bitwise_equal(c, e3, 9), "3x3 omp");
    }
    matmul_naive_ijk(a3, b3, c, 3);
    check(bitwise_equal(c, e3, 9), "3x3 naive");
}

static void test_identity_and_zero(int n)
{
    printf("[identity/zero] n=%d\n", n);
    const size_t count = (size_t)n * (size_t)n;
    double *A = alloc_matrix(n), *B = alloc_matrix(n), *I = alloc_matrix(n);
    double *Z = alloc_matrix(n), *C = alloc_matrix(n), *zero = alloc_matrix(n);
    if (!A || !B || !I || !Z || !C || !zero) {
        check(0, "allocation");
        goto done;
    }
    generate_inputs(A, B, n, MATMUL_DEFAULT_SEED);
    fill_identity(I, n);
    fill_value(Z, count, 0.0);
    fill_value(zero, count, 0.0);

    /* A*I == A and I*B == B exactly: every product is x*1 or x*0 and adding +0 is exact. */
    fill_value(C, count, NAN);
    matmul_seq(A, I, C, n);
    check(bitwise_equal(C, A, count), "seq A*I == A");
    fill_value(C, count, NAN);
    matmul_seq(I, B, C, n);
    check(bitwise_equal(C, B, count), "seq I*B == B");
    fill_value(C, count, NAN);
    matmul_seq(A, Z, C, n);
    check(bitwise_equal(C, zero, count), "seq A*0 == 0");

    for (int t = 0; t < N_THREADS; ++t) {
        omp_set_num_threads(THREADS[t]);
        fill_value(C, count, NAN);
        matmul_omp(A, I, C, n);
        check(bitwise_equal(C, A, count), "omp A*I == A");
        fill_value(C, count, NAN);
        matmul_omp(I, B, C, n);
        check(bitwise_equal(C, B, count), "omp I*B == B");
        fill_value(C, count, NAN);
        matmul_omp(Z, B, C, n);
        check(bitwise_equal(C, zero, count), "omp 0*B == 0");
    }
done:
    free_matrix(A); free_matrix(B); free_matrix(I);
    free_matrix(Z); free_matrix(C); free_matrix(zero);
}

/* Returns 1 if all OpenMP results were bitwise identical to the sequential one. */
static int test_random(int n)
{
    const size_t count = (size_t)n * (size_t)n;
    double *A = alloc_matrix(n), *B = alloc_matrix(n);
    double *Cs = alloc_matrix(n), *Cp = alloc_matrix(n), *Cr = alloc_matrix(n);
    int all_bitwise = 1;
    if (!A || !B || !Cs || !Cp || !Cr) {
        check(0, "allocation");
        all_bitwise = 0;
        goto done;
    }
    generate_inputs(A, B, n, MATMUL_DEFAULT_SEED);

    /* Garbage in C (NaN) proves every kernel overwrites/zeroes C itself. */
    fill_value(Cs, count, NAN);
    matmul_seq(A, B, Cs, n);
    matmul_reference_ld(A, B, Cr, n);
    double ref_ratio = tolerance_ratio(Cs, Cr, count);
    printf("[random] n=%-4d seq vs long-double ref: max rel err %.3e (tol ratio %.3e) %s\n",
           n, max_rel_error(Cs, Cr, count), ref_ratio, ref_ratio <= 1.0 ? "ok" : "FAIL");
    check(ref_ratio <= 1.0, "seq vs long double reference");

    fill_value(Cp, count, NAN);
    matmul_naive_ijk(A, B, Cp, n);
    int naive_bitwise = bitwise_equal(Cp, Cs, count);
    check(tolerance_ratio(Cp, Cs, count) <= 1.0, "naive vs seq within tolerance");
    printf("         naive vs seq: bitwise %s\n", naive_bitwise ? "identical" : "different");

    printf("         omp vs seq:");
    for (int t = 0; t < N_THREADS; ++t) {
        int p = THREADS[t];
        omp_set_num_threads(p);
        fill_value(Cp, count, NAN);
        matmul_omp(A, B, Cp, n);
        int within = tolerance_ratio(Cp, Cs, count) <= 1.0;
        int bitwise = bitwise_equal(Cp, Cs, count);
        all_bitwise &= bitwise;
        check(within, "omp vs seq within tolerance");
        printf(" p=%d:%s", p, !within ? "FAIL" : (bitwise ? "bitwise" : "tol-only"));
    }
    printf("\n");
done:
    free_matrix(A); free_matrix(B); free_matrix(Cs); free_matrix(Cp); free_matrix(Cr);
    return all_bitwise;
}

static void test_thread_counts(void)
{
    printf("[threads] requested vs actual team size (omp_set_dynamic(0))\n        ");
    for (int t = 0; t < N_THREADS; ++t) {
        int p = THREADS[t];
        int got = actual_threads(p);
        printf(" %d->%d", p, got);
        check(got == p, "actual thread count equals requested");
    }
    printf("\n");
}

static void test_inputs(void)
{
    printf("[inputs] determinism, range, seed sensitivity, save/load\n");
    const int n = 64;
    const size_t count = (size_t)n * n;
    double *A1 = alloc_matrix(n), *B1 = alloc_matrix(n), *A2 = alloc_matrix(n), *B2 = alloc_matrix(n);
    if (!A1 || !B1 || !A2 || !B2) {
        check(0, "allocation");
        goto done;
    }
    generate_inputs(A1, B1, n, MATMUL_DEFAULT_SEED);
    generate_inputs(A2, B2, n, MATMUL_DEFAULT_SEED);
    check(bitwise_equal(A1, A2, count) && bitwise_equal(B1, B2, count), "same seed -> identical inputs");
    int in_range = 1;
    for (size_t q = 0; q < count; ++q)
        in_range &= (A1[q] >= 0.0 && A1[q] < 1.0 && B1[q] >= 0.0 && B1[q] < 1.0);
    check(in_range, "values in [0, 1)");
    check(!bitwise_equal(A1, B1, count), "A and B differ");
    generate_inputs(A2, B2, n, MATMUL_DEFAULT_SEED + 1);
    check(!bitwise_equal(A1, A2, count), "different seed -> different inputs");

    const char *path = "test_roundtrip.bin";
    uint64_t seed = 0;
    fill_value(A2, count, 0.0);
    check(save_matrix(path, A1, n, MATMUL_DEFAULT_SEED) == 0, "save_matrix");
    check(load_matrix(path, A2, n, &seed) == 0, "load_matrix");
    check(bitwise_equal(A1, A2, count) && seed == MATMUL_DEFAULT_SEED, "save/load round trip");
    printf("         (an error message about 32x32 below is expected)\n");
    check(load_matrix(path, A2, 32, NULL) == -1, "load_matrix rejects wrong n");
    remove(path);
done:
    free_matrix(A1); free_matrix(B1); free_matrix(A2); free_matrix(B2);
}

static void test_degenerate(void)
{
    printf("[degenerate] n=0 is a no-op\n");
    double sentinel = 42.0;
    matmul_seq(NULL, NULL, &sentinel, 0);
    matmul_omp(NULL, NULL, &sentinel, 0);
    matmul_naive_ijk(NULL, NULL, &sentinel, 0);
    check(sentinel == 42.0, "n=0 leaves C untouched");
}

/* ------------------------------------------------------------------ dump */

static int dump(int n, uint64_t seed, const char *dir, int p)
{
    char path[1024];
    double *A = alloc_matrix(n), *B = alloc_matrix(n), *C = alloc_matrix(n);
    int rc = 0;
    if (!A || !B || !C) {
        fprintf(stderr, "dump: allocation failed for n=%d\n", n);
        rc = 1;
        goto done;
    }
    generate_inputs(A, B, n, seed);
    snprintf(path, sizeof path, "%s/A_n%d_s%llu.bin", dir, n, (unsigned long long)seed);
    rc |= save_matrix(path, A, n, seed) != 0;
    snprintf(path, sizeof path, "%s/B_n%d_s%llu.bin", dir, n, (unsigned long long)seed);
    rc |= save_matrix(path, B, n, seed) != 0;

    matmul_seq(A, B, C, n);
    snprintf(path, sizeof path, "%s/C_seq_n%d_s%llu.bin", dir, n, (unsigned long long)seed);
    rc |= save_matrix(path, C, n, seed) != 0;

    omp_set_num_threads(p);
    matmul_omp(A, B, C, n);
    snprintf(path, sizeof path, "%s/C_omp_p%d_n%d_s%llu.bin", dir, p, n, (unsigned long long)seed);
    rc |= save_matrix(path, C, n, seed) != 0;
    if (rc == 0)
        printf("dump: wrote A, B, C_seq, C_omp (p=%d) for n=%d seed=%llu to %s\n",
               p, n, (unsigned long long)seed, dir);
done:
    free_matrix(A); free_matrix(B); free_matrix(C);
    return rc;
}

int main(int argc, char **argv)
{
    /* Unbuffered stdout keeps expected stderr messages in order in logs. */
    setvbuf(stdout, NULL, _IONBF, 0);
    omp_set_dynamic(0);

    if (argc >= 2 && strcmp(argv[1], "--dump") == 0) {
        if (argc < 5) {
            fprintf(stderr, "usage: %s --dump N SEED DIR [P]\n", argv[0]);
            return 2;
        }
        int n = atoi(argv[2]);
        uint64_t seed = strtoull(argv[3], NULL, 10);
        int p = argc >= 6 ? atoi(argv[5]) : 4;
        if (n <= 0 || p <= 0) {
            fprintf(stderr, "dump: N and P must be positive\n");
            return 2;
        }
        return dump(n, seed, argv[4], p);
    }

    printf("Correctness tests: RTOL=%.0e ATOL=%.0e, seed=%llu, omp_get_num_procs()=%d\n",
           RTOL, ATOL, (unsigned long long)MATMUL_DEFAULT_SEED, omp_get_num_procs());
    test_known_small();
    test_thread_counts();
    test_inputs();
    test_degenerate();

    const int id_sizes[] = {1, 7, 64};
    for (size_t s = 0; s < sizeof id_sizes / sizeof id_sizes[0]; ++s)
        test_identity_and_zero(id_sizes[s]);

    const int sizes[] = {1, 7, 64, 100, 257};
    int all_bitwise = 1;
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s)
        all_bitwise &= test_random(sizes[s]);

    printf("\nSummary: %d checks, %d failures. seq vs omp bitwise identical for all n and p: %s\n",
           g_checks, g_failures, all_bitwise ? "YES" : "NO");
    printf(g_failures == 0 ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
    return g_failures == 0 ? 0 : 1;
}
