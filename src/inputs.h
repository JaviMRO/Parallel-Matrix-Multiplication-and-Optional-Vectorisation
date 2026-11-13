#ifndef INPUTS_H
#define INPUTS_H

#include <stddef.h>
#include <stdint.h>

/* Default seed of the experiments (recorded in the README and in the CSVs). */
#define MATMUL_DEFAULT_SEED UINT64_C(20250301)

/* Alignment of every matrix buffer, in bytes (one cache line). */
#define MATRIX_ALIGNMENT 64

/* SplitMix64 step: advances *state and returns the next 64-bit output. */
uint64_t splitmix64_next(uint64_t *state);

/* Fills M[0..count-1] with doubles uniformly distributed in [0, 1). */
void fill_random(double *M, size_t count, uint64_t *state);

/*
 * Deterministic inputs: one SplitMix64 stream seeded with `seed` fills A
 * (row-major) and then B. Same (n, seed) => bit-identical A and B.
 */
void generate_inputs(double *A, double *B, int n, uint64_t seed);

/* Allocates an n x n matrix aligned to MATRIX_ALIGNMENT; NULL on failure. */
double *alloc_matrix(int n);
void free_matrix(double *M);

/* Sum of all elements in storage order (deterministic, used to consume C). */
double matrix_checksum(const double *M, size_t count);

/*
 * Binary file format (little-endian, x86-64):
 *   char[8]  magic "MATMULv1"
 *   uint64   rows
 *   uint64   cols
 *   uint64   seed used to generate the matrix (0 if unknown)
 *   double   rows * cols elements, row-major
 * Both functions return 0 on success and -1 on failure (message on stderr).
 */
int save_matrix(const char *path, const double *M, int n, uint64_t seed);
int load_matrix(const char *path, double *M, int n, uint64_t *seed_out);

#endif /* INPUTS_H */
