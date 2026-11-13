#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L /* posix_memalign under -std=c11 */
#endif

#include "inputs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h> /* _aligned_malloc / _aligned_free */
#endif

static const char MAGIC[8] = {'M', 'A', 'T', 'M', 'U', 'L', 'v', '1'};

uint64_t splitmix64_next(uint64_t *state)
{
    /* Reference constants from Steele, Lea and Flood (2014), as used by Vigna. */
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

void fill_random(double *M, size_t count, uint64_t *state)
{
    /* Top 53 bits -> exactly representable double in [0, 1). */
    for (size_t idx = 0; idx < count; ++idx)
        M[idx] = (double)(splitmix64_next(state) >> 11) * 0x1.0p-53;
}

void generate_inputs(double *A, double *B, int n, uint64_t seed)
{
    const size_t count = (size_t)n * (size_t)n;
    uint64_t state = seed;
    fill_random(A, count, &state);
    fill_random(B, count, &state);
}

double *alloc_matrix(int n)
{
    if (n <= 0)
        return NULL;
    const size_t count = (size_t)n * (size_t)n;
    if (count > SIZE_MAX / sizeof(double))
        return NULL;
    size_t bytes = count * sizeof(double);
    /* Round up to a multiple of the alignment (required by aligned_alloc-style APIs). */
    bytes = (bytes + MATRIX_ALIGNMENT - 1) / MATRIX_ALIGNMENT * MATRIX_ALIGNMENT;
#ifdef _WIN32
    return (double *)_aligned_malloc(bytes, MATRIX_ALIGNMENT);
#else
    void *p = NULL;
    if (posix_memalign(&p, MATRIX_ALIGNMENT, bytes) != 0)
        return NULL;
    return (double *)p;
#endif
}

void free_matrix(double *M)
{
#ifdef _WIN32
    _aligned_free(M);
#else
    free(M);
#endif
}

double matrix_checksum(const double *M, size_t count)
{
    double sum = 0.0;
    for (size_t idx = 0; idx < count; ++idx)
        sum += M[idx];
    return sum;
}

static int write_u64(FILE *f, uint64_t v)
{
    return fwrite(&v, sizeof v, 1, f) == 1 ? 0 : -1;
}

static int read_u64(FILE *f, uint64_t *v)
{
    return fread(v, sizeof *v, 1, f) == 1 ? 0 : -1;
}

int save_matrix(const char *path, const double *M, int n, uint64_t seed)
{
    const size_t count = (size_t)n * (size_t)n;
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "save_matrix: cannot open '%s' for writing\n", path);
        return -1;
    }
    int rc = 0;
    if (fwrite(MAGIC, sizeof MAGIC, 1, f) != 1 ||
        write_u64(f, (uint64_t)n) != 0 ||
        write_u64(f, (uint64_t)n) != 0 ||
        write_u64(f, seed) != 0 ||
        fwrite(M, sizeof(double), count, f) != count) {
        fprintf(stderr, "save_matrix: write error on '%s'\n", path);
        rc = -1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "save_matrix: close error on '%s'\n", path);
        rc = -1;
    }
    return rc;
}

int load_matrix(const char *path, double *M, int n, uint64_t *seed_out)
{
    const size_t count = (size_t)n * (size_t)n;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_matrix: cannot open '%s'\n", path);
        return -1;
    }
    char magic[sizeof MAGIC];
    uint64_t rows = 0, cols = 0, seed = 0;
    int rc = -1;
    if (fread(magic, sizeof magic, 1, f) != 1 || memcmp(magic, MAGIC, sizeof MAGIC) != 0) {
        fprintf(stderr, "load_matrix: '%s' is not a MATMULv1 file\n", path);
    } else if (read_u64(f, &rows) != 0 || read_u64(f, &cols) != 0 || read_u64(f, &seed) != 0) {
        fprintf(stderr, "load_matrix: truncated header in '%s'\n", path);
    } else if (rows != (uint64_t)n || cols != (uint64_t)n) {
        fprintf(stderr, "load_matrix: '%s' is %llux%llu, expected %dx%d\n", path,
                (unsigned long long)rows, (unsigned long long)cols, n, n);
    } else if (fread(M, sizeof(double), count, f) != count) {
        fprintf(stderr, "load_matrix: truncated data in '%s'\n", path);
    } else if (fgetc(f) != EOF) {
        fprintf(stderr, "load_matrix: trailing bytes in '%s'\n", path);
    } else {
        if (seed_out)
            *seed_out = seed;
        rc = 0;
    }
    fclose(f);
    return rc;
}
