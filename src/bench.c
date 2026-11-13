/*
 * Benchmark harness: times one (method, n, p) configuration and appends one
 * CSV row per repetition (warm-ups included, flagged with is_warmup = 1).
 *
 * Usage:
 *   bench.exe --method {seq,omp,naive} --n N --out FILE.csv
 *             [--threads P] [--reps R] [--warmup W] [--seed S]
 *             [--pin {none,cores}] [--input-dir DIR]
 *             [--time-budget SECONDS] [--mem-budget BYTES]
 *             [--run-id ID] [--config-index K]
 *   bench.exe --print-csv-header
 *
 * Protocol:
 *   - Inputs are generated (or loaded) and all buffers are allocated and
 *     touched BEFORE any timing. Only the kernel call is inside the timed
 *     region, measured with a monotonic clock (timing.c).
 *   - W warm-up repetitions (default 2) are run and recorded but flagged; R
 *     measured repetitions follow (default 10).
 *   - After every repetition the checksum of C is computed (outside the timed
 *     region) and written to the CSV, so the result is always consumed.
 *   - Time budget: if one repetition takes longer than --time-budget seconds,
 *     that row gets status TIMEOUT (its real time is kept) and no further
 *     repetitions are run. Memory budget: if 3*8*n^2 bytes exceed
 *     --mem-budget, nothing is allocated and a SKIPPED_MEMORY_BUDGET row is
 *     written. Failures are written as ERROR rows. Times are never invented:
 *     time_s is empty for rows without a completed kernel call.
 *   - The OpenMP team size is checked inside a parallel region before every
 *     repetition (p_actual) together with omp_get_max_threads(). With
 *     --pin cores each OpenMP thread is pinned to one logical processor (see
 *     affinity.h) and affinity_ok records whether every thread is still the
 *     same OS thread running on its assigned processor. A repetition whose
 *     team size differs from p gets status P_MISMATCH, and one whose check
 *     failed gets AFFINITY_FAIL (time kept, row flagged).
 *   - proc_bind / places columns hold the REQUESTED OMP_PROC_BIND / OMP_PLACES
 *     environment values. libgomp on Windows (MinGW-w64) ignores them; the
 *     affinity actually applied is given by pin_mode and pinned_lps.
 */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "affinity.h"
#include "build_info.h"
#include "inputs.h"
#include "matmul.h"
#include "timing.h"

#define MAX_THREADS 256

typedef enum { METHOD_SEQ, METHOD_OMP, METHOD_NAIVE } method_t;

typedef struct {
    method_t method;
    const char *method_name;
    int n;
    int threads;
    int reps;
    int warmup;
    uint64_t seed;
    int pin;                 /* 0 none, 1 cores */
    const char *input_dir;   /* NULL: generate in memory */
    double time_budget;      /* seconds per repetition */
    uint64_t mem_budget;     /* bytes */
    const char *out_path;
    const char *run_id;
    int config_index;
} options;

/* Per-run state shared by every CSV row. */
typedef struct {
    cpu_topology topo;
    int p_actual;
    int omp_max_threads;
    int omp_dynamic;
    int oversubscribed;
    char pinned_lps[MAX_THREADS * 4];
    uint64_t rss_before_alloc;
    uint64_t theoretical_bytes;
    char input_source[64];
} run_state;

static const char *CSV_HEADER =
    "timestamp,run_id,config_index,n,method,dtype,p_requested,p_actual,"
    "omp_max_threads,omp_dynamic,logical_procs,physical_cores,oversubscribed,"
    "pin_mode,pinned_lps,affinity_ok,proc_bind,places,rep,is_warmup,status,"
    "time_s,peak_rss_bytes,rss_before_alloc_bytes,theoretical_bytes,checksum,"
    "seed,input_source,time_budget_s,peak_rss_source,compiler,compiler_flags";

/* OS thread id and assigned LP of every OpenMP thread (set while pinning). */
static unsigned long g_thread_ids[MAX_THREADS];
static int g_thread_lps[MAX_THREADS];

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --method {seq,omp,naive} --n N --out FILE.csv [--threads P]\n"
            "          [--reps R] [--warmup W] [--seed S] [--pin {none,cores}]\n"
            "          [--input-dir DIR] [--time-budget SEC] [--mem-budget BYTES]\n"
            "          [--run-id ID] [--config-index K]\n", prog);
}

static int parse_int(const char *s, int min, int *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < min || v > INT_MAX)
        return -1;
    *out = (int)v;
    return 0;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end || s[0] == '-')
        return -1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_args(int argc, char **argv, options *o)
{
    memset(o, 0, sizeof *o);
    o->method_name = NULL;
    o->n = -1;
    o->threads = 1;
    o->reps = 10;
    o->warmup = 2;
    o->seed = MATMUL_DEFAULT_SEED;
    o->pin = 0;
    o->time_budget = 300.0;
    o->mem_budget = UINT64_C(2) << 30; /* 2 GiB */
    o->run_id = "manual";
    o->config_index = -1;

    for (int a = 1; a < argc; ++a) {
        const char *k = argv[a];
        const char *v = a + 1 < argc ? argv[a + 1] : NULL;
        if (!v) {
            fprintf(stderr, "missing value for %s\n", k);
            return -1;
        }
        ++a;
        if (!strcmp(k, "--method")) {
            o->method_name = v;
            if (!strcmp(v, "seq")) o->method = METHOD_SEQ;
            else if (!strcmp(v, "omp")) o->method = METHOD_OMP;
            else if (!strcmp(v, "naive")) o->method = METHOD_NAIVE;
            else { fprintf(stderr, "unknown method '%s'\n", v); return -1; }
        } else if (!strcmp(k, "--n")) {
            if (parse_int(v, 1, &o->n)) { fprintf(stderr, "bad --n\n"); return -1; }
        } else if (!strcmp(k, "--threads")) {
            if (parse_int(v, 1, &o->threads) || o->threads > MAX_THREADS) {
                fprintf(stderr, "bad --threads (1..%d)\n", MAX_THREADS); return -1;
            }
        } else if (!strcmp(k, "--reps")) {
            if (parse_int(v, 1, &o->reps)) { fprintf(stderr, "bad --reps\n"); return -1; }
        } else if (!strcmp(k, "--warmup")) {
            if (parse_int(v, 0, &o->warmup)) { fprintf(stderr, "bad --warmup\n"); return -1; }
        } else if (!strcmp(k, "--seed")) {
            if (parse_u64(v, &o->seed)) { fprintf(stderr, "bad --seed\n"); return -1; }
        } else if (!strcmp(k, "--pin")) {
            if (!strcmp(v, "none")) o->pin = 0;
            else if (!strcmp(v, "cores")) o->pin = 1;
            else { fprintf(stderr, "bad --pin\n"); return -1; }
        } else if (!strcmp(k, "--input-dir")) {
            o->input_dir = v;
        } else if (!strcmp(k, "--time-budget")) {
            char *end = NULL;
            o->time_budget = strtod(v, &end);
            if (!end || *end || !(o->time_budget > 0)) { fprintf(stderr, "bad --time-budget\n"); return -1; }
        } else if (!strcmp(k, "--mem-budget")) {
            if (parse_u64(v, &o->mem_budget)) { fprintf(stderr, "bad --mem-budget\n"); return -1; }
        } else if (!strcmp(k, "--out")) {
            o->out_path = v;
        } else if (!strcmp(k, "--run-id")) {
            o->run_id = v;
        } else if (!strcmp(k, "--config-index")) {
            if (parse_int(v, 0, &o->config_index)) { fprintf(stderr, "bad --config-index\n"); return -1; }
        } else {
            fprintf(stderr, "unknown option '%s'\n", k);
            return -1;
        }
    }
    if (!o->method_name || o->n < 1 || !o->out_path) {
        fprintf(stderr, "--method, --n and --out are required\n");
        return -1;
    }
    if (o->method != METHOD_OMP && o->threads != 1) {
        fprintf(stderr, "--threads must be 1 for method '%s'\n", o->method_name);
        return -1;
    }
    if (o->reps < 5)
        fprintf(stderr, "warning: --reps %d < 5 (only acceptable for smoke tests)\n", o->reps);
    return 0;
}

static const char *env_or_unset(const char *name)
{
    const char *v = getenv(name);
    return (v && *v) ? v : "unset";
}

static void utc_timestamp(char *buf, size_t size)
{
    time_t now = time(NULL);
    struct tm *t = gmtime(&now);
    if (!t || !strftime(buf, size, "%Y-%m-%dT%H:%M:%SZ", t))
        snprintf(buf, size, "unknown");
}

static FILE *open_csv(const char *path)
{
    FILE *f = fopen(path, "ab+");
    if (!f) {
        fprintf(stderr, "cannot open '%s' for appending: %s\n", path, strerror(errno));
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
        fprintf(f, "%s\n", CSV_HEADER);
    return f;
}

/*
 * Appends one row. time_s < 0 means "no completed measurement" (empty cell);
 * checksum is written only when has_checksum is set.
 */
static void write_row(FILE *f, const options *o, const run_state *s, int rep, int is_warmup,
                      const char *status, int affinity_ok, double time_s,
                      int has_checksum, double checksum)
{
    char ts[32], time_buf[32] = "", sum_buf[40] = "", aff_buf[8] = "NA";
    utc_timestamp(ts, sizeof ts);
    if (time_s >= 0.0)
        snprintf(time_buf, sizeof time_buf, "%.9f", time_s);
    if (has_checksum)
        snprintf(sum_buf, sizeof sum_buf, "%.17g", checksum);
    if (affinity_ok >= 0)
        snprintf(aff_buf, sizeof aff_buf, "%d", affinity_ok);
    const int p_requested = o->method == METHOD_OMP ? o->threads : 1;

    fprintf(f,
            "%s,%s,%d,%d,%s,double,%d,%d,%d,%d,%d,%d,%d,%s,\"%s\",%s,\"%s\",\"%s\",%d,%d,%s,"
            "%s,%llu,%llu,%llu,%s,%llu,\"%s\",%.1f,%s,\"%s\",\"%s\"\n",
            ts, o->run_id, o->config_index, o->n, o->method_name, p_requested, s->p_actual,
            s->omp_max_threads, s->omp_dynamic, s->topo.logical_procs, s->topo.physical_cores,
            s->oversubscribed, o->pin ? "cores" : "none", s->pinned_lps, aff_buf,
            env_or_unset("OMP_PROC_BIND"), env_or_unset("OMP_PLACES"), rep, is_warmup, status,
            time_buf, (unsigned long long)peak_rss_bytes(),
            (unsigned long long)s->rss_before_alloc, (unsigned long long)s->theoretical_bytes,
            sum_buf, (unsigned long long)o->seed, s->input_source, o->time_budget,
            peak_rss_source(), MATMUL_COMPILER,
            o->method == METHOD_OMP ? MATMUL_CFLAGS " " MATMUL_OMP_FLAGS : MATMUL_CFLAGS);
    fflush(f);
}

/*
 * Creates the OpenMP team (p threads), records each thread's OS id and, if
 * requested, pins thread t to its LP. Returns the team size, or -1 if pinning
 * failed.
 */
static int setup_team(const options *o, run_state *s)
{
    int team = 0, pin_failures = 0;
    #pragma omp parallel default(none) shared(o, s, team, pin_failures, g_thread_ids, g_thread_lps)
    {
        const int t = omp_get_thread_num();
        #pragma omp single
        team = omp_get_num_threads();
        g_thread_ids[t] = current_thread_id();
        g_thread_lps[t] = -1;
        if (o->pin) {
            const int lp = topology_lp_for_thread(&s->topo, t);
            g_thread_lps[t] = lp;
            if (lp < 0 || pin_current_thread(lp) != 0) {
                #pragma omp atomic
                ++pin_failures;
            }
        }
    }
    return pin_failures ? -1 : team;
}

/*
 * Runs an (untimed) parallel region with the same team and checks that every
 * thread is the same OS thread as at setup and, if pinned, runs on its LP.
 * Writes the observed team size to *team_out. Returns 1 if all checks pass.
 */
static int verify_team(const options *o, int *team_out)
{
    int team = 0, bad = 0;
    #pragma omp parallel default(none) shared(o, team, bad, g_thread_ids, g_thread_lps)
    {
        const int t = omp_get_thread_num();
        #pragma omp single
        team = omp_get_num_threads();
        int ok = current_thread_id() == g_thread_ids[t];
        if (o->pin)
            ok = ok && current_processor() == g_thread_lps[t];
        if (!ok) {
            #pragma omp atomic
            ++bad;
        }
    }
    *team_out = team;
    return bad == 0;
}

static int prepare_inputs(const options *o, run_state *s, double *A, double *B)
{
    if (!o->input_dir) {
        generate_inputs(A, B, o->n, o->seed);
        snprintf(s->input_source, sizeof s->input_source, "generated");
        return 0;
    }
    char pa[1024], pb[1024];
    snprintf(pa, sizeof pa, "%s/A_n%d_s%llu.bin", o->input_dir, o->n, (unsigned long long)o->seed);
    snprintf(pb, sizeof pb, "%s/B_n%d_s%llu.bin", o->input_dir, o->n, (unsigned long long)o->seed);
    FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb");
    const int both_exist = fa && fb;
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    if (both_exist) {
        uint64_t sa = 0, sb = 0;
        if (load_matrix(pa, A, o->n, &sa) || load_matrix(pb, B, o->n, &sb) ||
            sa != o->seed || sb != o->seed) {
            fprintf(stderr, "input files for n=%d are invalid\n", o->n);
            return -1;
        }
        snprintf(s->input_source, sizeof s->input_source, "loaded");
        return 0;
    }
    generate_inputs(A, B, o->n, o->seed);
    if (save_matrix(pa, A, o->n, o->seed) || save_matrix(pb, B, o->n, o->seed))
        return -1;
    snprintf(s->input_source, sizeof s->input_source, "generated+saved");
    return 0;
}

static void run_kernel(method_t m, const double *A, const double *B, double *C, int n)
{
    switch (m) {
    case METHOD_SEQ:   matmul_seq(A, B, C, n); break;
    case METHOD_OMP:   matmul_omp(A, B, C, n); break;
    case METHOD_NAIVE: matmul_naive_ijk(A, B, C, n); break;
    }
}

int main(int argc, char **argv)
{
    /* Lets scripts obtain the exact CSV schema without duplicating it. */
    if (argc == 2 && !strcmp(argv[1], "--print-csv-header")) {
        printf("%s\n", CSV_HEADER);
        return 0;
    }

    options o;
    if (parse_args(argc, argv, &o)) {
        usage(argv[0]);
        return 2;
    }
    FILE *csv = open_csv(o.out_path);
    if (!csv)
        return 1;

    run_state s;
    memset(&s, 0, sizeof s);
    s.rss_before_alloc = peak_rss_bytes();
    s.theoretical_bytes = UINT64_C(3) * sizeof(double) * (uint64_t)o.n * (uint64_t)o.n;
    snprintf(s.input_source, sizeof s.input_source, "none");
    cpu_topology_query(&s.topo);
    if (o.pin && s.topo.n_order == 0) {
        fprintf(stderr, "--pin cores requested but the CPU topology is unavailable\n");
        write_row(csv, &o, &s, 0, 0, "ERROR", -1, -1.0, 0, 0.0);
        fclose(csv);
        return 1;
    }

    /* OpenMP configuration: no dynamic adjustment, exactly p threads. */
    omp_set_dynamic(0);
    const int p = o.method == METHOD_OMP ? o.threads : 1;
    omp_set_num_threads(p);
    s.omp_max_threads = omp_get_max_threads();
    s.omp_dynamic = omp_get_dynamic();
    s.oversubscribed = s.topo.logical_procs > 0 && p > s.topo.logical_procs;

    if (s.theoretical_bytes > o.mem_budget) {
        fprintf(stderr, "n=%d needs %llu bytes > memory budget %llu\n", o.n,
                (unsigned long long)s.theoretical_bytes, (unsigned long long)o.mem_budget);
        write_row(csv, &o, &s, 0, 0, "SKIPPED_MEMORY_BUDGET", -1, -1.0, 0, 0.0);
        fclose(csv);
        return 0;
    }

    double *A = alloc_matrix(o.n), *B = alloc_matrix(o.n), *C = alloc_matrix(o.n);
    if (!A || !B || !C || prepare_inputs(&o, &s, A, B)) {
        fprintf(stderr, "allocation or input preparation failed for n=%d\n", o.n);
        write_row(csv, &o, &s, 0, 0, "ERROR", -1, -1.0, 0, 0.0);
        free_matrix(A); free_matrix(B); free_matrix(C);
        fclose(csv);
        return 1;
    }
    /* Touch C so first-touch page faults are not attributed to the kernel. */
    memset(C, 0, (size_t)o.n * (size_t)o.n * sizeof(double));

    /* Thread team: for seq/naive the "team" is the main thread only. */
    if (o.method == METHOD_OMP) {
        s.p_actual = setup_team(&o, &s);
    } else {
        s.p_actual = 1;
        g_thread_ids[0] = current_thread_id();
        g_thread_lps[0] = o.pin ? topology_lp_for_thread(&s.topo, 0) : -1;
        if (o.pin && pin_current_thread(g_thread_lps[0]) != 0)
            s.p_actual = -1;
    }
    if (s.p_actual < 0) {
        fprintf(stderr, "thread pinning failed\n");
        write_row(csv, &o, &s, 0, 0, "ERROR", -1, -1.0, 0, 0.0);
        free_matrix(A); free_matrix(B); free_matrix(C);
        fclose(csv);
        return 1;
    }
    if (o.pin) {
        size_t used = 0;
        for (int t = 0; t < s.p_actual && used + 8 < sizeof s.pinned_lps; ++t)
            used += (size_t)snprintf(s.pinned_lps + used, sizeof s.pinned_lps - used,
                                     t ? ";%d" : "%d", g_thread_lps[t]);
    } else {
        snprintf(s.pinned_lps, sizeof s.pinned_lps, "none");
    }

    int exit_code = 0;
    const int total = o.warmup + o.reps;
    for (int r = 0; r < total; ++r) {
        const int is_warmup = r < o.warmup;
        const int rep = is_warmup ? r + 1 : r - o.warmup + 1;

        /* Untimed check of team size and affinity before every repetition. */
        int affinity_ok = 1;
        if (o.method == METHOD_OMP) {
            int team = 0;
            affinity_ok = verify_team(&o, &team);
            s.p_actual = team;
        } else if (o.pin) {
            affinity_ok = current_processor() == g_thread_lps[0];
        }

        const double t0 = now_seconds();
        run_kernel(o.method, A, B, C, o.n);
        const double t1 = now_seconds();
        const double elapsed = t1 - t0;

        const double checksum = matrix_checksum(C, (size_t)o.n * (size_t)o.n);
        const int timed_out = elapsed > o.time_budget;
        const int mismatch = o.method == METHOD_OMP && s.p_actual != o.threads;
        if (mismatch)
            fprintf(stderr, "warning: team size %d != requested %d\n", s.p_actual, o.threads);
        if (!affinity_ok)
            fprintf(stderr, "warning: thread identity/affinity check failed\n");
        /* The measured time is always kept; the status flags contaminated rows. */
        const char *status = timed_out ? "TIMEOUT"
                           : mismatch ? "P_MISMATCH"
                           : !affinity_ok ? "AFFINITY_FAIL" : "OK";
        write_row(csv, &o, &s, rep, is_warmup, status, affinity_ok, elapsed, 1, checksum);
        if (!isfinite(checksum)) {
            fprintf(stderr, "non-finite checksum\n");
            exit_code = 1;
        }
        if (timed_out) {
            fprintf(stderr, "repetition took %.6g s > budget %.6g s: stopping\n",
                    elapsed, o.time_budget);
            exit_code = 3;
            break;
        }
    }

    free_matrix(A); free_matrix(B); free_matrix(C);
    fclose(csv);
    return exit_code;
}
