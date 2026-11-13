#ifndef AFFINITY_H
#define AFFINITY_H

/*
 * CPU topology and manual thread pinning.
 *
 * libgomp (MinGW-w64) ignores OMP_PROC_BIND / OMP_PLACES on Windows
 * ("Affinity not supported on this configuration"), so the benchmark pins its
 * OpenMP threads itself with SetThreadAffinityMask.
 *
 * Pinning order ("cores" mode): first one logical processor (LP) of every
 * physical core, then the second SMT sibling of every core, and so on. With
 * p <= physical cores each thread has a core to itself; with
 * physical < p <= logical the extra threads share cores through SMT; with
 * p > logical, threads t and t + logical share the same LP (oversubscription).
 */

#define AFFINITY_MAX_LP 64 /* one Windows processor group */

typedef struct {
    int physical_cores;          /* -1 if unknown */
    int logical_procs;           /* -1 if unknown */
    int order[AFFINITY_MAX_LP];  /* LP ids in "cores" pinning order */
    int n_order;                 /* valid entries in order[] (0 if unknown) */
} cpu_topology;

/* Fills *topo. Returns 0 on success, -1 if the topology is unavailable. */
int cpu_topology_query(cpu_topology *topo);

/* LP assigned to OpenMP thread `thread` in "cores" mode, or -1 if unknown. */
int topology_lp_for_thread(const cpu_topology *topo, int thread);

/* Restricts the calling thread to logical processor `lp`. 0 on success. */
int pin_current_thread(int lp);

/* LP the calling thread is running on right now, or -1 if unknown. */
int current_processor(void);

/* OS identifier of the calling thread. */
unsigned long current_thread_id(void);

#endif /* AFFINITY_H */
