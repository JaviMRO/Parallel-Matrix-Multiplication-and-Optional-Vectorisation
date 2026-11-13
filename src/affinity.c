#ifndef _WIN32
#define _GNU_SOURCE /* sched_getcpu, sysconf(_SC_NPROCESSORS_ONLN) */
#endif

#include "affinity.h"

#include <stdlib.h>
#include <string.h>

int topology_lp_for_thread(const cpu_topology *topo, int thread)
{
    if (topo->n_order <= 0 || thread < 0)
        return -1;
    return topo->order[thread % topo->n_order];
}

#ifdef _WIN32

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 /* GetCurrentProcessorNumber needs Vista+ */
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int cpu_topology_query(cpu_topology *topo)
{
    memset(topo, 0, sizeof *topo);
    topo->physical_cores = -1;
    topo->logical_procs = -1;

    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len == 0)
        return -1;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *info = malloc(len);
    if (!info)
        return -1;
    if (!GetLogicalProcessorInformation(info, &len)) {
        free(info);
        return -1;
    }
    const int entries = (int)(len / sizeof *info);

    /* Collect the LP ids of every physical core (ascending bit order). */
    int core_lps[AFFINITY_MAX_LP][AFFINITY_MAX_LP];
    int core_size[AFFINITY_MAX_LP];
    int cores = 0, logical = 0, max_smt = 0;
    for (int e = 0; e < entries && cores < AFFINITY_MAX_LP; ++e) {
        if (info[e].Relationship != RelationProcessorCore)
            continue;
        ULONG_PTR mask = info[e].ProcessorMask;
        core_size[cores] = 0;
        for (int bit = 0; bit < AFFINITY_MAX_LP; ++bit)
            if (mask & ((ULONG_PTR)1 << bit))
                core_lps[cores][core_size[cores]++] = bit;
        logical += core_size[cores];
        if (core_size[cores] > max_smt)
            max_smt = core_size[cores];
        ++cores;
    }
    free(info);
    if (cores == 0)
        return -1;

    /* Round-robin over cores: sibling 0 of every core, then sibling 1, ... */
    int k = 0;
    for (int s = 0; s < max_smt; ++s)
        for (int c = 0; c < cores; ++c)
            if (s < core_size[c] && k < AFFINITY_MAX_LP)
                topo->order[k++] = core_lps[c][s];

    topo->physical_cores = cores;
    topo->logical_procs = logical;
    topo->n_order = k;
    return 0;
}

int pin_current_thread(int lp)
{
    if (lp < 0 || lp >= AFFINITY_MAX_LP)
        return -1;
    DWORD_PTR mask = (DWORD_PTR)1 << lp;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0 ? 0 : -1;
}

int current_processor(void)
{
    return (int)GetCurrentProcessorNumber();
}

unsigned long current_thread_id(void)
{
    return (unsigned long)GetCurrentThreadId();
}

#else /* POSIX: topology/pinning not implemented; use OMP_PROC_BIND/OMP_PLACES there. */

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

int cpu_topology_query(cpu_topology *topo)
{
    memset(topo, 0, sizeof *topo);
    topo->physical_cores = -1;
    long lp = sysconf(_SC_NPROCESSORS_ONLN);
    topo->logical_procs = lp > 0 ? (int)lp : -1;
    return -1;
}

int pin_current_thread(int lp)
{
    (void)lp;
    return -1;
}

int current_processor(void)
{
    return sched_getcpu();
}

unsigned long current_thread_id(void)
{
    return (unsigned long)pthread_self();
}

#endif
