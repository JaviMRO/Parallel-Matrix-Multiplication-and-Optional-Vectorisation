#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L /* clock_gettime, CLOCK_MONOTONIC under -std=c11 */
#endif

#include "timing.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

static double qpc_frequency(void)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq); /* fixed at boot; never fails on XP+ */
    return (double)freq.QuadPart;
}

double now_seconds(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / qpc_frequency();
}

double timer_resolution(void)
{
    return 1.0 / qpc_frequency();
}

uint64_t peak_rss_bytes(void)
{
    PROCESS_MEMORY_COUNTERS pmc = {0};
    pmc.cb = sizeof pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return (uint64_t)pmc.PeakWorkingSetSize;
    return 0;
}

const char *peak_rss_source(void)
{
    return "GetProcessMemoryInfo.PeakWorkingSetSize";
}

#else /* POSIX */

#include <sys/resource.h>
#include <time.h>

double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double timer_resolution(void)
{
    struct timespec ts;
    clock_getres(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

uint64_t peak_rss_bytes(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return (uint64_t)ru.ru_maxrss * 1024u; /* Linux reports KiB */
    return 0;
}

const char *peak_rss_source(void)
{
    return "getrusage.ru_maxrss";
}

#endif
