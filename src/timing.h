#ifndef TIMING_H
#define TIMING_H

#include <stdint.h>

/*
 * Monotonic wall-clock time in seconds from an arbitrary origin. Only
 * differences between two calls are meaningful.
 *   Windows: QueryPerformanceCounter.  POSIX: clock_gettime(CLOCK_MONOTONIC).
 */
double now_seconds(void);

/* Resolution of now_seconds() in seconds. */
double timer_resolution(void);

/*
 * Peak resident memory of the whole process so far, in bytes; 0 if unknown.
 *   Windows: GetProcessMemoryInfo().PeakWorkingSetSize.
 *   Linux:   getrusage(RUSAGE_SELF).ru_maxrss (KiB) * 1024.
 */
uint64_t peak_rss_bytes(void);

/* Name of the peak-memory metric/tool, for reporting. */
const char *peak_rss_source(void);

#endif /* TIMING_H */
