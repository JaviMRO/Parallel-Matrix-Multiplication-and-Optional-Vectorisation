# Naive i-j-k vs sequential i-k-j

Source: `results/raw/bench_full_quiet.csv`

| n | t_naive_median_s | t_seq_median_s | gflops_naive | gflops_seq | naive_over_seq |
|---|---|---|---|---|---|
| 128 | 0.00151 | 6.080e-04 | 2.778 | 6.899 | 2.483 |
| 256 | 0.02464 | 0.005108 | 1.362 | 6.57 | 4.823 |
| 512 | 0.1802 | 0.0397 | 1.49 | 6.761 | 4.539 |
| 1024 | 1.518 | 0.3326 | 1.415 | 6.457 | 4.563 |
