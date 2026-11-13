# OpenMP overhead at p = 1: T_omp(1) / T_seq

Source: `results/raw/bench_full_quiet.csv`

| n | t_seq_median_s | t_omp1_median_s | overhead_ratio | ratio_lo | ratio_hi | ratio_range_includes_1 |
|---|---|---|---|---|---|---|
| 128 | 6.080e-04 | 6.145e-04 | 1.011 | 0.9862 | 1.057 | True |
| 256 | 0.005108 | 0.005218 | 1.022 | 1.003 | 1.042 | False |
| 512 | 0.0397 | 0.03878 | 0.9767 | 0.9341 | 1.035 | True |
| 1024 | 0.3326 | 0.3243 | 0.975 | 0.9595 | 1.017 | True |
| 2048 | 3.496 | 3.543 | 1.014 | 1.005 | 1.023 | False |
| 3072 | 12.24 | 12.28 | 1.003 | 0.9977 | 1.007 | True |
