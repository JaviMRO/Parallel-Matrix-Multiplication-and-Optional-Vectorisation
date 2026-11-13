# Vectorisation study: omp medians across repeated runs

Source: `results/raw/bench_simd_quiet2.csv, results/raw/bench_scalar_ctrl_quiet2.csv, results/raw/bench_vcheck_scalar_a.csv, results/raw/bench_vcheck_scalar_b.csv, results/raw/bench_vcheck_simd_a.csv, results/raw/bench_vcheck_simd_b.csv`

| run | build | n | p_requested | median_s | q1_s | q3_s | between_run_spread |
|---|---|---|---|---|---|---|---|
| scalar_ctrl_quiet2 | scalar | 2048 | 8 | 0.6265 | 0.5992 | 0.6346 | 0.04738 |
| vcheck_scalar_a | scalar | 2048 | 8 | 0.6237 | 0.6194 | 0.65 | 0.04738 |
| vcheck_scalar_b | scalar | 2048 | 8 | 0.5973 | 0.5901 | 0.6153 | 0.04738 |
| simd_quiet2 | simd | 2048 | 8 | 0.802 | 0.7885 | 0.8137 | 0.5806 |
| vcheck_simd_a | simd | 2048 | 8 | 0.5552 | 0.5465 | 0.5597 | 0.5806 |
| vcheck_simd_b | simd | 2048 | 8 | 0.4519 | 0.4448 | 0.4858 | 0.5806 |
| scalar_ctrl_quiet2 | scalar | 2048 | 16 | 0.4481 | 0.4364 | 0.4787 | 0.03751 |
| vcheck_scalar_a | scalar | 2048 | 16 | 0.4652 | 0.4244 | 0.4898 | 0.03751 |
| vcheck_scalar_b | scalar | 2048 | 16 | 0.4594 | 0.4438 | 0.4924 | 0.03751 |
| simd_quiet2 | simd | 2048 | 16 | 0.331 | 0.3215 | 0.4239 | 0.07892 |
| vcheck_simd_a | simd | 2048 | 16 | 0.3483 | 0.3212 | 0.389 | 0.07892 |
| vcheck_simd_b | simd | 2048 | 16 | 0.3583 | 0.3249 | 0.3715 | 0.07892 |
| scalar_ctrl_quiet2 | scalar | 3072 | 8 | 2.085 | 2.025 | 2.16 | 0.01608 |
| vcheck_scalar_a | scalar | 3072 | 8 | 2.086 | 1.998 | 2.135 | 0.01608 |
| vcheck_scalar_b | scalar | 3072 | 8 | 2.118 | 2.035 | 2.144 | 0.01608 |
| simd_quiet2 | simd | 3072 | 8 | 2.302 | 2.009 | 2.81 | 0.1024 |
| vcheck_simd_a | simd | 3072 | 8 | 2.115 | 2.015 | 2.219 | 0.1024 |
| vcheck_simd_b | simd | 3072 | 8 | 2.08 | 1.964 | 2.258 | 0.1024 |
| scalar_ctrl_quiet2 | scalar | 3072 | 16 | 1.753 | 1.716 | 1.795 | 0.02374 |
| vcheck_scalar_a | scalar | 3072 | 16 | 1.731 | 1.713 | 1.775 | 0.02374 |
| vcheck_scalar_b | scalar | 3072 | 16 | 1.712 | 1.688 | 1.76 | 0.02374 |
| simd_quiet2 | simd | 3072 | 16 | 1.394 | 1.371 | 1.423 | 0.0354 |
| vcheck_simd_a | simd | 3072 | 16 | 1.346 | 1.315 | 1.374 | 0.0354 |
| vcheck_simd_b | simd | 3072 | 16 | 1.366 | 1.334 | 1.389 | 0.0354 |
