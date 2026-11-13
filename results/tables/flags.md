# Flagged configurations (superlinear, slowdown, high dispersion, oversubscribed)

Source: `results/raw/bench_full_quiet.csv`

| n | p_requested | speedup | speedup_lo | speedup_hi | efficiency | iqr_rel | flags |
|---|---|---|---|---|---|---|---|
| 128 | 16 | 5.179 | 0.1478 | 5.278 | 0.3237 | 33.68 | high_dispersion |
| 128 | 32 | 4.59 | 4.261 | 4.669 | 0.1435 | 0.0738 | oversubscribed |
| 256 | 8 | 3.045 | 1.954 | 3.425 | 0.3806 | 0.6543 | high_dispersion |
| 256 | 16 | 6.513 | 3.981 | 8.016 | 0.407 | 0.8091 | high_dispersion |
| 256 | 32 | 3.855 | 3.394 | 4.711 | 0.1205 | 0.3056 | high_dispersion;oversubscribed |
| 512 | 2 | 2.062 | 1.978 | 2.156 | 1.031 | 0.02735 | superlinear |
| 512 | 4 | 4.09 | 3.202 | 4.271 | 1.023 | 0.2553 | superlinear;high_dispersion |
| 512 | 32 | 7.647 | 6.87 | 8.337 | 0.239 | 0.1368 | high_dispersion;oversubscribed |
| 1024 | 2 | 2.063 | 2.01 | 2.099 | 1.031 | 0.0185 | superlinear_robust |
| 1024 | 4 | 4.127 | 4.017 | 4.228 | 1.032 | 0.02625 | superlinear_robust |
| 1024 | 6 | 6.218 | 6.088 | 6.377 | 1.036 | 0.0213 | superlinear_robust |
| 1024 | 8 | 8.068 | 7.808 | 8.326 | 1.009 | 0.03926 | superlinear |
| 1024 | 16 | 7.637 | 6.265 | 8.321 | 0.4773 | 0.274 | high_dispersion |
| 1024 | 32 | 8.166 | 7.878 | 8.558 | 0.2552 | 0.05744 | oversubscribed |
| 2048 | 32 | 7.095 | 6.735 | 7.548 | 0.2217 | 0.1068 | high_dispersion;oversubscribed |
| 3072 | 32 | 4.045 | 3.9 | 4.141 | 0.1264 | 0.05677 | oversubscribed |
