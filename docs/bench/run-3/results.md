Synthetic data: 100000 base + 1000 disjoint queries, dim 128, seed 42; M=16, efConstruction=200, k=10; 8 thread(s) for parallel runs; median of 3 timed run(s); kernel backend: neon; macos aarch64; 1-minute load average 12.3 at the start and 20.1 at the end of the run. Wall-clock throughput depends on what else runs on the machine; CPU-time columns count only this process's CPU time.

| dataset | build | threads | wall time (s) | CPU time (s) | heap bytes / vector (measured) | memory_usage() / vector |
|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | 8 | 7.17 | 39.85 | 715 | 715 |
| gaussian | heuristic, sequential | 1 | 37.16 | 36.17 | 701 | 701 |
| gaussian | simple, parallel | 8 | 7.79 | 32.84 | 715 | 715 |
| clustered | heuristic, parallel | 8 | 2.52 | 12.65 | 715 | 715 |
| clustered | heuristic, sequential | 1 | 12.60 | 12.08 | 701 | 701 |
| clustered | simple, parallel | 8 | 3.40 | 13.14 | 715 | 715 |

| dataset | index | distances | ef | recall@10 | QPS, 1 thread (wall) | QPS, 1 thread (per CPU-second) | QPS, 8 threads (wall) |
|---|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | f32 | 16 | 0.2210 | 22307 | 22467 | 86207 |
| gaussian | heuristic, parallel | f32 | 32 | 0.3329 | 13136 | 13185 | 52887 |
| gaussian | heuristic, parallel | f32 | 64 | 0.4742 | 6403 | 6652 | 31172 |
| gaussian | heuristic, parallel | f32 | 128 | 0.6242 | 3848 | 3866 | 19092 |
| gaussian | heuristic, parallel | f32 | 256 | 0.7689 | 1894 | 1903 | 10528 |
| gaussian | heuristic, sequential | f32 | 16 | 0.2208 | 25502 | 25749 | 90468 |
| gaussian | heuristic, sequential | f32 | 32 | 0.3296 | 11548 | 11802 | 70012 |
| gaussian | heuristic, sequential | f32 | 64 | 0.4726 | 7946 | 7971 | 39494 |
| gaussian | heuristic, sequential | f32 | 128 | 0.6271 | 3716 | 3749 | 19609 |
| gaussian | heuristic, sequential | f32 | 256 | 0.7671 | 1988 | 1997 | 8959 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 16 | 0.2202 | 41856 | 44635 | 62360 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 32 | 0.3266 | 27653 | 27652 | 26491 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 64 | 0.4740 | 8359 | 12354 | 37943 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 128 | 0.6284 | 2950 | 4459 | 20931 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 256 | 0.7646 | 2869 | 3150 | 16642 |
| gaussian | simple, parallel | f32 | 16 | 0.2366 | 23407 | 24615 | 70480 |
| gaussian | simple, parallel | f32 | 32 | 0.3474 | 15980 | 16093 | 57715 |
| gaussian | simple, parallel | f32 | 64 | 0.4787 | 6709 | 7110 | 28586 |
| gaussian | simple, parallel | f32 | 128 | 0.6244 | 4106 | 4163 | 19745 |
| gaussian | simple, parallel | f32 | 256 | 0.7553 | 1973 | 2031 | 10129 |
| clustered | heuristic, parallel | f32 | 16 | 0.7939 | 25249 | 25875 | 80321 |
| clustered | heuristic, parallel | f32 | 32 | 0.9234 | 17919 | 18012 | 62821 |
| clustered | heuristic, parallel | f32 | 64 | 0.9847 | 10219 | 11208 | 36284 |
| clustered | heuristic, parallel | f32 | 128 | 0.9988 | 6988 | 7159 | 27496 |
| clustered | heuristic, parallel | f32 | 256 | 1.0000 | 5629 | 5712 | 24678 |
| clustered | heuristic, sequential | f32 | 16 | 0.7938 | 28113 | 28290 | 102629 |
| clustered | heuristic, sequential | f32 | 32 | 0.9253 | 17323 | 17424 | 70587 |
| clustered | heuristic, sequential | f32 | 64 | 0.9867 | 11002 | 11427 | 26468 |
| clustered | heuristic, sequential | f32 | 128 | 0.9988 | 7073 | 7536 | 22385 |
| clustered | heuristic, sequential | f32 | 256 | 1.0000 | 5884 | 5925 | 20001 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 16 | 0.7939 | 50225 | 50335 | 162906 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 32 | 0.9236 | 29722 | 30144 | 55350 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 64 | 0.9865 | 17521 | 18401 | 63821 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 128 | 0.9986 | 11017 | 11398 | 52535 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 256 | 1.0000 | 6243 | 6630 | 30531 |
| clustered | simple, parallel | f32 | 16 | 0.7105 | 23075 | 23974 | 103727 |
| clustered | simple, parallel | f32 | 32 | 0.8177 | 15016 | 15830 | 57297 |
| clustered | simple, parallel | f32 | 64 | 0.8811 | 10840 | 11414 | 54477 |
| clustered | simple, parallel | f32 | 128 | 0.9178 | 7841 | 7905 | 39538 |
| clustered | simple, parallel | f32 | 256 | 0.9366 | 4706 | 5145 | 16910 |

| dataset | exact brute force QPS |
|---|---:|
| gaussian | 2497 (8 threads) |
| clustered | 2486 (8 threads) |
