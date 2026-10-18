Synthetic data: 100000 base + 1000 disjoint queries, dim 128, seed 42; M=16, efConstruction=200, k=10; 8 thread(s) for parallel runs; median of 3 timed run(s); kernel backend: neon; macos aarch64; 1-minute load average 13.9 at the start and 12.3 at the end of the run. Wall-clock throughput depends on what else runs on the machine; CPU-time columns count only this process's CPU time.

| dataset | build | threads | wall time (s) | CPU time (s) | heap bytes / vector (measured) | memory_usage() / vector |
|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | 8 | 8.08 | 39.06 | 715 | 715 |
| gaussian | heuristic, sequential | 1 | 40.88 | 38.80 | 701 | 701 |
| gaussian | simple, parallel | 8 | 4.96 | 29.66 | 715 | 715 |
| clustered | heuristic, parallel | 8 | 4.03 | 13.59 | 715 | 715 |
| clustered | heuristic, sequential | 1 | 12.70 | 12.39 | 701 | 701 |
| clustered | simple, parallel | 8 | 2.61 | 12.59 | 715 | 715 |

| dataset | index | distances | ef | recall@10 | QPS, 1 thread (wall) | QPS, 1 thread (per CPU-second) | QPS, 8 threads (wall) |
|---|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | f32 | 16 | 0.2243 | 24636 | 24771 | 87537 |
| gaussian | heuristic, parallel | f32 | 32 | 0.3418 | 13809 | 13857 | 44377 |
| gaussian | heuristic, parallel | f32 | 64 | 0.4815 | 7031 | 7086 | 28807 |
| gaussian | heuristic, parallel | f32 | 128 | 0.6268 | 3656 | 3707 | 14162 |
| gaussian | heuristic, parallel | f32 | 256 | 0.7701 | 1610 | 1662 | 8234 |
| gaussian | heuristic, sequential | f32 | 16 | 0.2208 | 22415 | 23283 | 42177 |
| gaussian | heuristic, sequential | f32 | 32 | 0.3296 | 10481 | 11463 | 61791 |
| gaussian | heuristic, sequential | f32 | 64 | 0.4726 | 7177 | 7499 | 35801 |
| gaussian | heuristic, sequential | f32 | 128 | 0.6271 | 3137 | 3412 | 16851 |
| gaussian | heuristic, sequential | f32 | 256 | 0.7671 | 1620 | 1761 | 9366 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 16 | 0.2202 | 55424 | 55729 | 231786 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 32 | 0.3266 | 31453 | 31550 | 143958 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 64 | 0.4740 | 15805 | 15874 | 76211 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 128 | 0.6284 | 7297 | 7472 | 45847 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 256 | 0.7646 | 4091 | 4106 | 22573 |
| gaussian | simple, parallel | f32 | 16 | 0.2330 | 26652 | 26783 | 119956 |
| gaussian | simple, parallel | f32 | 32 | 0.3420 | 14948 | 14995 | 74894 |
| gaussian | simple, parallel | f32 | 64 | 0.4773 | 7978 | 8018 | 37022 |
| gaussian | simple, parallel | f32 | 128 | 0.6138 | 3624 | 3768 | 18789 |
| gaussian | simple, parallel | f32 | 256 | 0.7505 | 1868 | 1906 | 11735 |
| clustered | heuristic, parallel | f32 | 16 | 0.7924 | 24343 | 24501 | 100163 |
| clustered | heuristic, parallel | f32 | 32 | 0.9204 | 16590 | 16707 | 80551 |
| clustered | heuristic, parallel | f32 | 64 | 0.9859 | 11056 | 11263 | 39297 |
| clustered | heuristic, parallel | f32 | 128 | 0.9990 | 7459 | 7603 | 28322 |
| clustered | heuristic, parallel | f32 | 256 | 0.9999 | 5124 | 5330 | 16668 |
| clustered | heuristic, sequential | f32 | 16 | 0.7938 | 26271 | 26569 | 85426 |
| clustered | heuristic, sequential | f32 | 32 | 0.9253 | 16749 | 16886 | 81291 |
| clustered | heuristic, sequential | f32 | 64 | 0.9867 | 11185 | 11251 | 53102 |
| clustered | heuristic, sequential | f32 | 128 | 0.9988 | 7785 | 7848 | 41473 |
| clustered | heuristic, sequential | f32 | 256 | 1.0000 | 5601 | 5659 | 26688 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 16 | 0.7939 | 41570 | 42324 | 147548 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 32 | 0.9236 | 24396 | 25799 | 103743 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 64 | 0.9865 | 15361 | 15970 | 89096 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 128 | 0.9986 | 7413 | 9500 | 60420 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 256 | 1.0000 | 6615 | 6699 | 33169 |
| clustered | simple, parallel | f32 | 16 | 0.6914 | 26407 | 26759 | 101267 |
| clustered | simple, parallel | f32 | 32 | 0.7960 | 17351 | 17580 | 83388 |
| clustered | simple, parallel | f32 | 64 | 0.8545 | 11652 | 11807 | 58367 |
| clustered | simple, parallel | f32 | 128 | 0.8847 | 8081 | 8177 | 42857 |
| clustered | simple, parallel | f32 | 256 | 0.9080 | 5126 | 5236 | 10628 |

| dataset | exact brute force QPS |
|---|---:|
| gaussian | 1717 (8 threads) |
| clustered | 2648 (8 threads) |
