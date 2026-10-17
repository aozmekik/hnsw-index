Synthetic data: 100000 base + 1000 disjoint queries, dim 128, seed 42; M=16, efConstruction=200, k=10; 8 thread(s) for parallel runs; median of 3 timed run(s); kernel backend: neon; macos aarch64. Wall-clock throughput depends on what else runs on the machine; CPU-time columns count only this process's CPU time.

| dataset | build | threads | wall time (s) | CPU time (s) | heap bytes / vector (measured) | memory_usage() / vector |
|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | 8 | 6.57 | 40.70 | 715 | 715 |
| gaussian | heuristic, sequential | 1 | 47.79 | 42.46 | 701 | 701 |
| gaussian | simple, parallel | 8 | 8.87 | 33.08 | 715 | 715 |
| clustered | heuristic, parallel | 8 | 4.27 | 13.18 | 715 | 715 |
| clustered | heuristic, sequential | 1 | 17.74 | 13.45 | 701 | 701 |
| clustered | simple, parallel | 8 | 2.91 | 12.45 | 715 | 715 |

| dataset | index | distances | ef | recall@10 | QPS, 1 thread (wall) | QPS, 1 thread (per CPU-second) | QPS, 8 threads (wall) |
|---|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | f32 | 16 | 0.2225 | 19448 | 19581 | 106649 |
| gaussian | heuristic, parallel | f32 | 32 | 0.3358 | 11918 | 11974 | 66230 |
| gaussian | heuristic, parallel | f32 | 64 | 0.4773 | 6626 | 6671 | 27400 |
| gaussian | heuristic, parallel | f32 | 128 | 0.6282 | 3742 | 3785 | 17605 |
| gaussian | heuristic, parallel | f32 | 256 | 0.7712 | 1761 | 1795 | 8755 |
| gaussian | heuristic, sequential | f32 | 16 | 0.2208 | 16673 | 17915 | 50766 |
| gaussian | heuristic, sequential | f32 | 32 | 0.3296 | 8955 | 9464 | 44227 |
| gaussian | heuristic, sequential | f32 | 64 | 0.4726 | 5130 | 5530 | 18607 |
| gaussian | heuristic, sequential | f32 | 128 | 0.6271 | 2950 | 3109 | 14126 |
| gaussian | heuristic, sequential | f32 | 256 | 0.7671 | 1419 | 1579 | 6945 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 16 | 0.2202 | 36395 | 37955 | 121136 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 32 | 0.3266 | 14514 | 19347 | 40171 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 64 | 0.4740 | 10032 | 11027 | 42785 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 128 | 0.6284 | 2363 | 5426 | 22375 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 256 | 0.7646 | 2266 | 2755 | 14345 |
| gaussian | simple, parallel | f32 | 16 | 0.2412 | 17564 | 19630 | 62084 |
| gaussian | simple, parallel | f32 | 32 | 0.3501 | 9477 | 10994 | 35020 |
| gaussian | simple, parallel | f32 | 64 | 0.4782 | 4279 | 6036 | 31721 |
| gaussian | simple, parallel | f32 | 128 | 0.6145 | 3129 | 3434 | 12179 |
| gaussian | simple, parallel | f32 | 256 | 0.7491 | 1752 | 1882 | 3538 |
| clustered | heuristic, parallel | f32 | 16 | 0.7848 | 23699 | 24375 | 80784 |
| clustered | heuristic, parallel | f32 | 32 | 0.9219 | 11392 | 13854 | 43947 |
| clustered | heuristic, parallel | f32 | 64 | 0.9869 | 9946 | 10178 | 19131 |
| clustered | heuristic, parallel | f32 | 128 | 0.9991 | 6666 | 7031 | 23010 |
| clustered | heuristic, parallel | f32 | 256 | 0.9999 | 4514 | 5057 | 18444 |
| clustered | heuristic, sequential | f32 | 16 | 0.7938 | 23606 | 24027 | 76767 |
| clustered | heuristic, sequential | f32 | 32 | 0.9253 | 15603 | 15813 | 69091 |
| clustered | heuristic, sequential | f32 | 64 | 0.9867 | 8411 | 10302 | 31334 |
| clustered | heuristic, sequential | f32 | 128 | 0.9988 | 6331 | 7194 | 28844 |
| clustered | heuristic, sequential | f32 | 256 | 1.0000 | 4907 | 5282 | 25124 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 16 | 0.7939 | 31871 | 39163 | 158668 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 32 | 0.9236 | 24687 | 24996 | 93327 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 64 | 0.9865 | 14953 | 15192 | 77630 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 128 | 0.9986 | 9196 | 9418 | 39153 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 256 | 1.0000 | 5944 | 6126 | 30063 |
| clustered | simple, parallel | f32 | 16 | 0.6855 | 23805 | 26222 | 107063 |
| clustered | simple, parallel | f32 | 32 | 0.7911 | 14522 | 16735 | 71570 |
| clustered | simple, parallel | f32 | 64 | 0.8514 | 9316 | 10941 | 44008 |
| clustered | simple, parallel | f32 | 128 | 0.8822 | 6559 | 7127 | 36904 |
| clustered | simple, parallel | f32 | 256 | 0.9043 | 4741 | 5137 | 24938 |

| dataset | exact brute force QPS |
|---|---:|
| gaussian | 2906 (8 threads) |
| clustered | 1276 (8 threads) |
