Synthetic data: 100000 base + 1000 disjoint queries, dim 128, seed 42; M=16, efConstruction=200, k=10; 8 thread(s) for parallel runs; median of 3 timed run(s); kernel backend: neon; macos aarch64; 1-minute load average 11.5 at the start and 13.9 at the end of the run. Wall-clock throughput depends on what else runs on the machine; CPU-time columns count only this process's CPU time.

| dataset | build | threads | wall time (s) | CPU time (s) | heap bytes / vector (measured) | memory_usage() / vector |
|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | 8 | 6.51 | 38.53 | 715 | 715 |
| gaussian | heuristic, sequential | 1 | 37.71 | 36.19 | 701 | 701 |
| gaussian | simple, parallel | 8 | 7.04 | 32.11 | 715 | 715 |
| clustered | heuristic, parallel | 8 | 3.44 | 12.93 | 715 | 715 |
| clustered | heuristic, sequential | 1 | 11.63 | 11.36 | 701 | 701 |
| clustered | simple, parallel | 8 | 2.79 | 12.23 | 715 | 715 |

| dataset | index | distances | ef | recall@10 | QPS, 1 thread (wall) | QPS, 1 thread (per CPU-second) | QPS, 8 threads (wall) |
|---|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | f32 | 16 | 0.2271 | 23335 | 23432 | 127004 |
| gaussian | heuristic, parallel | f32 | 32 | 0.3390 | 13213 | 13261 | 76874 |
| gaussian | heuristic, parallel | f32 | 64 | 0.4745 | 7397 | 7426 | 39638 |
| gaussian | heuristic, parallel | f32 | 128 | 0.6261 | 3907 | 3922 | 16704 |
| gaussian | heuristic, parallel | f32 | 256 | 0.7684 | 1871 | 1951 | 10648 |
| gaussian | heuristic, sequential | f32 | 16 | 0.2208 | 19824 | 20534 | 41980 |
| gaussian | heuristic, sequential | f32 | 32 | 0.3296 | 7016 | 9730 | 31624 |
| gaussian | heuristic, sequential | f32 | 64 | 0.4726 | 6444 | 6542 | 27403 |
| gaussian | heuristic, sequential | f32 | 128 | 0.6271 | 3319 | 3414 | 10734 |
| gaussian | heuristic, sequential | f32 | 256 | 0.7671 | 1422 | 1651 | 6172 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 16 | 0.2202 | 43861 | 45564 | 117481 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 32 | 0.3266 | 23639 | 24036 | 105712 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 64 | 0.4740 | 14044 | 14305 | 50635 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 128 | 0.6284 | 7406 | 7589 | 35608 |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 256 | 0.7646 | 3620 | 3642 | 20407 |
| gaussian | simple, parallel | f32 | 16 | 0.2327 | 19044 | 20234 | 76655 |
| gaussian | simple, parallel | f32 | 32 | 0.3435 | 13133 | 13590 | 51512 |
| gaussian | simple, parallel | f32 | 64 | 0.4783 | 7912 | 7986 | 31960 |
| gaussian | simple, parallel | f32 | 128 | 0.6177 | 3694 | 3890 | 20824 |
| gaussian | simple, parallel | f32 | 256 | 0.7524 | 1663 | 1846 | 9114 |
| clustered | heuristic, parallel | f32 | 16 | 0.7838 | 23564 | 24285 | 74524 |
| clustered | heuristic, parallel | f32 | 32 | 0.9165 | 16085 | 16258 | 74984 |
| clustered | heuristic, parallel | f32 | 64 | 0.9851 | 10455 | 10703 | 42852 |
| clustered | heuristic, parallel | f32 | 128 | 0.9990 | 8042 | 8083 | 39236 |
| clustered | heuristic, parallel | f32 | 256 | 0.9996 | 6100 | 6122 | 30040 |
| clustered | heuristic, sequential | f32 | 16 | 0.7938 | 24649 | 25315 | 92154 |
| clustered | heuristic, sequential | f32 | 32 | 0.9253 | 16773 | 16867 | 67938 |
| clustered | heuristic, sequential | f32 | 64 | 0.9867 | 10534 | 10865 | 42872 |
| clustered | heuristic, sequential | f32 | 128 | 0.9988 | 8188 | 8232 | 32470 |
| clustered | heuristic, sequential | f32 | 256 | 1.0000 | 5245 | 5405 | 24813 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 16 | 0.7939 | 44920 | 46070 | 91168 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 32 | 0.9236 | 28876 | 28973 | 97470 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 64 | 0.9865 | 15705 | 16297 | 72324 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 128 | 0.9986 | 10833 | 10912 | 54455 |
| clustered | heuristic, sequential | int8 + f32 re-rank | 256 | 1.0000 | 6333 | 6520 | 26330 |
| clustered | simple, parallel | f32 | 16 | 0.6748 | 22353 | 25370 | 90166 |
| clustered | simple, parallel | f32 | 32 | 0.7796 | 11581 | 14659 | 38133 |
| clustered | simple, parallel | f32 | 64 | 0.8447 | 4159 | 8607 | 41144 |
| clustered | simple, parallel | f32 | 128 | 0.8831 | 5978 | 6988 | 25645 |
| clustered | simple, parallel | f32 | 256 | 0.9095 | 4728 | 5186 | 18832 |

| dataset | exact brute force QPS |
|---|---:|
| gaussian | 3022 (8 threads) |
| clustered | 1918 (8 threads) |
