Summary of 3 runs: median (min–max).

| dataset | build | threads | wall time (s) | CPU time (s) | heap bytes / vector |
|---|---|---:|---:|---:|---:|
| gaussian | heuristic, parallel | 8 | 7.2 (6.5–8.1) | 39.1 (38.5–39.8) | 715 |
| gaussian | heuristic, sequential | 1 | 37.7 (37.2–40.9) | 36.2 (36.2–38.8) | 701 |
| gaussian | simple, parallel | 8 | 7.0 (5.0–7.8) | 32.1 (29.7–32.8) | 715 |
| clustered | heuristic, parallel | 8 | 3.4 (2.5–4.0) | 12.9 (12.7–13.6) | 715 |
| clustered | heuristic, sequential | 1 | 12.6 (11.6–12.7) | 12.1 (11.4–12.4) | 701 |
| clustered | simple, parallel | 8 | 2.8 (2.6–3.4) | 12.6 (12.2–13.1) | 715 |

| dataset | index | distances | ef | recall@10 | QPS per CPU-second, 1 thread | QPS, 1 thread (wall) | QPS, 8 threads (wall) |
|---|---|---|---:|---:|---:|---:|---:|
| gaussian | heuristic, parallel | f32 | 16 | 0.2243 (0.2210–0.2271) | 23k (22k–25k) | 23k (22k–25k) | 88k (86k–127k) |
| gaussian | heuristic, parallel | f32 | 32 | 0.3390 (0.3329–0.3418) | 13k (13k–14k) | 13k (13k–14k) | 53k (44k–77k) |
| gaussian | heuristic, parallel | f32 | 64 | 0.4745 (0.4742–0.4815) | 7.1k (6.7k–7.4k) | 7.0k (6.4k–7.4k) | 31k (29k–40k) |
| gaussian | heuristic, parallel | f32 | 128 | 0.6261 (0.6242–0.6268) | 3.9k (3.7k–3.9k) | 3.8k (3.7k–3.9k) | 17k (14k–19k) |
| gaussian | heuristic, parallel | f32 | 256 | 0.7689 (0.7684–0.7701) | 1.9k (1.7k–2.0k) | 1.9k (1.6k–1.9k) | 11k (8.2k–11k) |
| gaussian | heuristic, sequential | f32 | 16 | 0.2208 | 23k (21k–26k) | 22k (20k–26k) | 42k (42k–90k) |
| gaussian | heuristic, sequential | f32 | 32 | 0.3296 | 11k (9.7k–12k) | 10k (7.0k–12k) | 62k (32k–70k) |
| gaussian | heuristic, sequential | f32 | 64 | 0.4726 | 7.5k (6.5k–8.0k) | 7.2k (6.4k–7.9k) | 36k (27k–39k) |
| gaussian | heuristic, sequential | f32 | 128 | 0.6271 | 3.4k (3.4k–3.7k) | 3.3k (3.1k–3.7k) | 17k (11k–20k) |
| gaussian | heuristic, sequential | f32 | 256 | 0.7671 | 1.8k (1.7k–2.0k) | 1.6k (1.4k–2.0k) | 9.0k (6.2k–9.4k) |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 16 | 0.2202 | 46k (45k–56k) | 44k (42k–55k) | 117k (62k–232k) |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 32 | 0.3266 | 28k (24k–32k) | 28k (24k–31k) | 106k (26k–144k) |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 64 | 0.4740 | 14k (12k–16k) | 14k (8.4k–16k) | 51k (38k–76k) |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 128 | 0.6284 | 7.5k (4.5k–7.6k) | 7.3k (3.0k–7.4k) | 36k (21k–46k) |
| gaussian | heuristic, sequential | int8 + f32 re-rank | 256 | 0.7646 | 3.6k (3.1k–4.1k) | 3.6k (2.9k–4.1k) | 20k (17k–23k) |
| gaussian | simple, parallel | f32 | 16 | 0.2330 (0.2327–0.2366) | 25k (20k–27k) | 23k (19k–27k) | 77k (70k–120k) |
| gaussian | simple, parallel | f32 | 32 | 0.3435 (0.3420–0.3474) | 15k (14k–16k) | 15k (13k–16k) | 58k (52k–75k) |
| gaussian | simple, parallel | f32 | 64 | 0.4783 (0.4773–0.4787) | 8.0k (7.1k–8.0k) | 7.9k (6.7k–8.0k) | 32k (29k–37k) |
| gaussian | simple, parallel | f32 | 128 | 0.6177 (0.6138–0.6244) | 3.9k (3.8k–4.2k) | 3.7k (3.6k–4.1k) | 20k (19k–21k) |
| gaussian | simple, parallel | f32 | 256 | 0.7524 (0.7505–0.7553) | 1.9k (1.8k–2.0k) | 1.9k (1.7k–2.0k) | 10k (9.1k–12k) |
| clustered | heuristic, parallel | f32 | 16 | 0.7924 (0.7838–0.7939) | 25k (24k–26k) | 24k (24k–25k) | 80k (75k–100k) |
| clustered | heuristic, parallel | f32 | 32 | 0.9204 (0.9165–0.9234) | 17k (16k–18k) | 17k (16k–18k) | 75k (63k–81k) |
| clustered | heuristic, parallel | f32 | 64 | 0.9851 (0.9847–0.9859) | 11k | 10k (10k–11k) | 39k (36k–43k) |
| clustered | heuristic, parallel | f32 | 128 | 0.9990 (0.9988–0.9990) | 7.6k (7.2k–8.1k) | 7.5k (7.0k–8.0k) | 28k (27k–39k) |
| clustered | heuristic, parallel | f32 | 256 | 0.9999 (0.9996–1.0000) | 5.7k (5.3k–6.1k) | 5.6k (5.1k–6.1k) | 25k (17k–30k) |
| clustered | heuristic, sequential | f32 | 16 | 0.7938 | 27k (25k–28k) | 26k (25k–28k) | 92k (85k–103k) |
| clustered | heuristic, sequential | f32 | 32 | 0.9253 | 17k | 17k | 71k (68k–81k) |
| clustered | heuristic, sequential | f32 | 64 | 0.9867 | 11k | 11k | 43k (26k–53k) |
| clustered | heuristic, sequential | f32 | 128 | 0.9988 | 7.8k (7.5k–8.2k) | 7.8k (7.1k–8.2k) | 32k (22k–41k) |
| clustered | heuristic, sequential | f32 | 256 | 1.0000 | 5.7k (5.4k–5.9k) | 5.6k (5.2k–5.9k) | 25k (20k–27k) |
| clustered | heuristic, sequential | int8 + f32 re-rank | 16 | 0.7939 | 46k (42k–50k) | 45k (42k–50k) | 148k (91k–163k) |
| clustered | heuristic, sequential | int8 + f32 re-rank | 32 | 0.9236 | 29k (26k–30k) | 29k (24k–30k) | 97k (55k–104k) |
| clustered | heuristic, sequential | int8 + f32 re-rank | 64 | 0.9865 | 16k (16k–18k) | 16k (15k–18k) | 72k (64k–89k) |
| clustered | heuristic, sequential | int8 + f32 re-rank | 128 | 0.9986 | 11k (9.5k–11k) | 11k (7.4k–11k) | 54k (53k–60k) |
| clustered | heuristic, sequential | int8 + f32 re-rank | 256 | 1.0000 | 6.6k (6.5k–6.7k) | 6.3k (6.2k–6.6k) | 31k (26k–33k) |
| clustered | simple, parallel | f32 | 16 | 0.6914 (0.6748–0.7105) | 25k (24k–27k) | 23k (22k–26k) | 101k (90k–104k) |
| clustered | simple, parallel | f32 | 32 | 0.7960 (0.7796–0.8177) | 16k (15k–18k) | 15k (12k–17k) | 57k (38k–83k) |
| clustered | simple, parallel | f32 | 64 | 0.8545 (0.8447–0.8811) | 11k (8.6k–12k) | 11k (4.2k–12k) | 54k (41k–58k) |
| clustered | simple, parallel | f32 | 128 | 0.8847 (0.8831–0.9178) | 7.9k (7.0k–8.2k) | 7.8k (6.0k–8.1k) | 40k (26k–43k) |
| clustered | simple, parallel | f32 | 256 | 0.9095 (0.9080–0.9366) | 5.2k (5.1k–5.2k) | 4.7k (4.7k–5.1k) | 17k (11k–19k) |
