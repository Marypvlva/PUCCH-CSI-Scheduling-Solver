# PUCCH CSI Scheduling Solver

This repository contains a Python prototype and a C++ solver for CSI allocation
on PUCCH under TDD uplink-slot constraints.

For each user, the solver chooses:

```text
(RB, CSI_period, CSI_offset)
```

The allocation is valid only if every periodic CSI transmission is in an uplink
slot and no two users reuse the same `(RB, slot)` resource.

## Recommended Mode

The recommended offline solver is the C++ `universal` mode:

```text
features -> feasible incumbent -> exact assignment/flow start
         -> bounded quality upgrades -> validation
```

The controller chooses a branch from cheap instance features such as user count,
uplink capacity, one-cell candidate availability, occupied-cell pressure, and
candidate cap. Its main invariant is:

```text
If an all-user feasible incumbent is found, do not return a solution with more
unallocated users.
```

This was added after reviewer feedback showed that local search could reduce RB
count while leaving schedulable users unallocated.

## Objective

The challenge objective is:

```text
objective = RB_used * sum_i(CSI_period_i + CSI_SRS_distance_i)
```

Lower is better. The Python prototype also supports proof and calibration tools
such as small-instance branch-and-bound, fixed-RB CP-SAT checks, and an
optimistic occupied-cell lower bound.

## Build And Run

Build the C++ solver:

```bash
c++ -O3 -std=c++17 -march=native pucch_csi_fast.cpp -o pucch_csi_fast
```

Generate benchmark instances with the Python prototype:

```bash
python3 pucch_csi.py --n 450 --seeds 1,2,3 --difficulty large --export-instances instances.jsonl
```

Run the default universal solver:

```bash
./pucch_csi_fast --input-jsonl instances.jsonl --mode universal --cap 64
```

Run the low-latency online construction:

```bash
./pucch_csi_fast --input-jsonl instances.jsonl --mode online --cap 24
```

Run another TDD pattern:

```bash
./pucch_csi_fast --input-jsonl instances.jsonl --mode universal --dl 7 --ul 3
```

## Useful Modes

| Mode | Purpose |
|---|---|
| `universal` | Recommended offline controller. Selects the best branch from instance features. |
| `online` / `sparse-online` | Fast all-user constructive mode for online-style latency. |
| `single-cell-flow` | Exact one-cell assignment kernel. Useful for dense cases. |
| `single-cell-upgrade` | One-cell flow start plus bounded shorter-period upgrades. |
| `assignment-upgrade` | Strong dense all-user start plus quality polish. |
| `priced-coloring` | Classical OR path with slot pricing and coloring-style repair. |
| `pbandit` | Parallel adaptive local-search tier. |
| `adaptive-fast` | Old raw greedy speed baseline, kept for regression testing. |


## Current Benchmark Snapshot

All numbers below are from generated `difficulty=large` reassessment runs.
Times are total runtime unless noted otherwise.

| Scenario | Mode | Avg objective | Avg RB | Avg unallocated | Avg time |
|---|---|---:|---:|---:|---:|
| Offline `N=450, cap=64` | `universal` | 1,025,299 | 14.00 | 0.00 | 1.728 s |
| Offline `N=3700, cap=128` | `universal` | 68,845,715 | 58.00 | 0.00 | 9.264 s |
| Online `N=450, cap=24` | `online` | 1,162,805 | 8.00 | 0.00 | 0.0027 s |

The dense `N=3700` generated benchmark also matches the occupied-cell
multiple-choice knapsack lower bound on every seed in the 10-seed run, so no
better solution exists within that optimistic relaxation for the same generated
instances.

## Documentation

- [Universal solver report](docs/solution_report.md)
- [Response to reviewer feedback](docs/feedback_response.md)

## Notes

- The solver is deterministic for a fixed input and seed-controlled benchmark
  generation.
- The C++ solver validates uplink-only placement and RB-slot collision
  constraints before reporting results.
- Full large-instance optimality is not claimed. Exact CP-SAT checks are used
  only for small instances and bounded residual neighborhoods.
