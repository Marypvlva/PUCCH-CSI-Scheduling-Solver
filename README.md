# PUCCH CSI Scheduling Solver

Single-file Python prototype for CSI allocation on PUCCH under TDD uplink-slot
constraints.

The solver treats each user assignment as `(RB, CSI period, CSI offset)` and
validates that every periodic CSI transmission is placed in an uplink slot and
that no `(RB, slot)` pair is reused.

## Implemented approaches

- Baseline greedy placement by local CSI cost `p + d`.
- Official baseline-style greedy placement by CSI-SRS distance.
- Greedy placement with adaptive new-RB penalty.
- Greedy placement with the true marginal product objective.
- Bounded RB-budget greedy sweep, mirroring the CP-SAT fixed-K formulation.
- Lightweight learned RB-budget helper: an integer calibrated model predicts a
  compact RB budget, then the deterministic bitset scheduler enforces all
  constraints.
- Fast portfolio over a few calibrated adaptive penalties, marginal scoring,
  and the learned helper. This improves quality while staying below a second on
  the tested synthetic large cases.
- Whole-RB eviction refinement: removes every user from a selected active RB
  and accepts the move only if all removed users can be reinserted into the
  remaining RBs with a better objective.
- Adaptive multi-operator ALNS refinement using random, worst-quality,
  crowded-RB, light-RB, and blocker-style destroy operators with simple online
  operator reweighting.
- Annealed ALNS refinement with regret repair: a slower variant that combines
  adaptive destroy/repair selection, dynamic regret insertion, simulated
  annealing-style acceptance, and related-removal operators based on periodic
  time-mask overlap.
- Contextual Thompson bandit ALNS with conflict-graph destroy operators and
  conflict-aware regret repair.
- Optional CP-SAT-guided repair of plateau/RB-compression neighborhoods for
  small or final-offline exact neighborhood improvement.
- C++ shared-incumbent parallel contextual bandit mode (`pbandit`) for a faster
  speed/quality tier.
- CP-SAT-guided whole-RB eviction for small/medium offline accuracy checks.
- Static regret ordering using the gap between the best and second-best
  placement.
- Optional local search and LNS-style destroy/repair portfolio.
- Small-instance branch-and-bound to estimate optimality gaps.
- OR-Tools CP-SAT model with fixed-RB-budget sweep for exact small-instance
  comparisons.
- Built-in feasibility validator and comparative benchmark harness.

## Objectives

The product objective is:

```text
RB_used * sum_i(period_i + distance_i)
```

CSI-SRS distance follows the task statement's modulo-offset definition, not the
minimum absolute distance between finite generated slot lists.

The code also supports a lexicographic objective:

```text
minimize RB_used first, then sum_i(period_i + distance_i)
```

This is useful because it separates RB compaction from CSI/SRS timing quality
and may be easier to justify than a multiplicative scalar objective.

## Run

```bash
python3 pucch_csi.py
```

Build and run the C++ fast adaptive kernel:

```bash
c++ -O3 -std=c++17 -march=native pucch_csi_fast.cpp -o pucch_csi_fast
python3 pucch_csi.py --n 450 --seeds 1,2,3 --difficulty large --export-instances /tmp/pucch_instances.jsonl
./pucch_csi_fast --input-jsonl /tmp/pucch_instances.jsonl --cap 24 --base-penalty 60
```

Run the C++ annealed ALNS accuracy/speed middle path:

```bash
./pucch_csi_fast --input-jsonl /tmp/pucch_instances.jsonl --mode annealed --cap 64 --base-penalty 60 --rounds 28
```

Run the current best C++ speed/quality mode:

```bash
./pucch_csi_fast --input-jsonl /tmp/pucch_instances.jsonl --mode pbandit --workers 2 --cap 64 --rounds 28
```

Compare constructive methods on larger synthetic instances:

```bash
python3 pucch_csi.py --n 100,300 --seeds 1,2,3 --difficulty large
```

Include the slower bounded RB-budget sweep:

```bash
python3 pucch_csi.py --n 50,250,450 --seeds 1 --difficulty large --include-budget-sweep
```

Try the learned budget helper:

```bash
python3 pucch_csi.py --n 50,250,450 --seeds 1 --difficulty large --include-learned
```

Try the fast tuned portfolio:

```bash
python3 pucch_csi.py --n 100,300,500 --seeds 1,2,3 --difficulty large --include-fast-portfolio
```

Try the slower whole-RB eviction accuracy pass:

```bash
python3 pucch_csi.py --n 50,250,450 --seeds 1,2,3 --difficulty large --include-fast-portfolio --include-rb-eviction
```

Try the adaptive ALNS accuracy pass:

```bash
python3 pucch_csi.py --n 50,250,450 --seeds 1,2,3 --difficulty large --include-fast-portfolio --include-adaptive-alns
```

Try the slower annealed ALNS accuracy pass:

```bash
python3 pucch_csi.py --n 50,250,450 --seeds 1,2,3 --difficulty large --include-fast-portfolio --include-annealed-alns
```

Try the current best Python accuracy pass:

```bash
python3 pucch_csi.py --n 250 --seeds 1,2,3 --difficulty large --include-contextual-ts-bandit-alns
```

Try optional CP-SAT-guided plateau/RB-compression repair:

```bash
python3 pucch_csi.py --n 50 --seeds 1,2 --difficulty large --include-contextual-ts-cpsat
```

Try CP-SAT-guided RB eviction on a small proof-calibration case:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 pucch_csi.py --n 25 --seeds 1 --difficulty medium --include-cpsat-rb-eviction
```

Run a stronger exact CP-SAT proof on a small instance:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 pucch_csi.py --cpsat-small --exact-n 20 --exact-seed 1 --difficulty medium --cpsat-time 5 --cpsat-max-rb 5
```

By default, heuristics keep the best 64 precomputed CSI period/offset candidates
per user. Use `--time-option-cap 0` to disable this pruning for full candidate
search.

Report post-scheduling uplink-grid occupancy, which is useful for constructing
the task's 10%, 50%, and 90% baseline-load test cases:

```bash
python3 pucch_csi.py --n 100,500 --seeds 1 --difficulty large
```

Optionally test with an already occupied uplink grid:

```bash
python3 pucch_csi.py --n 100 --seeds 1 --occupied-fraction 0.5
```

Use lexicographic objective:

```bash
python3 pucch_csi.py --n 100 --seeds 1 --objective lexicographic
```

Include the slower hybrid LNS portfolio:

```bash
python3 pucch_csi.py --n 25 --seeds 1 --include-hybrid
```

Run a small branch-and-bound optimality-gap check:

```bash
python3 pucch_csi.py --exact-small --exact-n 4 --exact-seed 1
```

Run the stronger CP-SAT exact sweep:

```bash
python3 pucch_csi.py --cpsat-small --exact-n 12 --exact-seed 1 --cpsat-time 2 --cpsat-max-rb 4
```

Report fixed-RB lower-bound evidence:

```bash
python3 pucch_csi.py --cpsat-lower-bound --exact-n 8 --exact-seed 1 --difficulty medium --cpsat-time 0.5
```

## Notes on claims

The constructive methods scale to hundreds of generated users in seconds on a
laptop-class CPU. This is useful for offline algorithm exploration, but it does
not meet the task document's real-system target of less than 0.5 ms. A production
implementation would need a lower-level language, precomputed tables, and a
strict online path.

The C++ fast adaptive kernel is the first lower-level implementation. With the
safer `--cap 64` setting on the same exported Python-generated `N=450`
instances, it matched Python adaptive objectives and ran in about 3.5 ms average
versus about 80 ms for Python. With the real-time `--cap 24` setting, the hot
prepared scheduling loop ran below 0.5 ms on the tested `N=50,250,450,500`
synthetic cases, with competitive or better product objective on average.

The C++ kernel also has an experimental `--mode annealed` path that ports the
annealed ALNS repair loop. On generated `N=50,250,450`, `difficulty=large`,
`seeds=1,2,3`, using `--cap 64 --rounds 28`, it averaged `59.5 ms` hot solve
time across all nine cases, with `N=450` cases around `122-177 ms`. This is far
faster than Python ALNS, but currently less accurate than the Python related
annealed implementation. With `--cap 24 --rounds 28`, the average hot solve time
was `24.4 ms`; with `--cap 24 --rounds 100`, it was `83.0 ms`. Treat this as a
speed/quality middle tier, not the final best-quality solver.

The fast path removes symmetric empty-RB scans: during construction it evaluates
already active RBs plus one representative empty RB. This preserves the packing
choice for empty RBs while greatly reducing redundant feasibility checks.

The learned helper is intentionally only a budget predictor. It does not predict
final allocations, because feasibility is a hard Boolean constraint. The
deterministic scheduler still performs every collision and uplink-slot check.
The helper always keeps the unconstrained adaptive greedy solution as a fallback,
so it should not introduce unscheduled users when the adaptive path schedules
everyone.

The hybrid portfolio can improve objective values on small and medium instances,
but it is currently slower and should be reported separately from the fast greedy
methods.

The branch-and-bound routine is intended only for small instances. For stronger
SOTA comparisons, use the OR-Tools CP-SAT sweep to report optimality gaps on
small cases plus runtime/quality tradeoffs on larger synthetic and official
challenge datasets.

Current exact checks show that the fast heuristics are not near-optimal on all
small instances. For example, CP-SAT proves `N=20, seed=1, medium` has optimum
objective `2275`, while the best heuristic in that demo reports `2910`
(`27.91%` gap). The gap is mostly from opening extra RBs; offline accuracy work
should therefore focus on CP-SAT/LNS RB-compression repair rather than pure
ordering tweaks.

With a longer fixed-K CP-SAT run, `N=25, seed=1, medium` is solved exactly up to
`K=3` in about 46 seconds. The optimum under the product objective is `3479` at
one RB, while the heuristic demo solution is `4227` (`21.50%` gap). This is a
useful proof-calibration case, but not a scalable production path.

The whole-RB eviction pass is a cheaper version of the same idea for larger
instances. On generated `N=50,250,450`, `difficulty=large`, `seeds=1,2,3`, it
matched the fast portfolio at `N=50` and `N=250`, and improved the `N=450`
average objective from `1,757,618.7` to `1,746,940.3` by evacuating one RB on a
dense seed. Runtime was around one second in Python for `N=450`, so it should be
reported as an offline accuracy refinement.

The adaptive ALNS pass is currently the strongest practical Python accuracy
mode. On generated `N=50,250,450`, `difficulty=large`, `seeds=1,2,3`, it improved
the fast portfolio averages from `21,308.3` to `20,423.3` at `N=50`, from
`534,854.7` to `501,004.0` at `N=250`, and from `1,757,618.7` to `1,639,122.3`
at `N=450`. Runtime was about `0.44 s`, `1.42 s`, and `2.36 s` respectively in
Python, so this remains an offline or batch refinement.

The annealed ALNS pass is the highest-quality Python mode so far under the
product objective. It uses regret repair and simulated-annealing acceptance.
On the same generated `N=50,250,450`, `difficulty=large`, `seeds=1,2,3`
benchmark, the related-removal version improved averages further to `20,110.0`,
`459,483.0`, and `1,481,212.7`. Runtime increased to about `0.41 s`, `3.17 s`,
and `8.18 s`. It is especially strong on dense cases, where `N=450` average RB
usage dropped from `41.00` in the fast portfolio to `34.67`. Report it
separately from real-time modes.

Under lexicographic RB-first scoring, annealed ALNS is more mixed at small size
but now clearly useful at larger size. On generated `N=250`, it improved the
fast portfolio from `533,696.0` to `465,701.7` average product objective and
reduced average RB count from `22.00` to `19.33`.

For lexicographic RB-first scoring, the fast portfolio includes regret ordering
because it can reduce the RB count even when it increases timing quality. On
generated `N=50`, `difficulty=large`, `seeds=1,2,3`, lexicographic fast
portfolio averaged `4.67` RBs versus `5.00` RBs before this candidate was added;
adaptive ALNS kept the same average RB count and reduced average quality from
`4589.7` to `4296.3`.

CP-SAT-guided whole-RB eviction can close part of the exactness gap on small
instances, but it is expensive. On `N=25, seed=1, medium`, it improved the fast
portfolio from `3 RB / 4485` to `2 RB / 3780` in about `18.9 s`. The full
fixed-K CP-SAT proof still finds `1 RB / 3479`, so this method is useful but not
a complete optimality proof. On `N=50, seed=1, large`, it took about `31.3 s`
and did not improve the fast portfolio.

An experimental CP-SAT LNS repair is available:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 pucch_csi.py --n 50 --seeds 1 --difficulty large --include-cpsat-lns
```

In current tests it is much slower than the heuristic portfolio and did not
improve `N=50` or `N=100` generated cases, because freezing most users prevents
the global RB-compression moves that the full CP-SAT sweep can prove on small
instances. Treat it as experimental evidence, not the recommended solver.

The literal product objective is degenerate when all users are unscheduled and
no RB is used, because the product becomes zero. The CP-SAT sweep therefore
selects solutions lexicographically by unscheduled-user count first, then by the
task objective.
