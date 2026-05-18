# PUCCH CSI Scheduling Solver

Single-file Python implementation of a hybrid heuristic for PUCCH CSI scheduling.

The solver combines:
- adaptive greedy construction,
- randomized multistart search,
- local one-user reassignment,
- destroy-and-repair optimization,
- portfolio selection over several parameter settings.

## Run

```bash
python pucch_csi.py