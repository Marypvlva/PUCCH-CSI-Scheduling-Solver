import csv
import time
import numpy as np


# ============================================================
# CONFIG: все настройки тут, в запуске ничего прописывать не надо
# ============================================================

N = 1000
BATCH_SEEDS = [1, 2, 3]
MODE = "batch"
SIGMA = 1.0
BATCH_N_VALUES = [1000, 2000]
# Настройки для одного запуска, если MODE = "single"
SINGLE_L = 7
SINGLE_OFF = 0.5
SINGLE_SEED = 3
VERBOSE_SINGLE = True

# Настройки для batch-запуска, если MODE = "batch"
BATCH_L_VALUES = list(range(2, 11))
BATCH_OFF_VALUES = [0.25, 0.5]


# Основные настройки алгоритма
MAX_PASSES = 30
TARGET_GAIN = 5.5

# Быстрый swap для обычных случаев
QUICK_REMOVE_LIMIT = 300
QUICK_ADD_LIMIT = 300

# Smart candidate pool: gradient + power + spectral alignment
USE_SMART_CANDIDATES = True
SMART_POWER_WEIGHT = 0.15

# Полный rescue swap для трудных случаев
RESCUE_REMOVE_LIMIT = None
RESCUE_ADD_LIMIT = None

# Spectral rescue: слабое/сильное собственное направление
USE_SPECTRAL_RESCUE = True
SPECTRAL_STRENGTHS = [20, 30]
SPECTRAL_POWER_WEIGHT = 0.15
SPECTRAL_STOP_GAIN = 100.0

# Annealing rescue: лучший найденный баланс качество/время
USE_ANNEAL_RESCUE = True
ANNEAL_ITERATIONS = [4000, 8000]
ANNEAL_START_TEMPS = [0.015, 0.02, 0.035]
ANNEAL_END_TEMP = 0.0005
ANNEAL_REMOVE_LIMIT = 300
ANNEAL_ADD_LIMIT = 300
ANNEAL_REBUILD_EVERY = 50
ANNEAL_STOP_GAIN = 750.0

# Random rescue fallback. Порядок важен: этот порядок воспроизводил удачный perturb30.
USE_RANDOM_RESCUE = True
RANDOM_STRENGTHS = [3, 5, 10, 20, 30, 50]
RANDOM_RESTARTS = 8
RANDOM_STOP_GAIN = 550.0

# Экспериментальные фазы, которые тестировали и выключили: не окупились.
USE_MULTISWAP_RESCUE = False
USE_GRADIENT_RESCUE = False
USE_QR_DIVERSITY_RESCUE = False
USE_SPECTRAL_DEFICIT_RESCUE = False
USE_ELITE_ANNEAL_REFINEMENT = False
USE_MEMETIC_REFINEMENT = False

# Файлы результатов
SAVE_BATCH_CSV = True
BATCH_CSV_NAME = "antenna_batch_results.csv"

SAVE_SINGLE_MASK = True
SAVE_BATCH_MASKS = False


# ============================================================
# Data generation
# ============================================================

def generate_V(N: int, L: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    V = rng.normal(size=(N, L)) + 1j * rng.normal(size=(N, L))

    column_norms = np.linalg.norm(V, axis=0)
    V = V / column_norms

    antenna_max = np.max(np.linalg.norm(V, axis=1))
    V = V / antenna_max

    return V


# ============================================================
# Objective helpers
# ============================================================

def row_outer_mats(V):
    return V[:, :, None].conj() * V[:, None, :]


def build_S(V, active):
    return V[active].conj().T @ V[active]


def row_powers(V):
    return np.sum(np.abs(V) ** 2, axis=1)


def logdet_score_from_S(S, max_s, sigma=1.0):
    z = 1.0 / np.sqrt(max_s)
    G = z * S
    L = S.shape[0]

    M = G @ G.conj().T + sigma * np.eye(L)
    sign, logdet = np.linalg.slogdet(M)

    if sign <= 0:
        return -1e100

    return float(np.real(logdet))


def logdet_score(V, active, sigma=1.0):
    s = row_powers(V)
    S = build_S(V, active)
    return logdet_score_from_S(S, np.max(s[active]), sigma)


def raw_det_score(V, active, sigma=1.0):
    """
    Official GENERAL objective:
        det(V_eff V_eff* + sigma I)

    Higher is better.
    """
    s = row_powers(V)
    S = build_S(V, active)

    z = 1.0 / np.sqrt(np.max(s[active]))
    G = z * S
    L = G.shape[0]

    M = G @ G.conj().T + sigma * np.eye(L)
    return float(np.real(np.linalg.det(M)))


def row_quadratic_scores(V, M):
    """
    score[n] = v_n M v_n^*
    For Hermitian M this is real.
    """
    return np.real(np.einsum("ni,ij,nj->n", V, M, V.conj()))


def compute_logdet_gradient(S, max_s, sigma=1.0):
    """
    Gradient proxy for log det(sigma I + z^2 S S*) with respect to S,
    treating z as locally fixed.

    Used only for candidate ranking, not as final scoring.
    """
    z2 = 1.0 / max_s
    L = S.shape[0]

    M = z2 * (S @ S.conj().T) + sigma * np.eye(L)
    invM = np.linalg.inv(M)

    grad = 2.0 * z2 * (S @ invM)
    grad = 0.5 * (grad + grad.conj().T)

    return grad


# ============================================================
# Candidate ordering
# ============================================================

def interleave_limited(order_lists, limit):
    if limit is None:
        raise ValueError("limit must not be None in interleave_limited")

    result = []
    seen = set()
    max_len = max(len(x) for x in order_lists) if order_lists else 0

    for pos in range(max_len):
        for arr in order_lists:
            if pos >= len(arr):
                continue

            value = int(arr[pos])
            if value in seen:
                continue

            seen.add(value)
            result.append(value)

            if len(result) >= limit:
                return np.array(result, dtype=int)

    return np.array(result, dtype=int)


def build_candidate_orders(V, active, S, s, remove_limit, add_limit, sigma=1.0):
    """
    Candidate pool for local search.

    Power-only mode:
        remove low-power active, add high-power inactive.

    Smart mode:
        combine gradient ranking, row power, and weak/strong spectral alignment.
    """
    act_idx = np.flatnonzero(active)
    off_idx = np.flatnonzero(~active)

    if not USE_SMART_CANDIDATES:
        rem_order = act_idx[np.argsort(s[act_idx])]
        add_order = off_idx[np.argsort(-s[off_idx])]

        if remove_limit is not None:
            rem_order = rem_order[: min(remove_limit, len(rem_order))]
        if add_limit is not None:
            add_order = add_order[: min(add_limit, len(add_order))]

        return rem_order, add_order

    # Full mode: avoid filtering. Exact local search is expensive but reliable.
    if remove_limit is None and add_limit is None:
        rem_order = act_idx[np.argsort(s[act_idx])]
        add_order = off_idx[np.argsort(-s[off_idx])]
        return rem_order, add_order

    max_s = np.max(s[active])
    grad = compute_logdet_gradient(S, max_s, sigma)
    grad_scores = row_quadratic_scores(V, grad)

    eigvals, eigvecs = np.linalg.eigh(S)
    weak_vec = eigvecs[:, 0]
    strong_vec = eigvecs[:, -1]

    weak_align = np.abs(V @ weak_vec) ** 2
    strong_align = np.abs(V @ strong_vec) ** 2

    if remove_limit is None:
        rem_order = act_idx[np.argsort(s[act_idx])]
    else:
        spectral_remove_score = (
            strong_align[act_idx]
            - weak_align[act_idx]
            - SMART_POWER_WEIGHT * s[act_idx]
        )

        order_low_grad = act_idx[np.argsort(grad_scores[act_idx])]
        order_low_power = act_idx[np.argsort(s[act_idx])]
        order_spectral = act_idx[np.argsort(-spectral_remove_score)]
        order_high_power = act_idx[np.argsort(-s[act_idx])]

        rem_order = interleave_limited(
            [
                order_low_grad,
                order_low_power,
                order_spectral,
                order_high_power,
            ],
            min(remove_limit, len(act_idx)),
        )

    if add_limit is None:
        add_order = off_idx[np.argsort(-s[off_idx])]
    else:
        combo_add_score = (
            grad_scores[off_idx]
            + weak_align[off_idx]
            + SMART_POWER_WEIGHT * s[off_idx]
        )

        order_high_grad = off_idx[np.argsort(-grad_scores[off_idx])]
        order_high_power = off_idx[np.argsort(-s[off_idx])]
        order_weak = off_idx[np.argsort(-weak_align[off_idx])]
        order_combo = off_idx[np.argsort(-combo_add_score)]

        add_order = interleave_limited(
            [
                order_high_grad,
                order_high_power,
                order_weak,
                order_combo,
            ],
            min(add_limit, len(off_idx)),
        )

    return rem_order, add_order


# ============================================================
# Baselines
# ============================================================

def h1_weakest_deletion(V, n_active):
    """
    H1 baseline:
    switch off weakest antennas.
    """
    s = row_powers(V)
    N = V.shape[0]
    n_off = N - n_active

    off = np.argsort(s)[:n_off]

    active = np.ones(N, dtype=bool)
    active[off] = False

    return active


def h2_interference_deletion(V, n_active):
    """
    H2 baseline:
    greedily remove antenna that minimizes off-diagonal interference.

    Vectorized version:
        ||offdiag(S - A_i)||_F^2
      = ||O||_F^2 - 2 Re <O, offdiag(A_i)> + ||offdiag(A_i)||_F^2
    """
    N, L = V.shape
    A = row_outer_mats(V)
    s = row_powers(V)

    # ||offdiag(A_i)||_F^2 = (sum |v|^2)^2 - sum |v|^4
    offA_norm = s ** 2 - np.sum(np.abs(V) ** 4, axis=1)

    active = np.ones(N, dtype=bool)
    S = V.conj().T @ V

    n_remove = N - n_active

    for _ in range(n_remove):
        idx = np.flatnonzero(active)

        O = S.copy()
        np.fill_diagonal(O, 0.0)

        const = float(np.real(np.sum(np.abs(O) ** 2)))

        # inner_i = <O, offdiag(A_i)>
        # A_i[p,q] = conj(v_i[p]) * v_i[q]
        inner = np.einsum(
            "pq,np,nq->n",
            np.conj(O),
            np.conj(V[idx]),
            V[idx],
            optimize=True,
        )

        vals = const - 2.0 * np.real(inner) + offA_norm[idx]

        best_pos = int(np.argmin(vals))
        best_i = int(idx[best_pos])

        active[best_i] = False
        S -= A[best_i]

    return active


# ============================================================
# Core methods: greedy deletion + local search
# ============================================================

def greedy_rank_one_deletion(V, n_active, sigma=1.0, verbose=False):
    """
    Greedy deletion optimized for GENERAL logdet objective.
    Uses rank-one updates:
        S_new = S - v_i^* v_i
    """
    N, L = V.shape
    A = row_outer_mats(V)
    s = row_powers(V)

    active = np.ones(N, dtype=bool)
    S = V.conj().T @ V

    while np.sum(active) > n_active:
        idx = np.flatnonzero(active)

        active_s = s[idx]
        order = np.argsort(active_s)

        max1_i = idx[order[-1]]
        max1 = active_s[order[-1]]
        max2 = active_s[order[-2]] if len(order) >= 2 else max1

        best_i = None
        best_score = -1e100

        for i in idx:
            S_new = S - A[i]
            max_s_new = max2 if i == max1_i else max1

            sc = logdet_score_from_S(S_new, max_s_new, sigma)

            if sc > best_score:
                best_score = sc
                best_i = i

        active[best_i] = False
        S -= A[best_i]

        if verbose and (np.sum(active) % 50 == 0 or np.sum(active) == n_active):
            print(f"[greedy] active={np.sum(active)} logdet={best_score:.10f}")

    return active


def vectorized_swap_local_search(
    V,
    active,
    sigma=1.0,
    max_passes=30,
    remove_limit=None,
    add_limit=None,
    verbose=False,
):
    """
    1-swap local search:
    replace one active antenna by one inactive antenna if objective improves.

    Full search is vectorized over candidate additions.
    """
    N, L = V.shape

    A = row_outer_mats(V)
    s = row_powers(V)

    active = active.copy()
    S = build_S(V, active)

    base = logdet_score_from_S(S, np.max(s[active]), sigma)
    eye = np.eye(L)

    for p in range(max_passes):
        rem_order, add_order = build_candidate_orders(
            V,
            active,
            S,
            s,
            remove_limit=remove_limit,
            add_limit=add_limit,
            sigma=sigma,
        )

        act_idx = np.flatnonzero(active)
        active_s = s[act_idx]
        sorted_active = act_idx[np.argsort(active_s)]

        max1_i = sorted_active[-1]
        max1 = s[max1_i]
        max2 = s[sorted_active[-2]] if len(sorted_active) >= 2 else max1

        best_score = base
        best_i = None
        best_j = None
        best_S = None

        A_add = A[add_order]
        s_add = s[add_order]

        for i in rem_order:
            S_minus = S - A[i]
            max_without_i = max2 if i == max1_i else max1

            max_s_new = np.maximum(max_without_i, s_add)
            z = 1.0 / np.sqrt(max_s_new)

            S_batch = S_minus[None, :, :] + A_add
            G_batch = S_batch * z[:, None, None]

            M_batch = G_batch @ np.swapaxes(G_batch.conj(), 1, 2)
            M_batch += sigma * eye[None, :, :]

            signs, logdets = np.linalg.slogdet(M_batch)
            scores = np.where(signs > 0, np.real(logdets), -1e100)

            pos = int(np.argmax(scores))
            sc = float(scores[pos])

            if sc > best_score + 1e-12:
                best_score = sc
                best_i = int(i)
                best_j = int(add_order[pos])
                best_S = S_minus + A[best_j]

        if best_i is None:
            if verbose:
                print(f"[swap] pass={p + 1}: no improvement, stop; logdet={base:.10f}")
            break

        active[best_i] = False
        active[best_j] = True
        S = best_S
        base = best_score

        if verbose:
            print(
                f"[swap] pass={p + 1}: logdet={base:.10f}, "
                f"out={best_i}, in={best_j}"
            )

    return active


# ============================================================
# Rescue methods
# ============================================================

def spectral_balance_perturb(V, active, strength=20, power_weight=0.15):
    """
    Directed perturbation for determinant maximization.

    Idea:
    - determinant likes balanced eigenvalues;
    - find the weakest eigen-direction of S;
    - add inactive antennas aligned with that weak direction;
    - remove active antennas aligned with the strongest direction.
    """
    active = active.copy()

    s = row_powers(V)
    S = build_S(V, active)

    eigvals, eigvecs = np.linalg.eigh(S)

    weak_vec = eigvecs[:, 0]
    strong_vec = eigvecs[:, -1]

    act_idx = np.flatnonzero(active)
    off_idx = np.flatnonzero(~active)

    strength = min(strength, len(act_idx), len(off_idx))

    add_alignment = np.abs(V[off_idx] @ weak_vec) ** 2
    add_scores = add_alignment + power_weight * s[off_idx]

    remove_strong_alignment = np.abs(V[act_idx] @ strong_vec) ** 2
    remove_weak_alignment = np.abs(V[act_idx] @ weak_vec) ** 2

    remove_scores = (
        remove_strong_alignment
        - remove_weak_alignment
        - power_weight * s[act_idx]
    )

    remove_idx = act_idx[np.argsort(-remove_scores)[:strength]]
    add_idx = off_idx[np.argsort(-add_scores)[:strength]]

    active[remove_idx] = False
    active[add_idx] = True

    return active


def annealing_kick(
    V,
    active,
    seed,
    sigma=1.0,
    iterations=4000,
    start_temp=0.02,
    end_temp=0.0005,
    remove_limit=300,
    add_limit=300,
    rebuild_every=50,
):
    """
    Simulated annealing / threshold-accepting kick.

    Starts from an existing active set, makes stochastic swaps inside a smart
    candidate pool, accepts improving swaps and sometimes accepts worsening swaps.
    The best visited active set is then polished by full local search.
    """
    rng = np.random.default_rng(seed)

    A = row_outer_mats(V)
    s = row_powers(V)

    active = active.copy()
    S = build_S(V, active)

    current_score = logdet_score_from_S(S, np.max(s[active]), sigma)

    best_active = active.copy()
    best_score = current_score

    rem_order = None
    add_order = None

    for it in range(iterations):
        if rem_order is None or add_order is None or it % rebuild_every == 0:
            rem_order, add_order = build_candidate_orders(
                V,
                active,
                S,
                s,
                remove_limit=remove_limit,
                add_limit=add_limit,
                sigma=sigma,
            )

        rem_candidates = rem_order[active[rem_order]]
        add_candidates = add_order[~active[add_order]]

        if len(rem_candidates) == 0 or len(add_candidates) == 0:
            rem_order = None
            add_order = None
            continue

        i = int(rng.choice(rem_candidates))
        j = int(rng.choice(add_candidates))

        S_new = S - A[i] + A[j]

        active[i] = False
        active[j] = True

        max_s_new = np.max(s[active])
        new_score = logdet_score_from_S(S_new, max_s_new, sigma)

        delta = new_score - current_score
        progress = it / max(iterations - 1, 1)
        temp = start_temp * ((end_temp / start_temp) ** progress)

        accept = delta >= 0.0

        if not accept and temp > 0.0:
            accept = rng.random() < np.exp(np.clip(delta / temp, -60.0, 0.0))

        if accept:
            S = S_new
            current_score = new_score

            if new_score > best_score:
                best_score = new_score
                best_active = active.copy()
        else:
            active[i] = True
            active[j] = False

    return best_active


def random_perturb(active, rng, strength):
    """
    Randomly swap `strength` active antennas with inactive antennas.
    Used only when spectral/annealing rescue is not enough.
    """
    active = active.copy()

    act_idx = np.flatnonzero(active)
    off_idx = np.flatnonzero(~active)

    strength = min(strength, len(act_idx), len(off_idx))

    rem = rng.choice(act_idx, size=strength, replace=False)
    add = rng.choice(off_idx, size=strength, replace=False)

    active[rem] = False
    active[add] = True

    return active


# ============================================================
# Solver
# ============================================================

def improve_from_start(
    V,
    start_active,
    sigma=1.0,
    max_passes=30,
    remove_limit=None,
    add_limit=None,
):
    return vectorized_swap_local_search(
        V,
        start_active,
        sigma=sigma,
        max_passes=max_passes,
        remove_limit=remove_limit,
        add_limit=add_limit,
        verbose=False,
    )


def package_result(
    active,
    method,
    h1,
    h2,
    h1_score,
    h2_score,
    ours_score,
    gain_percent,
    used_rescue,
    used_spectral,
    used_anneal,
    used_random,
):
    return {
        "active": active,
        "method": method,
        "h1": h1,
        "h2": h2,
        "h1_score": h1_score,
        "h2_score": h2_score,
        "ours_score": ours_score,
        "gain_percent": gain_percent,
        "used_rescue": used_rescue,
        "used_spectral": used_spectral,
        "used_anneal": used_anneal,
        "used_random": used_random,
    }


def current_gain_percent(best_active, V, best_baseline, sigma):
    raw = raw_det_score(V, best_active, sigma)
    gain = 100.0 * (raw - best_baseline) / best_baseline
    return raw, gain


def solve_general(
    V,
    n_active,
    seed=1,
    sigma=1.0,
    max_passes=30,
    target_gain=5.5,
    verbose=False,
):
    rng = np.random.default_rng(seed + 1000003)

    used_spectral = False
    used_anneal = False
    used_random = False

    h1 = h1_weakest_deletion(V, n_active)
    h2 = h2_interference_deletion(V, n_active)
    greedy = greedy_rank_one_deletion(V, n_active, sigma=sigma, verbose=False)

    h1_u = raw_det_score(V, h1, sigma)
    h2_u = raw_det_score(V, h2, sigma)
    best_baseline = max(h1_u, h2_u)

    best_name = None
    best_active = None
    best_logdet = -1e100

    # --------------------------------------------------------
    # Phase 1: fast starts
    # --------------------------------------------------------

    quick_starts = [
        ("H1+swap", h1),
        ("H2+swap", h2),
        ("greedy+swap", greedy),
    ]

    if verbose:
        print("[phase 1] quick starts")

    for name, start_active in quick_starts:
        t = time.time()

        improved = improve_from_start(
            V,
            start_active,
            sigma=sigma,
            max_passes=max_passes,
            remove_limit=QUICK_REMOVE_LIMIT,
            add_limit=QUICK_ADD_LIMIT,
        )

        sc = logdet_score(V, improved, sigma)

        if verbose:
            raw = raw_det_score(V, improved, sigma)
            print(
                f"[candidate] {name:24s} "
                f"logdet={sc:.10f} det={raw:.10e} "
                f"time={time.time() - t:.2f}s"
            )

        if sc > best_logdet:
            best_logdet = sc
            best_active = improved
            best_name = name

    best_raw, current_gain = current_gain_percent(best_active, V, best_baseline, sigma)

    if verbose:
        print(f"[quick gain] {current_gain:.4f}%")

    if current_gain >= target_gain:
        return package_result(
            active=best_active,
            method=best_name,
            h1=h1,
            h2=h2,
            h1_score=h1_u,
            h2_score=h2_u,
            ours_score=best_raw,
            gain_percent=current_gain,
            used_rescue=False,
            used_spectral=False,
            used_anneal=False,
            used_random=False,
        )

    if verbose:
        print("[phase 2] rescue starts")

    def try_candidate(name, start_active, remove_limit, add_limit, flag_name):
        nonlocal best_logdet, best_active, best_name
        nonlocal used_spectral, used_anneal, used_random

        t = time.time()

        improved = improve_from_start(
            V,
            start_active,
            sigma=sigma,
            max_passes=max_passes,
            remove_limit=remove_limit,
            add_limit=add_limit,
        )

        sc = logdet_score(V, improved, sigma)

        if verbose:
            raw = raw_det_score(V, improved, sigma)
            print(
                f"[candidate] {name:34s} "
                f"logdet={sc:.10f} det={raw:.10e} "
                f"time={time.time() - t:.2f}s"
            )

        if sc > best_logdet:
            best_logdet = sc
            best_active = improved
            best_name = name

        if flag_name == "spectral":
            used_spectral = True
        elif flag_name == "anneal":
            used_anneal = True
        elif flag_name == "random":
            used_random = True

    def maybe_return_if_gain_at_least(threshold):
        best_raw_now, gain_now = current_gain_percent(best_active, V, best_baseline, sigma)

        if gain_now >= threshold:
            return package_result(
                active=best_active,
                method=best_name,
                h1=h1,
                h2=h2,
                h1_score=h1_u,
                h2_score=h2_u,
                ours_score=best_raw_now,
                gain_percent=gain_now,
                used_rescue=True,
                used_spectral=used_spectral,
                used_anneal=used_anneal,
                used_random=used_random,
            )

        return None

    # --------------------------------------------------------
    # Phase 2A: spectral weak/strong rescue
    # --------------------------------------------------------

    if USE_SPECTRAL_RESCUE:
        base_masks = [
            ("best", best_active),
            ("H1", h1),
            ("H2", h2),
            ("greedy", greedy),
        ]

        for base_name, base_mask in base_masks:
            for strength in SPECTRAL_STRENGTHS:
                start = spectral_balance_perturb(
                    V,
                    base_mask,
                    strength=strength,
                    power_weight=SPECTRAL_POWER_WEIGHT,
                )

                try_candidate(
                    f"spectral{strength}-{base_name}+swap",
                    start,
                    RESCUE_REMOVE_LIMIT,
                    RESCUE_ADD_LIMIT,
                    "spectral",
                )

                out = maybe_return_if_gain_at_least(SPECTRAL_STOP_GAIN)
                if out is not None:
                    return out

    # --------------------------------------------------------
    # Phase 2B: annealing rescue inside smart candidate pool
    # --------------------------------------------------------

    if USE_ANNEAL_RESCUE:
        base_masks = [
            ("best", best_active),
            ("H1", h1),
            ("H2", h2),
            ("greedy", greedy),
        ]

        anneal_counter = 0

        for base_name, base_mask in base_masks:
            for iterations in ANNEAL_ITERATIONS:
                for temp in ANNEAL_START_TEMPS:
                    start = annealing_kick(
                        V,
                        base_mask,
                        seed=seed * 100000 + anneal_counter,
                        sigma=sigma,
                        iterations=iterations,
                        start_temp=temp,
                        end_temp=ANNEAL_END_TEMP,
                        remove_limit=ANNEAL_REMOVE_LIMIT,
                        add_limit=ANNEAL_ADD_LIMIT,
                        rebuild_every=ANNEAL_REBUILD_EVERY,
                    )

                    anneal_counter += 1

                    try_candidate(
                        f"anneal{iterations}-T{temp}-{base_name}+swap",
                        start,
                        RESCUE_REMOVE_LIMIT,
                        RESCUE_ADD_LIMIT,
                        "anneal",
                    )

                    out = maybe_return_if_gain_at_least(ANNEAL_STOP_GAIN)
                    if out is not None:
                        return out

    # --------------------------------------------------------
    # Phase 2C: random fallback in stable order
    # --------------------------------------------------------

    if USE_RANDOM_RESCUE:
        base_masks = [h1, h2, greedy, best_active]
        count = 0

        for base in base_masks:
            for strength in RANDOM_STRENGTHS:
                if count >= RANDOM_RESTARTS:
                    break

                start = random_perturb(base, rng, strength)

                try_candidate(
                    f"perturb{strength}+swap",
                    start,
                    RESCUE_REMOVE_LIMIT,
                    RESCUE_ADD_LIMIT,
                    "random",
                )

                count += 1

                out = maybe_return_if_gain_at_least(RANDOM_STOP_GAIN)
                if out is not None:
                    return out

            if count >= RANDOM_RESTARTS:
                break

    best_raw, current_gain = current_gain_percent(best_active, V, best_baseline, sigma)

    return package_result(
        active=best_active,
        method=best_name,
        h1=h1,
        h2=h2,
        h1_score=h1_u,
        h2_score=h2_u,
        ours_score=best_raw,
        gain_percent=current_gain,
        used_rescue=True,
        used_spectral=used_spectral,
        used_anneal=used_anneal,
        used_random=used_random,
    )


# ============================================================
# Run modes
# ============================================================

def run_single_case(N, L, off, seed, verbose=True):
    t0 = time.time()

    V = generate_V(N, L, seed)
    n_active = int(round(N * (1.0 - off)))

    print(f"N={N}, L={L}, off={off}, active={n_active}, seed={seed}")
    print("-" * 110)

    result = solve_general(
        V,
        n_active,
        seed=seed,
        sigma=SIGMA,
        max_passes=MAX_PASSES,
        target_gain=TARGET_GAIN,
        verbose=verbose,
    )

    print("GENERAL objective, det(...), higher is better")
    print(f"H1       = {result['h1_score']:.10e}")
    print(f"H2       = {result['h2_score']:.10e}")
    print(f"Ours     = {result['ours_score']:.10e}")
    print(f"best start/method = {result['method']}")
    print(f"used rescue   = {result['used_rescue']}")
    print(f"used spectral = {result['used_spectral']}")
    print(f"used anneal   = {result['used_anneal']}")
    print(f"used random   = {result['used_random']}")
    print(f"gain vs best baseline = {result['gain_percent']:.4f}%")
    print(f"total time = {time.time() - t0:.2f}s")

    if SAVE_SINGLE_MASK:
        out = f"active_mask_N{N}_L{L}_off{off}_general_seed{seed}.npy"
        np.save(out, result["active"].astype(np.int8))
        print(f"saved active mask = {out}")

    return result


def run_batch():
    rows = []

    print("Running batch for GENERAL objective...")
    print(
        f"N_values={BATCH_N_VALUES}, target_gain={TARGET_GAIN}, max_passes={MAX_PASSES}, "
        f"quick_limits=({QUICK_REMOVE_LIMIT}, {QUICK_ADD_LIMIT}), "
        f"smart_candidates={USE_SMART_CANDIDATES}, "
        f"spectral={USE_SPECTRAL_RESCUE}, anneal={USE_ANNEAL_RESCUE}, "
        f"random={USE_RANDOM_RESCUE}"
    )
    print("-" * 165)

    total_t0 = time.time()

    for current_N in BATCH_N_VALUES:
        for off in BATCH_OFF_VALUES:
            for L in BATCH_L_VALUES:
                for seed in BATCH_SEEDS:
                    t0 = time.time()

                    V = generate_V(current_N, L, seed)
                    n_active = int(round(current_N * (1.0 - off)))

                    result = solve_general(
                        V,
                        n_active,
                        seed=seed,
                        sigma=SIGMA,
                        max_passes=MAX_PASSES,
                        target_gain=TARGET_GAIN,
                        verbose=False,
                    )

                    elapsed = time.time() - t0

                    row = {
                        "N": current_N,
                        "L": L,
                        "off": off,
                        "active": n_active,
                        "seed": seed,
                        "H1": result["h1_score"],
                        "H2": result["h2_score"],
                        "baseline": max(result["h1_score"], result["h2_score"]),
                        "ours": result["ours_score"],
                        "gain_percent": result["gain_percent"],
                        "best_method": result["method"],
                        "used_rescue": result["used_rescue"],
                        "used_spectral": result["used_spectral"],
                        "used_anneal": result["used_anneal"],
                        "used_random": result["used_random"],
                        "time_sec": elapsed,
                    }

                    rows.append(row)

                    print(
                        f"N={current_N} L={L:2d} off={off:.2f} seed={seed} "
                        f"gain={result['gain_percent']:9.3f}% "
                        f"method={result['method'][:38]:38s} "
                        f"res={str(result['used_rescue']):5s} "
                        f"spec={str(result['used_spectral']):5s} "
                        f"ann={str(result['used_anneal']):5s} "
                        f"rand={str(result['used_random']):5s} "
                        f"time={elapsed:7.2f}s"
                    )

    gains = np.array([r["gain_percent"] for r in rows])
    times = np.array([r["time_sec"] for r in rows])

    print("-" * 165)
    print(f"cases = {len(rows)}")
    print(f"min gain     = {np.min(gains):.3f}%")
    print(f"mean gain    = {np.mean(gains):.3f}%")
    print(f"median gain  = {np.median(gains):.3f}%")
    print(f"max gain     = {np.max(gains):.3f}%")
    print(f"failed <5%   = {np.sum(gains < 5.0)}")
    print()
    print(f"min time     = {np.min(times):.2f}s")
    print(f"mean time    = {np.mean(times):.2f}s")
    print(f"median time  = {np.median(times):.2f}s")
    print(f"max time     = {np.max(times):.2f}s")
    print(f"total time   = {time.time() - total_t0:.2f}s")

    print()
    for current_N in BATCH_N_VALUES:
        subset = [r for r in rows if r["N"] == current_N]
        subset_gains = np.array([r["gain_percent"] for r in subset])
        subset_times = np.array([r["time_sec"] for r in subset])

        print(
            f"N={current_N}: "
            f"mean gain={np.mean(subset_gains):.3f}%, "
            f"min gain={np.min(subset_gains):.3f}%, "
            f"mean time={np.mean(subset_times):.2f}s, "
            f"max time={np.max(subset_times):.2f}s"
        )

    if SAVE_BATCH_CSV:
        with open(BATCH_CSV_NAME, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"saved csv = {BATCH_CSV_NAME}")

def main():
    if MODE == "single":
        run_single_case(
            N=N,
            L=SINGLE_L,
            off=SINGLE_OFF,
            seed=SINGLE_SEED,
            verbose=VERBOSE_SINGLE,
        )

    elif MODE == "batch":
        run_batch()

    else:
        raise ValueError("MODE must be either 'single' or 'batch'")


if __name__ == "__main__":
    main()
