import argparse
from collections import defaultdict
import json
import math
import random
import statistics
import time
from functools import lru_cache


L = 320
R = 58
UNSCHEDULED_PERIOD = 640
DEFAULT_TIME_OPTION_CAP = 64


def is_uplink_slot(slot: int, dl: int, ul: int) -> bool:
    return slot % (dl + ul) >= dl


def periodic_slots(period: int, offset: int) -> list[int]:
    return list(range(offset, L, period))


@lru_cache(maxsize=None)
def periodic_mask(period: int, offset: int) -> int:
    mask = 0
    for slot in periodic_slots(period, offset):
        mask |= 1 << slot
    return mask


@lru_cache(maxsize=None)
def uplink_mask(dl: int, ul: int) -> int:
    mask = 0
    for slot in range(L):
        if is_uplink_slot(slot, dl, ul):
            mask |= 1 << slot
    return mask


@lru_cache(maxsize=None)
def distance_between_resources(csi_period, csi_offset, srs_period, srs_offset) -> int:
    if srs_period >= csi_period:
        return abs(csi_offset - (srs_offset % csi_period))

    ratio = csi_period // srs_period
    return min(
        abs(csi_offset - ((srs_offset + i * srs_period) % csi_period))
        for i in range(ratio)
    )


@lru_cache(maxsize=None)
def distance_to_multiple_srs(csi_period, csi_offset, srs_period, srs_offsets) -> int:
    return min(
        distance_between_resources(csi_period, csi_offset, srs_period, srs_offset)
        for srs_offset in srs_offsets
    )


@lru_cache(maxsize=None)
def cached_time_options(csi_periods, srs_period, srs_offsets, dl, ul):
    options = []
    for period in csi_periods:
        for offset in range(period):
            mask = periodic_mask(period, offset)
            if mask & ~uplink_mask(dl, ul):
                continue
            distance = distance_to_multiple_srs(period, offset, srs_period, srs_offsets)
            options.append((period, offset, mask, distance, period + distance))
    return tuple(sorted(options, key=lambda x: (x[4], x[0], x[1])))


def time_options_for_user(user, dl, ul):
    return cached_time_options(
        tuple(user["csi_periods"]),
        user["srs_period"],
        tuple(user["srs_offsets"]),
        dl,
        ul,
    )


def can_place_csi(occupied, rb, period, offset, dl, ul) -> bool:
    mask = periodic_mask(period, offset)
    if mask & ~uplink_mask(dl, ul):
        return False
    return (occupied[rb] & mask) == 0


def place_csi(occupied, rb, period, offset) -> None:
    occupied[rb] |= periodic_mask(period, offset)


def remove_csi(occupied, rb, period, offset) -> None:
    if period == UNSCHEDULED_PERIOD:
        return
    occupied[rb] &= ~periodic_mask(period, offset)


def empty_occupied():
    return [0 for _ in range(R)]


def popcount(value):
    return bin(value).count("1")


def clone_occupied(occupied):
    return list(occupied) if occupied is not None else empty_occupied()


def build_occupied_from_answers(answers, initial_occupied=None):
    occupied = clone_occupied(initial_occupied)
    for a in answers:
        if a["period"] != UNSCHEDULED_PERIOD:
            place_csi(occupied, a["rb"], a["period"], a["offset"])
    return occupied


def used_rbs_from_occupied(occupied):
    return {rb for rb, mask in enumerate(occupied) if mask}


def candidate_rbs_from_occupied(occupied, rb_limit=None):
    used = sorted(rb for rb, mask in enumerate(occupied) if mask)
    if rb_limit is not None and len(used) >= rb_limit:
        return used

    for rb, mask in enumerate(occupied):
        if not mask:
            return used + [rb]
    return used


def make_answer(user_id, rb, period, offset, distance):
    return {
        "user_id": user_id,
        "rb": rb,
        "period": period,
        "offset": offset,
        "distance": distance,
        "quality": distance + period,
    }


def solution_stats(answers, initial_occupied=None):
    scheduled = [a for a in answers if a["period"] != UNSCHEDULED_PERIOD]
    used = {a["rb"] for a in scheduled}
    if initial_occupied is not None:
        used.update(rb for rb, mask in enumerate(initial_occupied) if mask)
    rb_used = len(used)
    quality_sum = sum(a["quality"] for a in answers)
    unscheduled = len(answers) - len(scheduled)
    occupied = build_occupied_from_answers(answers, initial_occupied)
    occupied_pairs = sum(popcount(mask) for mask in occupied)
    return {
        "rb_used": rb_used,
        "quality_sum": quality_sum,
        "objective": rb_used * quality_sum,
        "unscheduled": unscheduled,
        "occupied_pairs": occupied_pairs,
    }


def compact_solution_stats(answers, initial_occupied=None):
    scheduled = [a for a in answers if a["period"] != UNSCHEDULED_PERIOD]
    used = {a["rb"] for a in scheduled}
    occupied_pairs = sum(len(periodic_slots(a["period"], a["offset"])) for a in scheduled)
    if initial_occupied is not None:
        for rb, mask in enumerate(initial_occupied):
            if mask:
                used.add(rb)
                occupied_pairs += popcount(mask)
    rb_used = len(used)
    quality_sum = sum(a["quality"] for a in answers)
    unscheduled = len(answers) - len(scheduled)
    return {
        "rb_used": rb_used,
        "quality_sum": quality_sum,
        "objective": rb_used * quality_sum,
        "unscheduled": unscheduled,
        "occupied_pairs": occupied_pairs,
    }


def objective_value(answers, initial_occupied=None) -> int:
    return solution_stats(answers, initial_occupied)["objective"]


def objective_key(answers, objective="product", initial_occupied=None):
    stats = solution_stats(answers, initial_occupied)
    if objective == "lexicographic":
        return stats["unscheduled"], stats["rb_used"], stats["quality_sum"]
    return stats["unscheduled"], stats["objective"], stats["rb_used"], stats["quality_sum"]


def objective_key_from_stats(stats, objective="product"):
    if objective == "lexicographic":
        return stats["unscheduled"], stats["rb_used"], stats["quality_sum"]
    return stats["unscheduled"], stats["objective"], stats["rb_used"], stats["quality_sum"]


def objective_scalar(answers, objective="product", initial_occupied=None):
    stats = solution_stats(answers, initial_occupied)
    if objective == "lexicographic":
        return stats["unscheduled"] * 10**12 + stats["rb_used"] * 10**9 + stats["quality_sum"]
    return stats["unscheduled"] * 10**12 + stats["objective"]


def objective_scalar_from_stats(stats, objective="product"):
    if objective == "lexicographic":
        return stats["unscheduled"] * 10**12 + stats["rb_used"] * 10**9 + stats["quality_sum"]
    return stats["unscheduled"] * 10**12 + stats["objective"]


def validate_answers(users, answers, dl, ul, initial_occupied=None, allow_unscheduled=True):
    errors = []
    if len(answers) != len(users):
        errors.append(f"expected {len(users)} answers, got {len(answers)}")

    seen_users = set()
    occupied = set()
    if initial_occupied is not None:
        for rb, mask in enumerate(initial_occupied):
            for slot in range(L):
                if mask & (1 << slot):
                    occupied.add((rb, slot))
    for answer in answers:
        user_id = answer["user_id"]
        if user_id in seen_users:
            errors.append(f"user {user_id} appears more than once")
        seen_users.add(user_id)

        period = answer["period"]
        if period == UNSCHEDULED_PERIOD:
            if not allow_unscheduled:
                errors.append(f"user {user_id} is unscheduled")
            continue

        rb = answer["rb"]
        offset = answer["offset"]
        if not 0 <= rb < R:
            errors.append(f"user {user_id} uses invalid RB {rb}")
            continue
        if period not in users[user_id]["csi_periods"]:
            errors.append(f"user {user_id} uses unsupported CSI period {period}")
        if not 0 <= offset < period:
            errors.append(f"user {user_id} uses invalid offset {offset} for period {period}")

        for slot in periodic_slots(period, offset):
            if not is_uplink_slot(slot, dl, ul):
                errors.append(f"user {user_id} has CSI in non-uplink slot {slot}")
            pair = (rb, slot)
            if pair in occupied:
                errors.append(f"collision on RB {rb}, slot {slot}")
            occupied.add(pair)

    missing = set(range(len(users))) - seen_users
    if missing:
        errors.append(f"missing users: {sorted(missing)}")
    return errors


def candidate_score(
    scoring,
    objective,
    quality,
    distance,
    period,
    rb,
    offset,
    used_rbs,
    current_quality_sum,
    base_penalty,
    quality_weight,
):
    current_rb_used = len(used_rbs)
    opens_new_rb = rb not in used_rbs
    next_rb_used = current_rb_used + int(opens_new_rb)
    next_quality_sum = current_quality_sum + quality

    if scoring == "quality":
        key = (quality_weight * quality,)
    elif scoring == "baseline":
        key = (distance,)
    elif scoring == "adaptive":
        adaptive_penalty = base_penalty * (1 + current_rb_used / 5)
        new_rb_penalty = adaptive_penalty if opens_new_rb else 0
        key = (quality_weight * quality + new_rb_penalty,)
    elif scoring == "marginal":
        if objective == "lexicographic":
            key = (next_rb_used, next_quality_sum)
        else:
            current_objective = current_rb_used * current_quality_sum
            next_objective = next_rb_used * next_quality_sum
            key = (next_objective - current_objective,)
    else:
        raise ValueError(f"unknown scoring mode: {scoring}")

    return key + (distance, period, rb, offset)


def slot_pressure_from_occupied(occupied):
    pressure = [0] * L
    for mask in occupied:
        if not mask:
            continue
        for slot in range(L):
            if mask & (1 << slot):
                pressure[slot] += 1
    return pressure


def best_options_for_one_user(
    occupied,
    csi_periods,
    srs_period,
    srs_offsets,
    dl,
    ul,
    base_penalty=60,
    quality_weight=1.0,
    scoring="adaptive",
    objective="product",
    current_quality_sum=0,
    rb_limit=None,
    limit=2,
    time_option_cap=None,
):
    used_rbs = used_rbs_from_occupied(occupied)
    rb_candidates = candidate_rbs_from_occupied(occupied, rb_limit)
    best = []
    srs_offsets = tuple(srs_offsets)
    repair_scoring = scoring
    pressure = None
    if scoring == "conflict-aware":
        repair_scoring = "marginal"
        pressure = slot_pressure_from_occupied(occupied)

    time_options = cached_time_options(
        tuple(csi_periods),
        srs_period,
        srs_offsets,
        dl,
        ul,
    )
    cap = DEFAULT_TIME_OPTION_CAP if time_option_cap is None else time_option_cap
    if cap is not None:
        time_options = time_options[:cap]

    for period, offset, mask, distance, quality in time_options:
        for rb in rb_candidates:
            if occupied[rb] & mask:
                continue

            score = candidate_score(
                scoring=repair_scoring,
                objective=objective,
                quality=quality,
                distance=distance,
                period=period,
                rb=rb,
                offset=offset,
                used_rbs=used_rbs,
                current_quality_sum=current_quality_sum,
                base_penalty=base_penalty,
                quality_weight=quality_weight,
            )
            if pressure is not None:
                slots = periodic_slots(period, offset)
                pressure_score = sum(pressure[slot] for slot in slots)
                rb_load = popcount(occupied[rb])
                opens_new_rb = rb not in used_rbs
                score = score[:1] + (
                    pressure_score,
                    rb_load,
                    int(opens_new_rb),
                ) + score[1:]
            option = (score, rb, period, offset, distance)

            if len(best) < limit:
                best.append(option)
                best.sort(key=lambda x: x[0])
            elif option[0] < best[-1][0]:
                best[-1] = option
                best.sort(key=lambda x: x[0])

    return best


def find_best_for_one_user(
    occupied,
    csi_periods,
    srs_period,
    srs_offsets,
    dl,
    ul,
    base_penalty=60,
    quality_weight=1.0,
    scoring="adaptive",
    objective="product",
    current_quality_sum=0,
    rb_limit=None,
    time_option_cap=None,
):
    best = best_options_for_one_user(
        occupied=occupied,
        csi_periods=csi_periods,
        srs_period=srs_period,
        srs_offsets=srs_offsets,
        dl=dl,
        ul=ul,
        base_penalty=base_penalty,
        quality_weight=quality_weight,
        scoring=scoring,
        objective=objective,
        current_quality_sum=current_quality_sum,
        rb_limit=rb_limit,
        limit=1,
        time_option_cap=time_option_cap,
    )
    if not best:
        return 0, UNSCHEDULED_PERIOD, 0, UNSCHEDULED_PERIOD
    _, rb, period, offset, distance = best[0]
    return rb, period, offset, distance


def greedy_solve(
    users,
    dl,
    ul,
    base_penalty=60,
    quality_weight=1.0,
    scoring="adaptive",
    objective="product",
    order=None,
    rb_limit=None,
    initial_occupied=None,
):
    occupied = clone_occupied(initial_occupied)
    answers = []
    quality_sum = 0
    user_order = list(range(len(users))) if order is None else order

    for user_id in user_order:
        user = users[user_id]
        rb, period, offset, distance = find_best_for_one_user(
            occupied=occupied,
            csi_periods=user["csi_periods"],
            srs_period=user["srs_period"],
            srs_offsets=user["srs_offsets"],
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
            current_quality_sum=quality_sum,
            rb_limit=rb_limit,
        )

        if period != UNSCHEDULED_PERIOD:
            place_csi(occupied, rb, period, offset)

        answer = make_answer(user_id, rb, period, offset, distance)
        quality_sum += answer["quality"]
        answers.append(answer)

    return answers, objective_value(answers, initial_occupied)


def static_regret_order(
    users,
    dl,
    ul,
    base_penalty=60,
    quality_weight=1.0,
    scoring="marginal",
    objective="product",
):
    occupied = empty_occupied()
    ranked = []
    for user_id, user in enumerate(users):
        options = best_options_for_one_user(
            occupied=occupied,
            csi_periods=user["csi_periods"],
            srs_period=user["srs_period"],
            srs_offsets=user["srs_offsets"],
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
            current_quality_sum=0,
            limit=2,
        )
        if not options:
            regret = float("inf")
            best_score = float("inf")
        elif len(options) == 1:
            regret = float("inf")
            best_score = options[0][0][0]
        else:
            regret = options[1][0][0] - options[0][0][0]
            best_score = options[0][0][0]
        ranked.append((regret, -best_score, user_id))
    ranked.sort(reverse=True)
    return [user_id for _, _, user_id in ranked]


def regret_greedy_solve(
    users,
    dl,
    ul,
    base_penalty=60,
    quality_weight=1.0,
    scoring="marginal",
    objective="product",
    rb_limit=None,
    dynamic=False,
    initial_occupied=None,
):
    if not dynamic:
        order = static_regret_order(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
        )
        return greedy_solve(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
            order=order,
            rb_limit=rb_limit,
            initial_occupied=initial_occupied,
        )

    occupied = clone_occupied(initial_occupied)
    answers = []
    quality_sum = 0
    remaining = set(range(len(users)))

    while remaining:
        selected = None
        selected_option = None
        selected_key = None

        for user_id in remaining:
            user = users[user_id]
            options = best_options_for_one_user(
                occupied=occupied,
                csi_periods=user["csi_periods"],
                srs_period=user["srs_period"],
                srs_offsets=user["srs_offsets"],
                dl=dl,
                ul=ul,
                base_penalty=base_penalty,
                quality_weight=quality_weight,
                scoring=scoring,
                objective=objective,
                current_quality_sum=quality_sum,
                rb_limit=rb_limit,
                limit=2,
            )
            if not options:
                regret_key = (float("inf"), float("inf"))
                option = None
            elif len(options) == 1:
                regret_key = (float("inf"), -options[0][0][0])
                option = options[0]
            else:
                regret = options[1][0][0] - options[0][0][0]
                regret_key = (regret, -options[0][0][0])
                option = options[0]

            if selected_key is None or regret_key > selected_key:
                selected = user_id
                selected_option = option
                selected_key = regret_key

        if selected_option is None:
            answer = make_answer(selected, 0, UNSCHEDULED_PERIOD, 0, UNSCHEDULED_PERIOD)
        else:
            _, rb, period, offset, distance = selected_option
            place_csi(occupied, rb, period, offset)
            answer = make_answer(selected, rb, period, offset, distance)

        quality_sum += answer["quality"]
        answers.append(answer)
        remaining.remove(selected)

    return answers, objective_value(answers, initial_occupied)


def multistart_solve(
    users,
    dl,
    ul,
    base_penalty=60,
    attempts=50,
    seed=123,
    quality_weight=1.0,
    scoring="adaptive",
    objective="product",
    initial_occupied=None,
):
    random.seed(seed)

    best_answers = None
    best_key = None
    indexed_users = list(range(len(users)))

    for _ in range(attempts):
        order = indexed_users[:]
        random.shuffle(order)

        answers, _ = greedy_solve(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
            order=order,
            initial_occupied=initial_occupied,
        )

        key = objective_key(answers, objective, initial_occupied)
        if best_key is None or key < best_key:
            best_key = key
            best_answers = answers

    return best_answers, objective_value(best_answers, initial_occupied)


def local_search_one_user_moves(
    users,
    answers,
    dl,
    ul,
    max_rounds=2,
    objective="product",
    initial_occupied=None,
):
    answers = [a.copy() for a in answers]
    best_key = objective_key(answers, objective, initial_occupied)

    for _ in range(max_rounds):
        improved = False

        for idx, current_answer in enumerate(answers):
            user_id = current_answer["user_id"]
            user = users[user_id]

            occupied = build_occupied_from_answers(answers, initial_occupied)
            remove_csi(
                occupied,
                current_answer["rb"],
                current_answer["period"],
                current_answer["offset"],
            )

            local_best_answer = current_answer.copy()
            local_best_key = best_key

            quality_without_user = (
                solution_stats(answers, initial_occupied)["quality_sum"] - current_answer["quality"]
            )

            options = best_options_for_one_user(
                occupied=occupied,
                csi_periods=user["csi_periods"],
                srs_period=user["srs_period"],
                srs_offsets=user["srs_offsets"],
                dl=dl,
                ul=ul,
                scoring="marginal",
                objective=objective,
                current_quality_sum=quality_without_user,
                limit=128,
            )

            for _, rb, period, offset, distance in options:
                candidate_answer = make_answer(user_id, rb, period, offset, distance)
                trial_answers = answers[:]
                trial_answers[idx] = candidate_answer
                trial_key = objective_key(trial_answers, objective, initial_occupied)

                if trial_key < local_best_key:
                    local_best_key = trial_key
                    local_best_answer = candidate_answer

            if local_best_key < best_key:
                answers[idx] = local_best_answer
                best_key = local_best_key
                improved = True

        if not improved:
            break

    return answers, objective_value(answers, initial_occupied)


def rb_loads(answers):
    loads = {rb: 0 for rb in range(R)}
    for answer in answers:
        if answer["period"] == UNSCHEDULED_PERIOD:
            continue
        loads[answer["rb"]] += len(periodic_slots(answer["period"], answer["offset"]))
    return loads


def answer_mask(answer):
    if answer["period"] == UNSCHEDULED_PERIOD:
        return 0
    return periodic_mask(answer["period"], answer["offset"])


def choose_destroy_indices(answers, destroy_size, random_fraction, seed):
    random.seed(seed)
    n = len(answers)
    count = min(destroy_size, n)
    random_count = int(count * random_fraction)
    structured_count = count - random_count
    loads = rb_loads(answers)

    ranked = sorted(
        list(enumerate(answers)),
        key=lambda x: (
            x[1]["quality"],
            loads.get(x[1]["rb"], 0),
            -x[1]["period"],
        ),
        reverse=True,
    )

    selected = [idx for idx, _ in ranked[:structured_count]]
    remaining = [i for i in range(n) if i not in selected]
    if random_count > 0 and remaining:
        selected += random.sample(remaining, k=min(random_count, len(remaining)))
    return selected


def repair_removed_users(
    users,
    answers,
    removed_indices,
    dl,
    ul,
    base_penalty=60,
    attempts=20,
    seed=123,
    quality_weight=1.0,
    scoring="marginal",
    objective="product",
    initial_occupied=None,
):
    random.seed(seed)

    base_answers = [a.copy() for a in answers]
    removed_answers = [base_answers[i] for i in removed_indices]

    for i in sorted(removed_indices, reverse=True):
        base_answers.pop(i)

    best_answers = None
    best_key = None

    for _ in range(attempts):
        trial_answers = [a.copy() for a in base_answers]
        occupied = build_occupied_from_answers(trial_answers, initial_occupied)
        quality_sum = solution_stats(trial_answers, initial_occupied)["quality_sum"]

        order = removed_answers[:]
        random.shuffle(order)

        success = True

        for old_answer in order:
            user_id = old_answer["user_id"]
            user = users[user_id]

            rb, period, offset, distance = find_best_for_one_user(
                occupied=occupied,
                csi_periods=user["csi_periods"],
                srs_period=user["srs_period"],
                srs_offsets=user["srs_offsets"],
                dl=dl,
                ul=ul,
                base_penalty=base_penalty,
                quality_weight=quality_weight,
                scoring=scoring,
                objective=objective,
                current_quality_sum=quality_sum,
            )

            if period == UNSCHEDULED_PERIOD:
                success = False
                break

            place_csi(occupied, rb, period, offset)
            answer = make_answer(user_id, rb, period, offset, distance)
            trial_answers.append(answer)
            quality_sum += answer["quality"]

        if not success:
            continue

        key = objective_key(trial_answers, objective, initial_occupied)
        if best_key is None or key < best_key:
            best_key = key
            best_answers = trial_answers

    if best_answers is None:
        return answers, objective_value(answers, initial_occupied)

    return best_answers, objective_value(best_answers, initial_occupied)


def repair_removed_users_regret(
    users,
    answers,
    removed_indices,
    dl,
    ul,
    base_penalty=60,
    quality_weight=1.0,
    scoring="marginal",
    objective="product",
    initial_occupied=None,
):
    base_answers = [a.copy() for a in answers]
    remaining_answers = [base_answers[i] for i in removed_indices]

    for i in sorted(removed_indices, reverse=True):
        base_answers.pop(i)

    trial_answers = [a.copy() for a in base_answers]
    occupied = build_occupied_from_answers(trial_answers, initial_occupied)
    quality_sum = solution_stats(trial_answers, initial_occupied)["quality_sum"]

    while remaining_answers:
        selected_idx = None
        selected_option = None
        selected_key = None

        for idx, old_answer in enumerate(remaining_answers):
            user = users[old_answer["user_id"]]
            options = best_options_for_one_user(
                occupied=occupied,
                csi_periods=user["csi_periods"],
                srs_period=user["srs_period"],
                srs_offsets=user["srs_offsets"],
                dl=dl,
                ul=ul,
                base_penalty=base_penalty,
                quality_weight=quality_weight,
                scoring=scoring,
                objective=objective,
                current_quality_sum=quality_sum,
                limit=2,
            )
            if not options:
                return answers, objective_value(answers, initial_occupied)
            if len(options) == 1:
                regret = 10**9
            else:
                regret = options[1][0][0] - options[0][0][0]
            choice_key = (regret, -options[0][0][0], -old_answer["quality"])
            if selected_key is None or choice_key > selected_key:
                selected_key = choice_key
                selected_idx = idx
                selected_option = options[0]

        _, rb, period, offset, distance = selected_option
        if period == UNSCHEDULED_PERIOD:
            return answers, objective_value(answers, initial_occupied)

        old_answer = remaining_answers.pop(selected_idx)
        answer = make_answer(old_answer["user_id"], rb, period, offset, distance)
        place_csi(occupied, rb, period, offset)
        quality_sum += answer["quality"]
        trial_answers.append(answer)

    return trial_answers, objective_value(trial_answers, initial_occupied)


def destroy_repair_search(
    users,
    answers,
    dl,
    ul,
    base_penalty=60,
    destroy_size=10,
    rounds=10,
    repair_attempts=20,
    seed=123,
    quality_weight=1.0,
    random_fraction=0.3,
    scoring="marginal",
    objective="product",
    initial_occupied=None,
):
    answers = [a.copy() for a in answers]
    best_key = objective_key(answers, objective, initial_occupied)

    for round_id in range(rounds):
        selected_indices = choose_destroy_indices(
            answers,
            destroy_size=destroy_size,
            random_fraction=random_fraction,
            seed=seed + round_id,
        )

        candidate_answers, _ = repair_removed_users(
            users=users,
            answers=answers,
            removed_indices=selected_indices,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            attempts=repair_attempts,
            seed=seed + round_id,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )

        candidate_key = objective_key(candidate_answers, objective, initial_occupied)
        if candidate_key < best_key:
            answers = candidate_answers
            best_key = candidate_key

            answers, _ = local_search_one_user_moves(
                users,
                answers,
                dl,
                ul,
                max_rounds=1,
                objective=objective,
                initial_occupied=initial_occupied,
            )
            best_key = objective_key(answers, objective, initial_occupied)

    return answers, objective_value(answers, initial_occupied)


def answers_to_output(answers):
    answers_sorted = sorted(answers, key=lambda x: x["user_id"])
    return [(a["rb"], a["period"], a["offset"]) for a in answers_sorted]


def final_solver(
    users,
    dl,
    ul,
    base_penalty=60,
    attempts=5,
    local_rounds=1,
    seed=123,
    quality_weight=1.0,
    random_destroy_fraction=0.3,
    scoring="marginal",
    objective="product",
    use_regret_start=False,
    initial_occupied=None,
):
    if use_regret_start:
        answers, _ = regret_greedy_solve(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )
    else:
        answers, _ = multistart_solve(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            attempts=attempts,
            seed=seed,
            quality_weight=quality_weight,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )

    answers, _ = local_search_one_user_moves(
        users,
        answers,
        dl,
        ul,
        max_rounds=local_rounds,
        objective=objective,
        initial_occupied=initial_occupied,
    )

    destroy_size = max(8, min(25, len(users) // 8))
    answers, score = destroy_repair_search(
        users=users,
        answers=answers,
        dl=dl,
        ul=ul,
        base_penalty=base_penalty,
        destroy_size=destroy_size,
        rounds=2,
        repair_attempts=4,
        seed=seed + 777,
        quality_weight=quality_weight,
        random_fraction=random_destroy_fraction,
        scoring=scoring,
        objective=objective,
        initial_occupied=initial_occupied,
    )

    return answers_to_output(answers), score


def portfolio_solver(users, dl, ul, objective="product", initial_occupied=None):
    configs = [
        {
            "name": "adaptive-multistart",
            "base_penalty": 60,
            "attempts": 5,
            "local_rounds": 1,
            "seed_shift": 1000,
            "quality_weight": 1.0,
            "random_destroy_fraction": 0.3,
            "scoring": "adaptive",
            "use_regret_start": False,
        },
        {
            "name": "marginal-multistart",
            "base_penalty": 60,
            "attempts": 5,
            "local_rounds": 1,
            "seed_shift": 3000,
            "quality_weight": 1.0,
            "random_destroy_fraction": 0.3,
            "scoring": "marginal",
            "use_regret_start": False,
        },
        {
            "name": "marginal-regret",
            "base_penalty": 60,
            "attempts": 1,
            "local_rounds": 1,
            "seed_shift": 5000,
            "quality_weight": 1.0,
            "random_destroy_fraction": 0.3,
            "scoring": "marginal",
            "use_regret_start": True,
        },
    ]

    best_output = None
    best_answers = None
    best_key = None
    best_config = None

    for cfg in configs:
        output, _ = final_solver(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=cfg["base_penalty"],
            attempts=cfg["attempts"],
            local_rounds=cfg["local_rounds"],
            seed=cfg["seed_shift"],
            quality_weight=cfg["quality_weight"],
            random_destroy_fraction=cfg["random_destroy_fraction"],
            scoring=cfg["scoring"],
            objective=objective,
            use_regret_start=cfg["use_regret_start"],
            initial_occupied=initial_occupied,
        )
        answers = output_to_answers(users, output)
        key = objective_key(answers, objective, initial_occupied)

        if best_key is None or key < best_key:
            best_key = key
            best_output = output
            best_answers = answers
            best_config = cfg

    return best_output, objective_value(best_answers, initial_occupied), best_config


def budget_sweep_greedy_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    scoring_modes=None,
    max_rb=None,
):
    initial_occupied = clone_occupied(initial_occupied)
    initial_used = len({rb for rb, mask in enumerate(initial_occupied) if mask})
    scoring_modes = scoring_modes or ["adaptive"]
    best_answers = None
    best_key = None
    best_detail = None

    warm_starts = []
    for scoring in ["adaptive", "marginal"]:
        answers, _ = greedy_solve(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=60,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        warm_starts.append((scoring, answers))
        key = objective_key(answers, objective, initial_occupied)
        if best_key is None or key < best_key:
            best_key = key
            best_answers = answers
            best_detail = f"{scoring}@free"

    if max_rb is None:
        best_rb = min(solution_stats(answers, initial_occupied)["rb_used"] for _, answers in warm_starts)
        max_rb = min(R, max(initial_used, best_rb + 2))
    else:
        max_rb = min(max_rb, R)
    min_rb = max(1, initial_used, max(1, max_rb - 7))

    for scoring in scoring_modes:
        for rb_limit in range(min_rb, max_rb + 1):
            answers, _ = greedy_solve(
                users=users,
                dl=dl,
                ul=ul,
                base_penalty=60,
                scoring=scoring,
                objective=objective,
                rb_limit=rb_limit,
                initial_occupied=initial_occupied,
            )
            key = objective_key(answers, objective, initial_occupied)
            if best_key is None or key < best_key:
                best_key = key
                best_answers = answers
                best_detail = f"{scoring}@K={rb_limit}"

            if solution_stats(answers, initial_occupied)["unscheduled"] == 0:
                # Larger K can still improve quality, but if product already
                # worsens for several consecutive K values it is rarely useful.
                pass

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def learned_budget_hint(users, dl, ul, initial_occupied=None):
    initial_occupied = clone_occupied(initial_occupied)
    uplink_slots = sum(1 for slot in range(L) if is_uplink_slot(slot, dl, ul))
    min_slot_demand = 0

    for user in users:
        options = time_options_for_user(user, dl, ul)
        if not options:
            continue
        best_by_quality = options[0]
        min_slot_demand += popcount(best_by_quality[2])

    active_initial = len({rb for rb, mask in enumerate(initial_occupied) if mask})
    load_lower_bound = max(1, (min_slot_demand + uplink_slots - 1) // uplink_slots)

    # Quantized calibrated model: round(0.72 * lower_bound) using integer
    # arithmetic. The deterministic scheduler still enforces feasibility.
    predicted_extra = max(1, (18 * load_lower_bound + 12) // 25)
    return min(R, max(active_initial, predicted_extra))


def learned_budget_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    window=0,
):
    initial_occupied = clone_occupied(initial_occupied)
    hint = learned_budget_hint(users, dl, ul, initial_occupied)
    free_answers, _ = greedy_solve(
        users=users,
        dl=dl,
        ul=ul,
        base_penalty=60,
        scoring="adaptive",
        objective=objective,
        initial_occupied=initial_occupied,
    )
    best_answers = free_answers
    best_key = objective_key(free_answers, objective, initial_occupied)
    best_detail = "adaptive@free"
    free_rb = solution_stats(free_answers, initial_occupied)["rb_used"]

    if hint >= free_rb:
        return best_answers, objective_value(best_answers, initial_occupied), best_detail

    for rb_limit in range(max(1, hint - window), min(R, hint + window) + 1):
        answers, _ = greedy_solve(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=60,
            scoring="adaptive",
            objective=objective,
            rb_limit=rb_limit,
            initial_occupied=initial_occupied,
        )
        key = objective_key(answers, objective, initial_occupied)
        if solution_stats(answers, initial_occupied)["unscheduled"] == 0 and key < best_key:
            best_key = key
            best_answers = answers
            best_detail = f"Khat={hint},K={rb_limit}"

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def rb_eviction_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    max_passes=2,
    max_targets=12,
):
    initial_occupied = clone_occupied(initial_occupied)
    best_answers, _, best_detail = fast_portfolio_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    best_answers = [answer.copy() for answer in best_answers]
    best_key = objective_key(best_answers, objective, initial_occupied)

    for _ in range(max_passes):
        improved = False
        loads = rb_loads(best_answers)
        active_targets = [
            rb
            for rb, load in sorted(loads.items(), key=lambda item: (item[1], item[0]))
            if load > 0 and initial_occupied[rb] == 0
        ][:max_targets]

        for target_rb in active_targets:
            current_stats = solution_stats(best_answers, initial_occupied)
            if current_stats["rb_used"] <= 1:
                break

            fixed_answers = [
                answer.copy()
                for answer in best_answers
                if answer["period"] == UNSCHEDULED_PERIOD or answer["rb"] != target_rb
            ]
            removed_answers = [
                answer.copy()
                for answer in best_answers
                if answer["period"] != UNSCHEDULED_PERIOD and answer["rb"] == target_rb
            ]
            if not removed_answers:
                continue

            rb_limit = current_stats["rb_used"] - 1
            removed_orders = [
                sorted(
                    removed_answers,
                    key=lambda answer: (
                        -popcount(periodic_mask(answer["period"], answer["offset"])),
                        -answer["quality"],
                        answer["user_id"],
                    ),
                ),
                sorted(removed_answers, key=lambda answer: (-answer["quality"], answer["user_id"])),
                sorted(removed_answers, key=lambda answer: (answer["period"], answer["user_id"])),
            ]

            for scoring, base_penalty in [("marginal", 60), ("adaptive", 80), ("adaptive", 120)]:
                for order in removed_orders:
                    trial_answers = [answer.copy() for answer in fixed_answers]
                    occupied = build_occupied_from_answers(trial_answers, initial_occupied)
                    quality_sum = solution_stats(trial_answers, initial_occupied)["quality_sum"]
                    success = True

                    for old_answer in order:
                        user_id = old_answer["user_id"]
                        user = users[user_id]
                        rb, period, offset, distance = find_best_for_one_user(
                            occupied=occupied,
                            csi_periods=user["csi_periods"],
                            srs_period=user["srs_period"],
                            srs_offsets=user["srs_offsets"],
                            dl=dl,
                            ul=ul,
                            base_penalty=base_penalty,
                            scoring=scoring,
                            objective=objective,
                            current_quality_sum=quality_sum,
                            rb_limit=rb_limit,
                        )
                        if period == UNSCHEDULED_PERIOD or rb == target_rb:
                            success = False
                            break

                        answer = make_answer(user_id, rb, period, offset, distance)
                        place_csi(occupied, rb, period, offset)
                        quality_sum += answer["quality"]
                        trial_answers.append(answer)

                    if not success:
                        continue

                    if validate_answers(users, trial_answers, dl, ul, initial_occupied):
                        continue

                    trial_key = objective_key(trial_answers, objective, initial_occupied)
                    if trial_key < best_key:
                        best_key = trial_key
                        best_answers = trial_answers
                        best_detail = (
                            f"evict-rb{target_rb}:{scoring}{base_penalty},"
                            f"{len(removed_answers)} users"
                        )
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def fast_portfolio_solver(users, dl, ul, objective="product", initial_occupied=None):
    initial_occupied = clone_occupied(initial_occupied)
    candidates = []

    for base_penalty in [40, 60, 80]:
        answers, _ = greedy_solve(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            scoring="adaptive",
            objective=objective,
            initial_occupied=initial_occupied,
        )
        candidates.append((f"adaptive{base_penalty}", answers))

    answers, _ = greedy_solve(
        users=users,
        dl=dl,
        ul=ul,
        scoring="marginal",
        objective=objective,
        initial_occupied=initial_occupied,
    )
    candidates.append(("marginal", answers))

    answers, _ = regret_greedy_solve(
        users=users,
        dl=dl,
        ul=ul,
        scoring="marginal",
        objective=objective,
        initial_occupied=initial_occupied,
    )
    candidates.append(("regret-marginal", answers))

    answers, _, detail = learned_budget_solver(
        users=users,
        dl=dl,
        ul=ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    candidates.append((f"learned:{detail}", answers))

    best_detail = None
    best_answers = None
    best_key = None
    for detail, answers in candidates:
        key = objective_key(answers, objective, initial_occupied)
        if best_key is None or key < best_key:
            best_key = key
            best_answers = answers
            best_detail = detail

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def compression_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    max_drop=8,
):
    initial_occupied = clone_occupied(initial_occupied)
    best_answers, _, best_detail = fast_portfolio_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    best_key = objective_key(best_answers, objective, initial_occupied)
    best_rb = solution_stats(best_answers, initial_occupied)["rb_used"]

    min_rb = max(1, best_rb - max_drop)
    for rb_limit in range(best_rb - 1, min_rb - 1, -1):
        for scoring, base_penalty in [("adaptive", 60), ("adaptive", 80), ("marginal", 60)]:
            answers, _ = greedy_solve(
                users=users,
                dl=dl,
                ul=ul,
                base_penalty=base_penalty,
                scoring=scoring,
                objective=objective,
                rb_limit=rb_limit,
                initial_occupied=initial_occupied,
            )
            stats = solution_stats(answers, initial_occupied)
            if stats["unscheduled"] != 0:
                continue
            key = objective_key(answers, objective, initial_occupied)
            if key < best_key:
                best_key = key
                best_answers = answers
                best_detail = f"compress:{scoring}{base_penalty}@K={rb_limit}"

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def alns_destroy_indices(answers, operator, destroy_size, seed):
    random.seed(seed)
    scheduled = [
        (idx, answer)
        for idx, answer in enumerate(answers)
        if answer["period"] != UNSCHEDULED_PERIOD
    ]
    if not scheduled:
        return []

    count = min(destroy_size, len(scheduled))
    loads = rb_loads(answers)

    if operator == "random":
        return random.sample([idx for idx, _ in scheduled], k=count)

    if operator == "worst-quality":
        ranked = sorted(
            scheduled,
            key=lambda item: (item[1]["quality"], loads.get(item[1]["rb"], 0)),
            reverse=True,
        )
        return [idx for idx, _ in ranked[:count]]

    if operator == "crowded-rb":
        target_rb = max(loads, key=lambda rb: (loads[rb], -rb))
        selected = [idx for idx, answer in scheduled if answer["rb"] == target_rb]
        if len(selected) < count:
            fillers = sorted(
                (
                    (idx, answer)
                    for idx, answer in scheduled
                    if idx not in selected
                ),
                key=lambda item: (item[1]["quality"], loads.get(item[1]["rb"], 0)),
                reverse=True,
            )
            selected += [idx for idx, _ in fillers[: count - len(selected)]]
        return selected[:count]

    if operator == "light-rb":
        nonempty = [(rb, load) for rb, load in loads.items() if load > 0]
        target_rb = min(nonempty, key=lambda item: (item[1], item[0]))[0]
        selected = [idx for idx, answer in scheduled if answer["rb"] == target_rb]
        if len(selected) < count:
            fillers = sorted(
                (
                    (idx, answer)
                    for idx, answer in scheduled
                    if idx not in selected
                ),
                key=lambda item: (loads.get(item[1]["rb"], 0), item[1]["quality"]),
                reverse=True,
            )
            selected += [idx for idx, _ in fillers[: count - len(selected)]]
        return selected[:count]

    if operator == "blockers":
        ranked = sorted(
            scheduled,
            key=lambda item: (
                loads.get(item[1]["rb"], 0),
                popcount(periodic_mask(item[1]["period"], item[1]["offset"])),
                item[1]["quality"],
            ),
            reverse=True,
        )
        return [idx for idx, _ in ranked[:count]]

    if operator == "conflict-graph":
        pressure = [0] * L
        for _, answer in scheduled:
            for slot in periodic_slots(answer["period"], answer["offset"]):
                pressure[slot] += 1
        ranked = sorted(
            scheduled,
            key=lambda item: (
                sum(pressure[slot] for slot in periodic_slots(item[1]["period"], item[1]["offset"])),
                loads.get(item[1]["rb"], 0),
                item[1]["quality"],
            ),
            reverse=True,
        )
        return [idx for idx, _ in ranked[:count]]

    if operator == "related-time":
        seed_idx, seed_answer = random.choice(scheduled)
        seed_mask = answer_mask(seed_answer)
        ranked = sorted(
            scheduled,
            key=lambda item: (
                item[0] == seed_idx,
                item[1]["rb"] == seed_answer["rb"],
                popcount(answer_mask(item[1]) & seed_mask),
                -abs(item[1]["period"] - seed_answer["period"]),
                -abs(item[1]["offset"] - seed_answer["offset"]),
                item[1]["quality"],
            ),
            reverse=True,
        )
        return [idx for idx, _ in ranked[:count]]

    if operator == "evacuation-related":
        nonempty = [(rb, load) for rb, load in loads.items() if load > 0]
        target_rb = min(nonempty, key=lambda item: (item[1], item[0]))[0]
        target_items = [(idx, answer) for idx, answer in scheduled if answer["rb"] == target_rb]
        target_mask = 0
        for _, answer in target_items:
            target_mask |= answer_mask(answer)
        selected = [idx for idx, _ in target_items]
        if len(selected) < count:
            related = sorted(
                (
                    (idx, answer)
                    for idx, answer in scheduled
                    if idx not in selected
                ),
                key=lambda item: (
                    popcount(answer_mask(item[1]) & target_mask),
                    loads.get(item[1]["rb"], 0),
                    item[1]["quality"],
                ),
                reverse=True,
            )
            selected += [idx for idx, _ in related[: count - len(selected)]]
        return selected[:count]

    raise ValueError(f"unknown ALNS operator: {operator}")


def weighted_choice(weights, rng):
    total = sum(weights.values())
    pick = rng.random() * total
    running = 0.0
    for name, weight in weights.items():
        running += weight
        if running >= pick:
            return name
    return next(iter(weights))


def repair_with_mode(
    mode,
    users,
    answers,
    removed_indices,
    dl,
    ul,
    base_penalty,
    repair_attempts,
    seed,
    scoring,
    objective,
    initial_occupied,
):
    if mode == "conflict":
        return repair_removed_users_regret(
            users=users,
            answers=answers,
            removed_indices=removed_indices,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            scoring="conflict-aware",
            objective=objective,
            initial_occupied=initial_occupied,
        )
    if mode == "regret":
        return repair_removed_users_regret(
            users=users,
            answers=answers,
            removed_indices=removed_indices,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )
    return repair_removed_users(
        users=users,
        answers=answers,
        removed_indices=removed_indices,
        dl=dl,
        ul=ul,
        base_penalty=base_penalty,
        attempts=repair_attempts,
        seed=seed,
        scoring=scoring,
        objective=objective,
        initial_occupied=initial_occupied,
    )


def adaptive_alns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    rounds=16,
    repair_attempts=5,
    seed=4400,
):
    answers, _, detail = rb_eviction_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
        max_passes=1,
    )
    best_key = objective_key(answers, objective, initial_occupied)
    best_detail = f"start:{detail}"
    operators = [
        "random",
        "worst-quality",
        "crowded-rb",
        "light-rb",
        "blockers",
        "related-time",
        "evacuation-related",
    ]
    weights = {operator: 1.0 for operator in operators}
    rng = random.Random(seed)
    destroy_size = max(8, min(32, len(users) // 10))

    for round_id in range(rounds):
        operator = weighted_choice(weights, rng)
        removed_indices = alns_destroy_indices(
            answers,
            operator=operator,
            destroy_size=destroy_size,
            seed=seed + round_id,
        )
        if not removed_indices:
            continue

        candidate, _ = repair_removed_users(
            users=users,
            answers=answers,
            removed_indices=removed_indices,
            dl=dl,
            ul=ul,
            base_penalty=80 if operator in {"light-rb", "blockers", "evacuation-related"} else 60,
            attempts=repair_attempts,
            seed=seed + 1000 + round_id,
            scoring="adaptive" if operator in {"light-rb", "evacuation-related"} else "marginal",
            objective=objective,
            initial_occupied=initial_occupied,
        )
        if validate_answers(users, candidate, dl, ul, initial_occupied):
            weights[operator] *= 0.9
            continue

        key = objective_key(candidate, objective, initial_occupied)
        if key < best_key:
            answers = candidate
            best_key = key
            weights[operator] += 3.0
            best_detail = f"alns:{operator}@round={round_id}"
        else:
            weights[operator] = max(0.2, weights[operator] * 0.92)

    return answers, objective_value(answers, initial_occupied), best_detail


def annealed_alns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    rounds=28,
    repair_attempts=4,
    seed=7600,
):
    start_answers, _, detail = rb_eviction_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
        max_passes=1,
    )
    current_answers = [answer.copy() for answer in start_answers]
    best_answers = [answer.copy() for answer in start_answers]
    current_scalar = objective_scalar(current_answers, objective, initial_occupied)
    best_key = objective_key(best_answers, objective, initial_occupied)
    best_detail = f"start:{detail}"

    operators = [
        "random",
        "worst-quality",
        "crowded-rb",
        "light-rb",
        "blockers",
        "related-time",
        "evacuation-related",
    ]
    repair_modes = ["random", "regret"]
    weights = {(operator, mode): 1.0 for operator in operators for mode in repair_modes}
    rng = random.Random(seed)
    base_destroy = max(8, min(40, len(users) // 9))
    temperature = max(100.0, current_scalar * 0.015)

    for round_id in range(rounds):
        operator, repair_mode = weighted_choice(weights, rng)
        jitter = rng.choice([-4, 0, 4, 8])
        destroy_size = max(6, min(len(users), base_destroy + jitter))
        removed_indices = alns_destroy_indices(
            current_answers,
            operator=operator,
            destroy_size=destroy_size,
            seed=seed + round_id,
        )
        if not removed_indices:
            continue

        scoring = "adaptive" if operator in {"light-rb", "blockers", "evacuation-related"} else "marginal"
        candidate, _ = repair_with_mode(
            mode=repair_mode,
            users=users,
            answers=current_answers,
            removed_indices=removed_indices,
            dl=dl,
            ul=ul,
            base_penalty=90 if scoring == "adaptive" else 60,
            repair_attempts=repair_attempts,
            seed=seed + 1000 + round_id,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        if validate_answers(users, candidate, dl, ul, initial_occupied):
            weights[(operator, repair_mode)] *= 0.85
            temperature *= 0.93
            continue

        candidate_scalar = objective_scalar(candidate, objective, initial_occupied)
        delta = candidate_scalar - current_scalar
        accept = delta <= 0
        if not accept and delta < 10**11:
            accept = rng.random() < math.exp(-delta / max(temperature, 1.0))

        if accept:
            current_answers = candidate
            current_scalar = candidate_scalar
            weights[(operator, repair_mode)] += 0.4

        candidate_key = objective_key(candidate, objective, initial_occupied)
        if candidate_key < best_key:
            best_answers = candidate
            best_key = candidate_key
            best_detail = f"annealed:{operator}/{repair_mode}@round={round_id}"
            weights[(operator, repair_mode)] += 4.0
        elif not accept:
            weights[(operator, repair_mode)] = max(0.2, weights[(operator, repair_mode)] * 0.9)

        temperature *= 0.90

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def bandit_alns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    rounds=32,
    repair_attempts=4,
    seed=9100,
):
    start_answers, _, detail = rb_eviction_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
        max_passes=1,
    )
    current_answers = [answer.copy() for answer in start_answers]
    best_answers = [answer.copy() for answer in start_answers]
    current_scalar = objective_scalar(current_answers, objective, initial_occupied)
    best_key = objective_key(best_answers, objective, initial_occupied)
    best_detail = f"start:{detail}"

    operators = [
        "random",
        "worst-quality",
        "crowded-rb",
        "light-rb",
        "blockers",
        "related-time",
        "evacuation-related",
    ]
    repair_modes = ["random", "regret"]
    size_multipliers = [
        ("base", 1.0),
        ("large", 1.35),
        ("xlarge", 1.65),
    ]
    operator_arms = [
        (operator, repair_mode)
        for operator in operators
        for repair_mode in repair_modes
    ]
    operator_weights = {arm: 1.0 for arm in operator_arms}
    size_weights = {size_name: 1.0 for size_name, _ in size_multipliers}
    size_by_name = dict(size_multipliers)
    rng = random.Random(seed)
    base_destroy = max(8, min(40, len(users) // 9))
    temperature = max(100.0, current_scalar * 0.018)

    for round_id in range(rounds):
        operator, repair_mode = weighted_choice(operator_weights, rng)
        size_name = weighted_choice(size_weights, rng)
        size_mult = size_by_name[size_name]
        jitter = rng.choice([-2, 0, 2])
        destroy_size = max(6, min(len(users), int(round(base_destroy * size_mult)) + jitter))
        removed_indices = alns_destroy_indices(
            current_answers,
            operator=operator,
            destroy_size=destroy_size,
            seed=seed + round_id,
        )
        if not removed_indices:
            operator_weights[(operator, repair_mode)] *= 0.9
            size_weights[size_name] *= 0.9
            continue

        scoring = "adaptive" if operator in {"light-rb", "blockers", "evacuation-related"} else "marginal"
        candidate, _ = repair_with_mode(
            mode=repair_mode,
            users=users,
            answers=current_answers,
            removed_indices=removed_indices,
            dl=dl,
            ul=ul,
            base_penalty=95 if scoring == "adaptive" else 60,
            repair_attempts=repair_attempts,
            seed=seed + 1000 + round_id,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        if validate_answers(users, candidate, dl, ul, initial_occupied):
            operator_weights[(operator, repair_mode)] = max(
                0.2, operator_weights[(operator, repair_mode)] * 0.82
            )
            size_weights[size_name] = max(0.2, size_weights[size_name] * 0.82)
            temperature *= 0.94
            continue

        candidate_scalar = objective_scalar(candidate, objective, initial_occupied)
        delta = candidate_scalar - current_scalar
        accept = delta <= 0
        if not accept and delta < 10**11:
            accept = rng.random() < math.exp(-delta / max(temperature, 1.0))

        if accept:
            current_answers = candidate
            current_scalar = candidate_scalar
            operator_weights[(operator, repair_mode)] += 0.35
            size_weights[size_name] += 0.2

        candidate_key = objective_key(candidate, objective, initial_occupied)
        if candidate_key < best_key:
            improvement = max(1.0, objective_scalar(best_answers, objective, initial_occupied) - candidate_scalar)
            best_answers = candidate
            best_key = candidate_key
            best_detail = f"bandit:{operator}/{repair_mode}/{size_name}@round={round_id}"
            reward = 3.0 + min(3.0, math.log1p(improvement) / 5)
            operator_weights[(operator, repair_mode)] += reward
            size_weights[size_name] += 0.7 * reward
        elif not accept:
            operator_weights[(operator, repair_mode)] = max(
                0.2, operator_weights[(operator, repair_mode)] * 0.9
            )
            size_weights[size_name] = max(0.2, size_weights[size_name] * 0.92)

        temperature *= 0.89

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def ucb_choice(stats, total_pulls, rng, exploration=0.8, epsilon=0.08):
    if rng.random() < epsilon:
        return rng.choice(list(stats))

    log_total = math.log(max(2, total_pulls))
    best_key = None
    best_score = None
    for key, state in stats.items():
        count = max(1, state["count"])
        score = state["mean"] + exploration * math.sqrt(log_total / count)
        if best_score is None or score > best_score:
            best_key = key
            best_score = score
    return best_key


def update_ucb(stats, key, reward):
    state = stats[key]
    state["count"] += 1
    state["mean"] += (reward - state["mean"]) / state["count"]


def ucb_bandit_alns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    rounds=32,
    repair_attempts=4,
    seed=12300,
):
    start_answers, _, detail = rb_eviction_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
        max_passes=1,
    )
    current_answers = [answer.copy() for answer in start_answers]
    best_answers = [answer.copy() for answer in start_answers]
    current_scalar = objective_scalar(current_answers, objective, initial_occupied)
    best_scalar = objective_scalar(best_answers, objective, initial_occupied)
    best_key = objective_key(best_answers, objective, initial_occupied)
    best_detail = f"start:{detail}"

    operator_priors = {
        ("evacuation-related", "regret"): (3, 1.20),
        ("related-time", "regret"): (3, 1.00),
        ("light-rb", "regret"): (2, 0.75),
        ("crowded-rb", "regret"): (2, 0.65),
        ("blockers", "regret"): (2, 0.55),
        ("evacuation-related", "random"): (1, 0.45),
        ("related-time", "random"): (1, 0.40),
    }
    size_priors = {
        "base": (2, 0.70, 1.0),
        "large": (3, 1.00, 1.35),
        "xlarge": (3, 1.15, 1.65),
    }
    operator_stats = {
        key: {"count": count, "mean": mean}
        for key, (count, mean) in operator_priors.items()
    }
    size_stats = {
        key: {"count": count, "mean": mean}
        for key, (count, mean, _) in size_priors.items()
    }
    size_multiplier = {key: mult for key, (_, _, mult) in size_priors.items()}
    rng = random.Random(seed)
    base_destroy = max(8, min(40, len(users) // 9))
    temperature = max(100.0, current_scalar * 0.014)
    total_operator_pulls = sum(state["count"] for state in operator_stats.values())
    total_size_pulls = sum(state["count"] for state in size_stats.values())

    warmup = [
        (("evacuation-related", "regret"), "xlarge"),
        (("related-time", "regret"), "large"),
        (("light-rb", "regret"), "large"),
        (("crowded-rb", "regret"), "base"),
    ]

    for round_id in range(rounds):
        if round_id < len(warmup):
            operator_key, size_name = warmup[round_id]
        else:
            epsilon = max(0.03, 0.16 * (1 - round_id / max(1, rounds)))
            operator_key = ucb_choice(
                operator_stats,
                total_operator_pulls,
                rng,
                exploration=0.75,
                epsilon=epsilon,
            )
            size_name = ucb_choice(
                size_stats,
                total_size_pulls,
                rng,
                exploration=0.55,
                epsilon=epsilon,
            )
        operator, repair_mode = operator_key
        size_mult = size_multiplier[size_name]
        destroy_size = max(
            6,
            min(len(users), int(round(base_destroy * size_mult)) + rng.choice([-2, 0, 2])),
        )
        removed_indices = alns_destroy_indices(
            current_answers,
            operator=operator,
            destroy_size=destroy_size,
            seed=seed + round_id,
        )
        if not removed_indices:
            update_ucb(operator_stats, operator_key, 0.0)
            update_ucb(size_stats, size_name, 0.0)
            total_operator_pulls += 1
            total_size_pulls += 1
            continue

        scoring = "adaptive" if operator in {"light-rb", "blockers", "evacuation-related"} else "marginal"
        candidate, _ = repair_with_mode(
            mode=repair_mode,
            users=users,
            answers=current_answers,
            removed_indices=removed_indices,
            dl=dl,
            ul=ul,
            base_penalty=95 if scoring == "adaptive" else 60,
            repair_attempts=repair_attempts,
            seed=seed + 1000 + round_id,
            scoring=scoring,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        if validate_answers(users, candidate, dl, ul, initial_occupied):
            reward = 0.0
        else:
            candidate_scalar = objective_scalar(candidate, objective, initial_occupied)
            current_stats = solution_stats(current_answers, initial_occupied)
            candidate_stats = solution_stats(candidate, initial_occupied)
            delta = candidate_scalar - current_scalar
            accept = delta <= 0
            if not accept and delta < 10**11:
                accept = rng.random() < math.exp(-delta / max(temperature, 1.0))

            accepted_gain = max(0.0, current_scalar - candidate_scalar) / max(1.0, current_scalar)
            best_gain = max(0.0, best_scalar - candidate_scalar) / max(1.0, best_scalar)
            rb_gain = max(0, current_stats["rb_used"] - candidate_stats["rb_used"])
            reward = 0.1 + 40.0 * accepted_gain + 80.0 * best_gain + 0.7 * rb_gain

            if accept:
                current_answers = candidate
                current_scalar = candidate_scalar

            candidate_key = objective_key(candidate, objective, initial_occupied)
            if candidate_key < best_key:
                best_answers = candidate
                best_scalar = candidate_scalar
                best_key = candidate_key
                best_detail = f"ucb:{operator}/{repair_mode}/{size_name}@round={round_id}"
                reward += 2.0

        update_ucb(operator_stats, operator_key, reward)
        update_ucb(size_stats, size_name, reward)
        total_operator_pulls += 1
        total_size_pulls += 1
        temperature *= 0.90

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def ucb_polish_alns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    polish_rounds=12,
    repair_attempts=3,
    seed=15100,
):
    best_answers, _, detail = annealed_alns_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    current_answers = [answer.copy() for answer in best_answers]
    current_scalar = objective_scalar(current_answers, objective, initial_occupied)
    best_scalar = current_scalar
    best_key = objective_key(best_answers, objective, initial_occupied)
    best_detail = f"annealed:{detail}"

    operator_priors = {
        ("related-time", "regret"): (3, 1.10),
        ("evacuation-related", "regret"): (3, 1.00),
        ("light-rb", "regret"): (2, 0.75),
        ("crowded-rb", "regret"): (2, 0.65),
    }
    size_priors = {
        "base": (3, 1.00, 1.0),
        "large": (2, 0.85, 1.25),
    }
    operator_stats = {
        key: {"count": count, "mean": mean}
        for key, (count, mean) in operator_priors.items()
    }
    size_stats = {
        key: {"count": count, "mean": mean}
        for key, (count, mean, _) in size_priors.items()
    }
    size_multiplier = {key: mult for key, (_, _, mult) in size_priors.items()}
    total_operator_pulls = sum(state["count"] for state in operator_stats.values())
    total_size_pulls = sum(state["count"] for state in size_stats.values())
    rng = random.Random(seed)
    base_destroy = max(8, min(32, len(users) // 12))
    temperature = max(50.0, current_scalar * 0.006)

    for round_id in range(polish_rounds):
        operator_key = ucb_choice(
            operator_stats,
            total_operator_pulls,
            rng,
            exploration=0.55,
            epsilon=0.04,
        )
        size_name = ucb_choice(
            size_stats,
            total_size_pulls,
            rng,
            exploration=0.35,
            epsilon=0.04,
        )
        operator, repair_mode = operator_key
        destroy_size = max(
            6,
            min(len(users), int(round(base_destroy * size_multiplier[size_name]))),
        )
        removed_indices = alns_destroy_indices(
            current_answers,
            operator=operator,
            destroy_size=destroy_size,
            seed=seed + round_id,
        )
        if not removed_indices:
            reward = 0.0
        else:
            scoring = "adaptive" if operator in {"light-rb", "evacuation-related"} else "marginal"
            candidate, _ = repair_with_mode(
                mode=repair_mode,
                users=users,
                answers=current_answers,
                removed_indices=removed_indices,
                dl=dl,
                ul=ul,
                base_penalty=95 if scoring == "adaptive" else 60,
                repair_attempts=repair_attempts,
                seed=seed + 1000 + round_id,
                scoring=scoring,
                objective=objective,
                initial_occupied=initial_occupied,
            )
            if validate_answers(users, candidate, dl, ul, initial_occupied):
                reward = 0.0
            else:
                candidate_scalar = objective_scalar(candidate, objective, initial_occupied)
                delta = candidate_scalar - current_scalar
                accept = delta <= 0
                if not accept and delta < 10**11:
                    accept = rng.random() < math.exp(-delta / max(temperature, 1.0))

                reward = 0.05
                if accept:
                    reward += max(0.0, current_scalar - candidate_scalar) / max(1.0, current_scalar) * 50
                    current_answers = candidate
                    current_scalar = candidate_scalar

                candidate_key = objective_key(candidate, objective, initial_occupied)
                if candidate_key < best_key:
                    reward += 3.0 + max(0.0, best_scalar - candidate_scalar) / max(1.0, best_scalar) * 80
                    best_answers = candidate
                    best_scalar = candidate_scalar
                    best_key = candidate_key
                    best_detail = f"ucb-polish:{operator}/{repair_mode}/{size_name}@round={round_id}"

        update_ucb(operator_stats, operator_key, reward)
        update_ucb(size_stats, size_name, reward)
        total_operator_pulls += 1
        total_size_pulls += 1
        temperature *= 0.88

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def contextual_bandit_context(
    answers,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    no_improve_rounds=0,
    last_gain_type="none",
    stats=None,
):
    if stats is None:
        stats = compact_solution_stats(answers, initial_occupied)
    loads = rb_loads(answers)
    active_loads = [load for load in loads.values() if load > 0]
    if not active_loads:
        return "quality-polish"

    rb_capacity = max(1, popcount(uplink_mask(dl, ul)))
    rb_lower_bound = max(1, math.ceil(stats["occupied_pairs"] / rb_capacity))
    avg_load = sum(active_loads) / len(active_loads)
    max_load = max(active_loads)
    light_rbs = sum(1 for load in active_loads if load <= max(1.0, 0.35 * avg_load))
    rb_slack = stats["rb_used"] - rb_lower_bound
    imbalance = max_load / max(1.0, avg_load)

    if no_improve_rounds >= 5:
        return "plateau"
    if last_gain_type == "rb" and no_improve_rounds <= 2:
        return "quality-polish"
    if objective == "lexicographic" and rb_slack > 0:
        return "rb-compression"
    if rb_slack >= 2 or (rb_slack >= 1 and light_rbs > 0):
        return "rb-compression"
    if imbalance >= 1.55:
        return "crowding"
    return "quality-polish"


def contextual_bandit_priors():
    return {
        "rb-compression": {
            ("evacuation-related", "regret", "large"): (4, 1.40, 1.35),
            ("light-rb", "regret", "large"): (3, 1.10, 1.30),
            ("conflict-graph", "conflict", "large"): (3, 1.15, 1.25),
            ("conflict-graph", "regret", "large"): (2, 0.90, 1.25),
            ("crowded-rb", "regret", "base"): (2, 0.75, 1.00),
            ("blockers", "regret", "large"): (2, 0.70, 1.25),
        },
        "crowding": {
            ("crowded-rb", "regret", "base"): (4, 1.20, 1.00),
            ("conflict-graph", "conflict", "base"): (3, 1.15, 1.00),
            ("conflict-graph", "regret", "base"): (3, 1.05, 1.00),
            ("blockers", "regret", "base"): (3, 0.95, 1.00),
            ("related-time", "regret", "base"): (2, 0.75, 1.00),
            ("evacuation-related", "regret", "large"): (2, 0.65, 1.30),
        },
        "quality-polish": {
            ("related-time", "regret", "base"): (4, 1.25, 1.00),
            ("worst-quality", "regret", "base"): (3, 1.00, 1.00),
            ("conflict-graph", "conflict", "base"): (2, 0.80, 1.00),
            ("conflict-graph", "regret", "base"): (2, 0.70, 1.00),
            ("related-time", "random", "base"): (2, 0.65, 1.00),
            ("crowded-rb", "regret", "base"): (2, 0.55, 1.00),
        },
        "plateau": {
            ("evacuation-related", "regret", "xlarge"): (3, 1.20, 1.65),
            ("conflict-graph", "conflict", "large"): (3, 1.20, 1.35),
            ("conflict-graph", "regret", "large"): (3, 1.10, 1.35),
            ("related-time", "regret", "large"): (3, 1.05, 1.35),
            ("random", "regret", "large"): (2, 0.75, 1.35),
            ("blockers", "regret", "large"): (2, 0.70, 1.35),
        },
    }


def instance_conditioned_priors(priors, users, dl, ul, initial_occupied=None):
    priors = {
        context: dict(context_priors)
        for context, context_priors in priors.items()
    }
    option_counts = [
        sum(1 for _ in time_options_for_user(user, dl, ul))
        for user in users
    ]
    avg_options = sum(option_counts) / max(1, len(option_counts))
    rb_capacity = max(1, popcount(uplink_mask(dl, ul)))
    min_pairs = 0
    for user in users:
        best = min(
            (popcount(mask) for _, _, mask, _, _ in time_options_for_user(user, dl, ul)),
            default=1,
        )
        min_pairs += best
    occupied_pairs = 0 if initial_occupied is None else sum(popcount(mask) for mask in initial_occupied)
    rb_lower_bound = max(1, math.ceil((min_pairs + occupied_pairs) / rb_capacity))
    pressure = rb_lower_bound / max(1, R)

    def bump(context, arm, count_delta=0, mean_delta=0.0):
        if arm not in priors[context]:
            return
        count, mean, multiplier = priors[context][arm]
        priors[context][arm] = (count + count_delta, mean + mean_delta, multiplier)

    if len(users) >= 300 or pressure > 0.35:
        bump("rb-compression", ("evacuation-related", "regret", "large"), 1, 0.15)
        bump("rb-compression", ("conflict-graph", "conflict", "large"), 1, 0.20)
        bump("rb-compression", ("conflict-graph", "regret", "large"), 1, 0.20)
        bump("plateau", ("conflict-graph", "conflict", "large"), 1, 0.15)
        bump("plateau", ("conflict-graph", "regret", "large"), 1, 0.15)
    if avg_options > 80:
        bump("quality-polish", ("related-time", "regret", "base"), 1, 0.10)
        bump("crowding", ("conflict-graph", "conflict", "base"), 1, 0.10)
        bump("crowding", ("conflict-graph", "regret", "base"), 1, 0.10)
    return priors


def make_contextual_stats(priors, users=None, dl=None, ul=None, initial_occupied=None):
    if users is not None and dl is not None and ul is not None:
        priors = instance_conditioned_priors(priors, users, dl, ul, initial_occupied)
    context_stats = {
        context: {
            arm: {"count": count, "mean": mean, "multiplier": multiplier}
            for arm, (count, mean, multiplier) in context_priors.items()
        }
        for context, context_priors in priors.items()
    }
    context_pulls = {
        context: sum(state["count"] for state in stats.values())
        for context, stats in context_stats.items()
    }
    return context_stats, context_pulls


def thompson_context_choice(stats, rng, exploration=0.6, epsilon=0.03):
    if rng.random() < epsilon:
        return rng.choice(list(stats))
    best_key = None
    best_score = None
    for key, state in stats.items():
        count = max(1, state["count"])
        score = state["mean"] + rng.gauss(0.0, exploration / math.sqrt(count))
        if best_score is None or score > best_score:
            best_key = key
            best_score = score
    return best_key


def contextual_bandit_alns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    polish_rounds=18,
    repair_attempts=3,
    seed=17200,
):
    best_answers, _, detail = annealed_alns_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    current_answers = [answer.copy() for answer in best_answers]
    current_scalar = objective_scalar(current_answers, objective, initial_occupied)
    current_stats = solution_stats(current_answers, initial_occupied)
    best_scalar = current_scalar
    best_key = objective_key(best_answers, objective, initial_occupied)
    best_stats = current_stats
    best_detail = f"annealed:{detail}"

    # Separate bandit memory by schedule state. The same move can be excellent
    # while evacuating an RB and poor while only polishing CSI quality.
    context_stats, context_pulls = make_contextual_stats(
        contextual_bandit_priors(), users, dl, ul, initial_occupied
    )

    rng = random.Random(seed)
    base_destroy = max(8, min(32, len(users) // 12))
    temperature = max(50.0, current_scalar * 0.006)
    no_improve_rounds = 0
    last_gain_type = "none"

    for round_id in range(polish_rounds):
        context = contextual_bandit_context(
            current_answers,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
            no_improve_rounds=no_improve_rounds,
            last_gain_type=last_gain_type,
            stats=current_stats,
        )
        epsilon = max(0.02, 0.10 * (1 - round_id / max(1, polish_rounds)))
        arm = ucb_choice(
            context_stats[context],
            context_pulls[context],
            rng,
            exploration=0.50 if context != "plateau" else 0.70,
            epsilon=epsilon,
        )
        operator, repair_mode, size_name = arm
        multiplier = context_stats[context][arm]["multiplier"]
        jitter = rng.choice([-2, 0, 2]) if context == "plateau" else 0
        destroy_size = max(
            6,
            min(len(users), int(round(base_destroy * multiplier)) + jitter),
        )
        removed_indices = alns_destroy_indices(
            current_answers,
            operator=operator,
            destroy_size=destroy_size,
            seed=seed + round_id,
        )
        if not removed_indices:
            reward = 0.0
        else:
            scoring = "adaptive" if operator in {"light-rb", "blockers", "evacuation-related", "conflict-graph"} else "marginal"
            candidate, _ = repair_with_mode(
                mode=repair_mode,
                users=users,
                answers=current_answers,
                removed_indices=removed_indices,
                dl=dl,
                ul=ul,
                base_penalty=100 if scoring == "adaptive" else 60,
                repair_attempts=repair_attempts,
                seed=seed + 1000 + round_id,
                scoring=scoring,
                objective=objective,
                initial_occupied=initial_occupied,
            )
            if validate_answers(users, candidate, dl, ul, initial_occupied):
                reward = 0.0
            else:
                candidate_scalar = objective_scalar(candidate, objective, initial_occupied)
                candidate_stats = solution_stats(candidate, initial_occupied)
                delta = candidate_scalar - current_scalar
                accept = delta <= 0
                if not accept and delta < 10**11:
                    accept = rng.random() < math.exp(-delta / max(temperature, 1.0))

                objective_gain = max(0.0, current_scalar - candidate_scalar) / max(1.0, current_scalar)
                best_gain = max(0.0, best_scalar - candidate_scalar) / max(1.0, best_scalar)
                rb_gain = max(0, current_stats["rb_used"] - candidate_stats["rb_used"])
                quality_gain = max(0, current_stats["quality_sum"] - candidate_stats["quality_sum"])
                reward = 0.03 + 45.0 * objective_gain + 90.0 * best_gain
                reward += 0.9 * rb_gain + min(1.5, quality_gain / 120.0)

                if accept:
                    current_answers = candidate
                    current_scalar = candidate_scalar
                    current_stats = candidate_stats

                candidate_key = objective_key(candidate, objective, initial_occupied)
                if candidate_key < best_key:
                    old_best_stats = best_stats
                    best_answers = candidate
                    best_scalar = candidate_scalar
                    best_key = candidate_key
                    best_stats = candidate_stats
                    best_detail = (
                        f"contextual-bandit:{context}:{operator}/{repair_mode}/"
                        f"{size_name}@round={round_id}"
                    )
                    no_improve_rounds = 0
                    if candidate_stats["rb_used"] < old_best_stats["rb_used"]:
                        last_gain_type = "rb"
                    elif candidate_stats["quality_sum"] < old_best_stats["quality_sum"]:
                        last_gain_type = "quality"
                    else:
                        last_gain_type = "objective"
                    reward += 2.5
                else:
                    no_improve_rounds += 1

        update_ucb(context_stats[context], arm, reward)
        context_pulls[context] += 1
        temperature *= 0.88

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def contextual_ts_bandit_alns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    polish_rounds=14,
    repair_attempts=2,
    seed=19300,
):
    best_answers, _, detail = annealed_alns_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    current_answers = [answer.copy() for answer in best_answers]
    current_stats = compact_solution_stats(current_answers, initial_occupied)
    current_scalar = objective_scalar_from_stats(current_stats, objective)
    best_stats = current_stats
    best_scalar = current_scalar
    best_key = objective_key_from_stats(best_stats, objective)
    best_detail = f"annealed:{detail}"

    context_stats, context_pulls = make_contextual_stats(
        contextual_bandit_priors(), users, dl, ul, initial_occupied
    )
    rng = random.Random(seed)
    base_destroy = max(8, min(30, len(users) // 13))
    temperature = max(50.0, current_scalar * 0.005)
    no_improve_rounds = 0
    last_gain_type = "none"

    for round_id in range(polish_rounds):
        if round_id >= 8 and no_improve_rounds >= 7:
            break

        context = contextual_bandit_context(
            current_answers,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
            no_improve_rounds=no_improve_rounds,
            last_gain_type=last_gain_type,
            stats=current_stats,
        )
        arm = thompson_context_choice(
            context_stats[context],
            rng,
            exploration=0.55 if context != "plateau" else 0.80,
            epsilon=max(0.015, 0.07 * (1 - round_id / max(1, polish_rounds))),
        )
        operator, repair_mode, size_name = arm
        multiplier = context_stats[context][arm]["multiplier"]
        if context == "quality-polish":
            multiplier = min(multiplier, 1.0)
        elif context == "plateau":
            multiplier *= 1.1
        destroy_size = max(
            6,
            min(len(users), int(round(base_destroy * multiplier)) + rng.choice([-1, 0, 1])),
        )

        removed_indices = alns_destroy_indices(
            current_answers,
            operator=operator,
            destroy_size=destroy_size,
            seed=seed + round_id,
        )
        if not removed_indices:
            reward = 0.0
            no_improve_rounds += 1
        else:
            scoring = "adaptive" if operator in {"light-rb", "blockers", "evacuation-related", "conflict-graph"} else "marginal"
            candidate, _ = repair_with_mode(
                mode=repair_mode,
                users=users,
                answers=current_answers,
                removed_indices=removed_indices,
                dl=dl,
                ul=ul,
                base_penalty=105 if scoring == "adaptive" else 60,
                repair_attempts=repair_attempts,
                seed=seed + 1000 + round_id,
                scoring=scoring,
                objective=objective,
                initial_occupied=initial_occupied,
            )
            candidate_stats = compact_solution_stats(candidate, initial_occupied)
            candidate_scalar = objective_scalar_from_stats(candidate_stats, objective)
            delta = candidate_scalar - current_scalar
            accept = delta <= 0
            if not accept and delta < 10**11:
                accept = rng.random() < math.exp(-delta / max(temperature, 1.0))

            objective_gain = max(0.0, current_scalar - candidate_scalar) / max(1.0, current_scalar)
            best_gain = max(0.0, best_scalar - candidate_scalar) / max(1.0, best_scalar)
            rb_gain = max(0, current_stats["rb_used"] - candidate_stats["rb_used"])
            quality_gain = max(0, current_stats["quality_sum"] - candidate_stats["quality_sum"])
            rb_penalty = max(0, candidate_stats["rb_used"] - current_stats["rb_used"])
            reward = 0.02 + 55.0 * objective_gain + 100.0 * best_gain
            reward += 1.1 * rb_gain + min(1.6, quality_gain / 100.0)
            if objective == "lexicographic":
                reward -= 0.8 * rb_penalty
            else:
                reward -= 0.25 * rb_penalty

            if accept:
                current_answers = candidate
                current_stats = candidate_stats
                current_scalar = candidate_scalar

            candidate_key = objective_key_from_stats(candidate_stats, objective)
            if candidate_key < best_key:
                old_best_stats = best_stats
                best_answers = candidate
                best_stats = candidate_stats
                best_scalar = candidate_scalar
                best_key = candidate_key
                best_detail = (
                    f"contextual-ts:{context}:{operator}/{repair_mode}/"
                    f"{size_name}@round={round_id}"
                )
                no_improve_rounds = 0
                if candidate_stats["rb_used"] < old_best_stats["rb_used"]:
                    last_gain_type = "rb"
                elif candidate_stats["quality_sum"] < old_best_stats["quality_sum"]:
                    last_gain_type = "quality"
                else:
                    last_gain_type = "objective"
                reward += 2.5
            else:
                no_improve_rounds += 1

        update_ucb(context_stats[context], arm, reward)
        context_pulls[context] += 1
        temperature *= 0.88

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def cpsat_guided_contextual_ts_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    rounds=2,
    time_limit_s=0.3,
    seed=27100,
):
    best_answers, _, detail = contextual_ts_bandit_alns_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    best_key = objective_key(best_answers, objective, initial_occupied)
    best_detail = f"contextual-ts:{detail}"
    rng = random.Random(seed)
    base_repair_size = max(12, min(52, len(users) // 7))
    operators = [
        "evacuation-related",
        "conflict-graph",
        "related-time",
        "light-rb",
    ]

    for round_id in range(rounds):
        current_stats = solution_stats(best_answers, initial_occupied)
        current_rb = current_stats["rb_used"]
        operator = operators[round_id % len(operators)]
        repair_size = min(
            len(users),
            int(round(base_repair_size * (1.0 + 0.25 * (round_id // len(operators))))),
        )
        selected_indices = alns_destroy_indices(
            best_answers,
            operator=operator,
            destroy_size=repair_size,
            seed=seed + round_id,
        )
        if not selected_indices:
            continue
        repair_user_ids = [best_answers[i]["user_id"] for i in selected_indices]
        if operator in {"evacuation-related", "light-rb"}:
            rb_limits = [max(1, current_rb - 1), current_rb]
        else:
            rb_limits = [current_rb]
        if round_id >= len(operators) and current_rb > 1:
            rb_limits = [max(1, current_rb - 1)] + [rb for rb in rb_limits if rb != current_rb - 1]

        for rb_limit in rb_limits:
            try:
                candidate, status = cpsat_repair_subset(
                    users=users,
                    answers=best_answers,
                    repair_user_ids=repair_user_ids,
                    dl=dl,
                    ul=ul,
                    rb_limit=rb_limit,
                    initial_occupied=initial_occupied,
                    time_limit_s=time_limit_s,
                    num_workers=8,
                )
            except RuntimeError:
                return best_answers, objective_value(best_answers, initial_occupied), (
                    best_detail + ";cpsat-unavailable"
                )
            if candidate is None or validate_answers(users, candidate, dl, ul, initial_occupied):
                continue
            candidate_key = objective_key(candidate, objective, initial_occupied)
            if candidate_key < best_key:
                best_answers = candidate
                best_key = candidate_key
                best_detail = f"cpsat-guided:{operator}@round={round_id},K={rb_limit},{status}"
                break

        # A small diversification nudge after a failed exact repair attempt.
        if "cpsat-guided" not in best_detail and round_id % 2 == 1:
            rng.shuffle(operators)

    return best_answers, objective_value(best_answers, initial_occupied), best_detail


def output_to_answers(users, output):
    answers = []
    for user_id, (rb, period, offset) in enumerate(output):
        if period == UNSCHEDULED_PERIOD:
            distance = UNSCHEDULED_PERIOD
        else:
            distance = distance_to_multiple_srs(
                period,
                offset,
                users[user_id]["srs_period"],
                tuple(users[user_id]["srs_offsets"]),
            )
        answers.append(make_answer(user_id, rb, period, offset, distance))
    return answers


def cpsat_assignment_options(users, dl, ul, initial_occupied=None):
    initial_occupied = clone_occupied(initial_occupied)
    options_by_user = []

    for user_id, user in enumerate(users):
        options = []
        for period, offset, mask, distance, quality in time_options_for_user(user, dl, ul):
            slots = periodic_slots(period, offset)
            for rb in range(R):
                if initial_occupied[rb] & mask:
                    continue
                options.append({
                    "user_id": user_id,
                    "rb": rb,
                    "period": period,
                    "offset": offset,
                    "mask": mask,
                    "slots": slots,
                    "distance": distance,
                    "quality": quality,
                })
        options_by_user.append(options)

    return options_by_user


def cpsat_allowed_rbs(rb_limit, initial_occupied):
    initial_used = {rb for rb, mask in enumerate(initial_occupied) if mask}
    if not initial_used:
        return set(range(min(rb_limit, R)))
    return set(range(R))


def cpsat_solve_fixed_rb_budget(
    users,
    dl,
    ul,
    rb_limit,
    initial_occupied=None,
    time_limit_s=5.0,
    num_workers=8,
    allow_unscheduled=True,
):
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("OR-Tools is not installed. Run: python3 -m pip install ortools") from exc

    initial_occupied = clone_occupied(initial_occupied)
    initial_used = {rb for rb, mask in enumerate(initial_occupied) if mask}
    if rb_limit < len(initial_used):
        return None
    allowed_rbs = cpsat_allowed_rbs(rb_limit, initial_occupied)

    model = cp_model.CpModel()
    options_by_user = []
    for user_id, user in enumerate(users):
        options = []
        for period, offset, mask, distance, quality in time_options_for_user(user, dl, ul):
            slots = periodic_slots(period, offset)
            for rb in allowed_rbs:
                if initial_occupied[rb] & mask:
                    continue
                options.append({
                    "user_id": user_id,
                    "rb": rb,
                    "period": period,
                    "offset": offset,
                    "mask": mask,
                    "slots": slots,
                    "distance": distance,
                    "quality": quality,
                })
        options_by_user.append(options)
    assignment_vars = []
    user_vars = defaultdict(list)
    collision_vars = defaultdict(list)
    rb_vars = [model.NewBoolVar(f"rb_{rb}") for rb in range(R)]

    for rb in initial_used:
        model.Add(rb_vars[rb] == 1)
    for rb in range(R):
        if rb not in allowed_rbs and rb not in initial_used:
            model.Add(rb_vars[rb] == 0)

    objective_terms = []
    unscheduled_terms = []
    option_lookup = {}

    for user_id, options in enumerate(options_by_user):
        if allow_unscheduled:
            var = model.NewBoolVar(f"x_u{user_id}_unscheduled")
            user_vars[user_id].append(var)
            assignment_vars.append(var)
            option_lookup[var.Index()] = {
                "user_id": user_id,
                "rb": 0,
                "period": UNSCHEDULED_PERIOD,
                "offset": 0,
                "distance": UNSCHEDULED_PERIOD,
                "quality": 2 * UNSCHEDULED_PERIOD,
                "unscheduled": True,
            }
            objective_terms.append((2 * UNSCHEDULED_PERIOD) * var)
            unscheduled_terms.append(var)
            model.AddHint(var, 1)

        for option_id, option in enumerate(options):
            var = model.NewBoolVar(f"x_u{user_id}_o{option_id}")
            assignment_vars.append(var)
            user_vars[user_id].append(var)
            option_lookup[var.Index()] = option
            objective_terms.append(option["quality"] * var)
            model.Add(var <= rb_vars[option["rb"]])

            for slot in option["slots"]:
                collision_vars[(option["rb"], slot)].append(var)

        if not user_vars[user_id]:
            return None
        model.AddExactlyOne(user_vars[user_id])

    for vars_for_cell in collision_vars.values():
        if len(vars_for_cell) > 1:
            model.Add(sum(vars_for_cell) <= 1)

    model.Add(sum(rb_vars) <= rb_limit)
    quality_upper_bound = len(users) * 2 * UNSCHEDULED_PERIOD + 1
    model.Minimize(sum(unscheduled_terms) * quality_upper_bound + sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers

    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "answers": None,
            "status": status_name,
            "objective_bound": None,
            "quality_sum": None,
            "rb_limit": rb_limit,
            "wall_time_s": solver.WallTime(),
        }

    answers = []
    for var in assignment_vars:
        if solver.BooleanValue(var):
            option = option_lookup[var.Index()]
            answers.append(
                make_answer(
                    option["user_id"],
                    option["rb"],
                    option["period"],
                    option["offset"],
                    option["distance"],
                )
            )

    answers.sort(key=lambda x: x["user_id"])
    stats = solution_stats(answers, initial_occupied)
    return {
        "answers": answers,
        "status": status_name,
        "objective_bound": solver.BestObjectiveBound(),
        "quality_sum": stats["quality_sum"],
        "rb_limit": rb_limit,
        "rb_used": stats["rb_used"],
        "product_objective": stats["objective"],
        "wall_time_s": solver.WallTime(),
    }


def cpsat_solve_product_sweep(
    users,
    dl,
    ul,
    initial_occupied=None,
    min_rb=None,
    max_rb=None,
    time_limit_s=5.0,
    num_workers=8,
    allow_unscheduled=True,
):
    initial_occupied = clone_occupied(initial_occupied)
    initial_used = len({rb for rb, mask in enumerate(initial_occupied) if mask})
    min_rb = initial_used if min_rb is None else max(min_rb, initial_used)
    max_rb = R if max_rb is None else min(max_rb, R)

    best = None
    results = []
    for rb_limit in range(min_rb, max_rb + 1):
        result = cpsat_solve_fixed_rb_budget(
            users=users,
            dl=dl,
            ul=ul,
            rb_limit=rb_limit,
            initial_occupied=initial_occupied,
            time_limit_s=time_limit_s,
            num_workers=num_workers,
            allow_unscheduled=allow_unscheduled,
        )
        if result is None:
            continue
        results.append(result)
        if not result["answers"]:
            continue

        key = objective_key(result["answers"], "product", initial_occupied)
        if best is None or key < best["key"]:
            best = dict(result)
            best["key"] = key

    return best, results


def cpsat_repair_subset(
    users,
    answers,
    repair_user_ids,
    dl,
    ul,
    rb_limit,
    initial_occupied=None,
    time_limit_s=2.0,
    num_workers=8,
    forbidden_rbs=None,
):
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("OR-Tools is not installed. Run: python3 -m pip install ortools") from exc

    repair_set = set(repair_user_ids)
    forbidden_rbs = set() if forbidden_rbs is None else set(forbidden_rbs)
    fixed_answers = [a.copy() for a in answers if a["user_id"] not in repair_set]
    fixed_occupied = build_occupied_from_answers(fixed_answers, initial_occupied)
    allowed_rbs = cpsat_allowed_rbs(rb_limit, fixed_occupied)
    allowed_rbs -= forbidden_rbs

    model = cp_model.CpModel()
    assignment_vars = []
    user_vars = defaultdict(list)
    collision_vars = defaultdict(list)
    option_lookup = {}
    objective_terms = []
    unscheduled_terms = []
    rb_vars = [model.NewBoolVar(f"rb_{rb}") for rb in range(R)]

    for rb, mask in enumerate(fixed_occupied):
        if mask:
            model.Add(rb_vars[rb] == 1)
        elif rb not in allowed_rbs:
            model.Add(rb_vars[rb] == 0)
    for rb in forbidden_rbs:
        model.Add(rb_vars[rb] == 0)

    for user_id in sorted(repair_set):
        user = users[user_id]
        option_count = 0
        var = model.NewBoolVar(f"x_u{user_id}_unscheduled")
        user_vars[user_id].append(var)
        assignment_vars.append(var)
        option_lookup[var.Index()] = {
            "user_id": user_id,
            "rb": 0,
            "period": UNSCHEDULED_PERIOD,
            "offset": 0,
            "distance": UNSCHEDULED_PERIOD,
            "quality": 2 * UNSCHEDULED_PERIOD,
        }
        objective_terms.append((2 * UNSCHEDULED_PERIOD) * var)
        unscheduled_terms.append(var)

        for period, offset, mask, distance, quality in time_options_for_user(user, dl, ul):
            if DEFAULT_TIME_OPTION_CAP is not None and option_count >= DEFAULT_TIME_OPTION_CAP:
                break
            slots = periodic_slots(period, offset)
            for rb in allowed_rbs:
                if fixed_occupied[rb] & mask:
                    continue
                option = {
                    "user_id": user_id,
                    "rb": rb,
                    "period": period,
                    "offset": offset,
                    "mask": mask,
                    "slots": slots,
                    "distance": distance,
                    "quality": quality,
                }
                var = model.NewBoolVar(f"x_u{user_id}_o{option_count}_rb{rb}")
                assignment_vars.append(var)
                user_vars[user_id].append(var)
                option_lookup[var.Index()] = option
                objective_terms.append(quality * var)
                model.Add(var <= rb_vars[rb])
                for slot in slots:
                    collision_vars[(rb, slot)].append(var)
            option_count += 1

        model.AddExactlyOne(user_vars[user_id])

    for vars_for_cell in collision_vars.values():
        if len(vars_for_cell) > 1:
            model.Add(sum(vars_for_cell) <= 1)

    model.Add(sum(rb_vars) <= rb_limit)
    quality_upper_bound = len(repair_set) * 2 * UNSCHEDULED_PERIOD + 1
    model.Minimize(sum(unscheduled_terms) * quality_upper_bound + sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, solver.StatusName(status)

    repaired = []
    for var in assignment_vars:
        if solver.BooleanValue(var):
            option = option_lookup[var.Index()]
            repaired.append(
                make_answer(
                    option["user_id"],
                    option["rb"],
                    option["period"],
                    option["offset"],
                    option["distance"],
                )
            )

    merged = fixed_answers + repaired
    merged.sort(key=lambda x: x["user_id"])
    return merged, solver.StatusName(status)


def cpsat_lns_refine_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    repair_size=24,
    rounds=4,
    time_limit_s=1.0,
):
    answers, _, detail = fast_portfolio_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
    )
    best_key = objective_key(answers, objective, initial_occupied)
    best_detail = f"start:{detail}"

    for round_id in range(rounds):
        selected_indices = choose_destroy_indices(
            answers,
            destroy_size=repair_size,
            random_fraction=0.25,
            seed=9000 + round_id,
        )
        repair_user_ids = [answers[i]["user_id"] for i in selected_indices]
        current_rb = solution_stats(answers, initial_occupied)["rb_used"]

        for rb_limit in [max(1, current_rb - 1), current_rb]:
            candidate, status = cpsat_repair_subset(
                users=users,
                answers=answers,
                repair_user_ids=repair_user_ids,
                dl=dl,
                ul=ul,
                rb_limit=rb_limit,
                initial_occupied=initial_occupied,
                time_limit_s=time_limit_s,
            )
            if candidate is None:
                continue
            if validate_answers(users, candidate, dl, ul, initial_occupied):
                continue
            key = objective_key(candidate, objective, initial_occupied)
            if key < best_key:
                answers = candidate
                best_key = key
                best_detail = f"cpsat-lns:{status}@round={round_id},K={rb_limit}"

    return answers, objective_value(answers, initial_occupied), best_detail


def users_on_rb(answers, rb):
    return [
        answer["user_id"]
        for answer in answers
        if answer["period"] != UNSCHEDULED_PERIOD and answer["rb"] == rb
    ]


def blocker_user_ids(answers, excluded_user_ids, limit):
    excluded = set(excluded_user_ids)
    loads = rb_loads(answers)
    ranked = sorted(
        (
            answer
            for answer in answers
            if answer["period"] != UNSCHEDULED_PERIOD and answer["user_id"] not in excluded
        ),
        key=lambda answer: (
            loads.get(answer["rb"], 0),
            answer["quality"],
            -answer["period"],
        ),
        reverse=True,
    )
    return [answer["user_id"] for answer in ranked[:limit]]


def cpsat_rb_eviction_solver(
    users,
    dl,
    ul,
    objective="product",
    initial_occupied=None,
    max_targets=8,
    blocker_counts=(0, 6, 12),
    time_limit_s=1.5,
):
    answers, _, detail = rb_eviction_refine_solver(
        users,
        dl,
        ul,
        objective=objective,
        initial_occupied=initial_occupied,
        max_passes=1,
        max_targets=max_targets,
    )
    best_key = objective_key(answers, objective, initial_occupied)
    best_detail = f"start:{detail}"
    loads = rb_loads(answers)
    targets = [
        rb
        for rb, load in sorted(loads.items(), key=lambda item: (item[1], item[0]))
        if load > 0 and (initial_occupied is None or initial_occupied[rb] == 0)
    ][:max_targets]

    for target_rb in targets:
        current_rb = solution_stats(answers, initial_occupied)["rb_used"]
        if current_rb <= 1:
            break
        target_users = users_on_rb(answers, target_rb)
        if not target_users:
            continue

        for blocker_count in blocker_counts:
            repair_user_ids = target_users + blocker_user_ids(
                answers,
                excluded_user_ids=target_users,
                limit=blocker_count,
            )
            candidate, status = cpsat_repair_subset(
                users=users,
                answers=answers,
                repair_user_ids=repair_user_ids,
                dl=dl,
                ul=ul,
                rb_limit=current_rb - 1,
                initial_occupied=initial_occupied,
                time_limit_s=time_limit_s,
                forbidden_rbs={target_rb},
            )
            if candidate is None:
                continue
            if validate_answers(users, candidate, dl, ul, initial_occupied):
                continue
            key = objective_key(candidate, objective, initial_occupied)
            if key < best_key:
                answers = candidate
                best_key = key
                best_detail = (
                    f"cpsat-evict-rb{target_rb}:{status},"
                    f"users={len(repair_user_ids)},K={current_rb - 1}"
                )
                break

    return answers, objective_value(answers, initial_occupied), best_detail


def generate_random_users(n: int, seed: int = 1, difficulty="medium"):
    random.seed(seed)

    possible_periods = [5, 10, 20, 40, 80, 160, 320]
    users = []

    for _ in range(n):
        if difficulty == "hard":
            start = random.choice([0, 1, 2, 3])
            srs_period = random.choice([20, 40, 80])
            srs_count = random.randint(1, 4)
        elif difficulty == "large":
            start = random.choice([2, 3, 4, 5])
            srs_period = random.choice([40, 80, 160])
            srs_count = random.randint(1, 3)
        else:
            start = random.choice([2, 3, 4])
            srs_period = random.choice([40, 80])
            srs_count = random.randint(1, 3)

        csi_periods = possible_periods[start:]
        srs_offsets = random.sample(range(srs_period), k=min(srs_count, srs_period))

        users.append({
            "csi_periods": csi_periods,
            "srs_period": srs_period,
            "srs_offsets": srs_offsets,
        })

    return users


def occupied_uplink_fraction(occupied, dl, ul):
    total_uplink_pairs = R * sum(1 for slot in range(L) if is_uplink_slot(slot, dl, ul))
    occupied_pairs = 0
    for mask in occupied:
        occupied_pairs += popcount(mask & uplink_mask(dl, ul))
    return occupied_pairs / total_uplink_pairs if total_uplink_pairs else 0


def generate_initial_occupied(dl, ul, target_fraction=0.0, seed=1):
    random.seed(seed)
    occupied = empty_occupied()
    uplink_slots = [slot for slot in range(L) if is_uplink_slot(slot, dl, ul)]
    all_pairs = [(rb, slot) for rb in range(R) for slot in uplink_slots]
    random.shuffle(all_pairs)
    target = int(len(all_pairs) * target_fraction)
    for rb, slot in all_pairs[:target]:
        occupied[rb] |= 1 << slot
    return occupied


def summarize_records(records):
    objectives = [r["objective"] for r in records]
    rbs = [r["rb_used"] for r in records]
    qualities = [r["quality_sum"] for r in records]
    times = [r["time_ms"] for r in records]
    unscheduled = [r["unscheduled"] for r in records]
    return {
        "avg_objective": statistics.mean(objectives),
        "min_objective": min(objectives),
        "max_objective": max(objectives),
        "avg_rbs": statistics.mean(rbs),
        "avg_quality": statistics.mean(qualities),
        "avg_time_ms": statistics.mean(times),
        "avg_unscheduled": statistics.mean(unscheduled),
    }


def run_method(name, users, dl, ul, objective, initial_occupied=None):
    if name == "baseline-greedy":
        answers, _ = greedy_solve(
            users,
            dl,
            ul,
            base_penalty=0,
            scoring="baseline",
            objective=objective,
            initial_occupied=initial_occupied,
        )
    elif name == "adaptive-greedy":
        answers, _ = greedy_solve(
            users,
            dl,
            ul,
            base_penalty=60,
            scoring="adaptive",
            objective=objective,
            initial_occupied=initial_occupied,
        )
    elif name == "marginal-greedy":
        answers, _ = greedy_solve(
            users,
            dl,
            ul,
            scoring="marginal",
            objective=objective,
            initial_occupied=initial_occupied,
        )
    elif name == "regret-marginal":
        answers, _ = regret_greedy_solve(
            users,
            dl,
            ul,
            scoring="marginal",
            objective=objective,
            initial_occupied=initial_occupied,
        )
    elif name == "budget-sweep":
        answers, _, detail = budget_sweep_greedy_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "learned-budget":
        answers, _, detail = learned_budget_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "fast-portfolio":
        answers, _, detail = fast_portfolio_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "compression-refine":
        answers, _, detail = compression_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "rb-eviction":
        answers, _, detail = rb_eviction_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "adaptive-alns":
        answers, _, detail = adaptive_alns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "annealed-alns":
        answers, _, detail = annealed_alns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "bandit-alns":
        answers, _, detail = bandit_alns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "ucb-bandit-alns":
        answers, _, detail = ucb_bandit_alns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "ucb-polish-alns":
        answers, _, detail = ucb_polish_alns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "contextual-bandit-alns":
        answers, _, detail = contextual_bandit_alns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "contextual-ts-bandit-alns":
        answers, _, detail = contextual_ts_bandit_alns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "contextual-ts-cpsat":
        answers, _, detail = cpsat_guided_contextual_ts_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "cpsat-lns":
        answers, _, detail = cpsat_lns_refine_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "cpsat-rb-evict":
        answers, _, detail = cpsat_rb_eviction_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        return answers, detail
    elif name == "hybrid-portfolio":
        output, _, cfg = portfolio_solver(
            users,
            dl,
            ul,
            objective=objective,
            initial_occupied=initial_occupied,
        )
        answers = output_to_answers(users, output)
        return answers, cfg["name"]
    else:
        raise ValueError(f"unknown method: {name}")
    return answers, "-"


def benchmark(
    n_values=None,
    seeds=None,
    difficulty="medium",
    objective="product",
    include_hybrid=False,
    include_budget_sweep=False,
    include_learned=False,
    include_fast_portfolio=False,
    include_compression=False,
    include_rb_eviction=False,
    include_adaptive_alns=False,
    include_annealed_alns=False,
    include_bandit_alns=False,
    include_ucb_bandit_alns=False,
    include_ucb_polish_alns=False,
    include_contextual_bandit_alns=False,
    include_contextual_ts_bandit_alns=False,
    include_contextual_ts_cpsat=False,
    include_cpsat_lns=False,
    include_cpsat_rb_eviction=False,
    occupied_fraction=0.0,
):
    dl, ul = 8, 2
    uplink_pairs = R * sum(1 for slot in range(L) if is_uplink_slot(slot, dl, ul))
    n_values = n_values or [50, 100]
    seeds = seeds or [1, 2, 3]
    methods = [
        "baseline-greedy",
        "adaptive-greedy",
        "marginal-greedy",
        "regret-marginal",
    ]
    if include_budget_sweep:
        methods.append("budget-sweep")
    if include_learned:
        methods.append("learned-budget")
    if include_fast_portfolio:
        methods.append("fast-portfolio")
    if include_compression:
        methods.append("compression-refine")
    if include_rb_eviction:
        methods.append("rb-eviction")
    if include_adaptive_alns:
        methods.append("adaptive-alns")
    if include_annealed_alns:
        methods.append("annealed-alns")
    if include_bandit_alns:
        methods.append("bandit-alns")
    if include_ucb_bandit_alns:
        methods.append("ucb-bandit-alns")
    if include_ucb_polish_alns:
        methods.append("ucb-polish-alns")
    if include_contextual_bandit_alns:
        methods.append("contextual-bandit-alns")
    if include_contextual_ts_bandit_alns:
        methods.append("contextual-ts-bandit-alns")
    if include_contextual_ts_cpsat:
        methods.append("contextual-ts-cpsat")
    if include_cpsat_lns:
        methods.append("cpsat-lns")
    if include_cpsat_rb_eviction:
        methods.append("cpsat-rb-evict")
    if include_hybrid:
        methods.append("hybrid-portfolio")

    for n in n_values:
        print(
            f"\nN={n}, difficulty={difficulty}, objective={objective}, "
            f"initial_occupied={occupied_fraction:.0%}",
            flush=True,
        )
        by_method = {method: [] for method in methods}

        for seed in seeds:
            users = generate_random_users(n=n, seed=seed, difficulty=difficulty)
            initial_occupied = generate_initial_occupied(
                dl,
                ul,
                target_fraction=occupied_fraction,
                seed=100000 + seed,
            )
            for method in methods:
                start = time.perf_counter()
                answers, detail = run_method(method, users, dl, ul, objective, initial_occupied)
                elapsed_ms = (time.perf_counter() - start) * 1000
                errors = validate_answers(users, answers, dl, ul, initial_occupied)
                stats = solution_stats(answers, initial_occupied)
                stats["time_ms"] = elapsed_ms
                stats["detail"] = detail
                stats["errors"] = len(errors)
                by_method[method].append(stats)

                status = "ok" if not errors else f"{len(errors)} errors"
                print(
                    f"  seed={seed:2d} {method:18s} "
                    f"obj={stats['objective']:8d} rb={stats['rb_used']:2d} "
                    f"q={stats['quality_sum']:5d} uns={stats['unscheduled']:2d} "
                    f"occ={stats['occupied_pairs'] / uplink_pairs:6.1%} "
                    f"time={elapsed_ms:8.1f} ms {status}"
                    ,
                    flush=True,
                )

        print("\n  Summary", flush=True)
        for method in methods:
            summary = summarize_records(by_method[method])
            print(
                f"  {method:18s} "
                f"avg_obj={summary['avg_objective']:9.1f} "
                f"avg_rb={summary['avg_rbs']:5.2f} "
                f"avg_q={summary['avg_quality']:7.1f} "
                f"avg_uns={summary['avg_unscheduled']:5.1f} "
                f"avg_time={summary['avg_time_ms']:8.1f} ms"
                ,
                flush=True,
            )


def exact_branch_and_bound(users, dl, ul, objective="product", max_nodes=250000):
    indexed = list(enumerate(users))

    feasible = {}
    for user_id, user in indexed:
        options = []
        for period in user["csi_periods"]:
            for offset in range(period):
                mask = periodic_mask(period, offset)
                if mask & ~uplink_mask(dl, ul):
                    continue
                distance = distance_to_multiple_srs(
                    period,
                    offset,
                    user["srs_period"],
                    tuple(user["srs_offsets"]),
                )
                options.append({
                    "period": period,
                    "offset": offset,
                    "mask": mask,
                    "distance": distance,
                    "quality": period + distance,
                })
        feasible[user_id] = sorted(options, key=lambda x: (x["quality"], x["period"], x["offset"]))

    order = sorted(indexed, key=lambda item: len(feasible[item[0]]))
    min_suffix = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        user_id, _ = order[i]
        min_suffix[i] = min_suffix[i + 1] + feasible[user_id][0]["quality"]

    best_answers = None
    best_key = None
    nodes = 0

    def search(pos, occupied, answers, quality_sum):
        nonlocal best_answers, best_key, nodes
        nodes += 1
        if nodes > max_nodes:
            return

        used_count = sum(1 for mask in occupied if mask)
        optimistic_rb = max(1, used_count)
        optimistic_quality = quality_sum + min_suffix[pos]
        optimistic_key = (
            0,
            optimistic_rb,
            optimistic_quality,
        ) if objective == "lexicographic" else (
            0,
            optimistic_rb * optimistic_quality,
            optimistic_rb,
            optimistic_quality,
        )
        if best_key is not None and optimistic_key >= best_key:
            return

        if pos == len(order):
            key = objective_key(answers, objective)
            if best_key is None or key < best_key:
                best_key = key
                best_answers = [a.copy() for a in answers]
            return

        user_id, _ = order[pos]
        current_used = sum(1 for mask in occupied if mask)
        rb_choices = list(range(current_used))
        if current_used < R:
            rb_choices.append(current_used)

        for option in feasible[user_id]:
            for rb in rb_choices:
                if occupied[rb] & option["mask"]:
                    continue
                occupied[rb] |= option["mask"]
                answers.append(
                    make_answer(
                        user_id,
                        rb,
                        option["period"],
                        option["offset"],
                        option["distance"],
                    )
                )
                search(pos + 1, occupied, answers, quality_sum + option["quality"])
                answers.pop()
                occupied[rb] &= ~option["mask"]

    search(0, empty_occupied(), [], 0)
    exact = nodes <= max_nodes
    return best_answers, objective_value(best_answers) if best_answers else None, exact, nodes


def exact_demo(n=8, seed=1):
    users = generate_random_users(n=n, seed=seed, difficulty="medium")
    start = time.perf_counter()
    exact_answers, exact_score, exact, nodes = exact_branch_and_bound(users, 8, 2)
    exact_ms = (time.perf_counter() - start) * 1000
    heuristic_answers, _ = regret_greedy_solve(users, 8, 2, scoring="marginal")
    heuristic_score = objective_value(heuristic_answers)
    gap = None
    if exact_score:
        gap = (heuristic_score / exact_score - 1) * 100
    print(
        f"exact_n={n} seed={seed} exact={exact} nodes={nodes} "
        f"exact_obj={exact_score} heuristic_obj={heuristic_score} "
        f"gap_pct={gap:.2f} time={exact_ms:.1f} ms"
    )


def cpsat_demo(
    n=8,
    seed=1,
    difficulty="medium",
    time_limit_s=5.0,
    max_rb=None,
    occupied_fraction=0.0,
):
    dl, ul = 8, 2
    users = generate_random_users(n=n, seed=seed, difficulty=difficulty)
    initial_occupied = generate_initial_occupied(
        dl,
        ul,
        target_fraction=occupied_fraction,
        seed=100000 + seed,
    )

    heuristic_answers, heuristic_detail = run_method(
        "hybrid-portfolio" if n <= 25 else "adaptive-greedy",
        users,
        dl,
        ul,
        "product",
        initial_occupied,
    )
    heuristic_stats = solution_stats(heuristic_answers, initial_occupied)

    start = time.perf_counter()
    best, results = cpsat_solve_product_sweep(
        users=users,
        dl=dl,
        ul=ul,
        initial_occupied=initial_occupied,
        max_rb=max_rb,
        time_limit_s=time_limit_s,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if best is None:
        print(
            f"cpsat_n={n} seed={seed} difficulty={difficulty} no feasible solution "
            f"elapsed={elapsed_ms:.1f} ms"
        )
        return

    errors = validate_answers(users, best["answers"], dl, ul, initial_occupied)
    exact = all(result["status"] == "OPTIMAL" for result in results if result.get("answers"))
    cpsat_stats = solution_stats(best["answers"], initial_occupied)
    gap = None
    if cpsat_stats["objective"]:
        gap = (heuristic_stats["objective"] / cpsat_stats["objective"] - 1) * 100
    gap_text = "n/a" if gap is None else f"{gap:.2f}"

    print(
        f"cpsat_n={n} seed={seed} difficulty={difficulty} "
        f"best_status={best['status']} sweep_exact={exact} "
        f"rb={cpsat_stats['rb_used']} q={cpsat_stats['quality_sum']} "
        f"obj={cpsat_stats['objective']} uns={cpsat_stats['unscheduled']} "
        f"validated={'yes' if not errors else 'no'} elapsed={elapsed_ms:.1f} ms"
    )
    print(
        f"heuristic={heuristic_detail} rb={heuristic_stats['rb_used']} "
        f"q={heuristic_stats['quality_sum']} obj={heuristic_stats['objective']} "
        f"uns={heuristic_stats['unscheduled']} gap_pct={gap_text}"
    )
    print("rb_limit,status,rb_used,quality,product,solver_time_ms")
    for result in results:
        if result.get("answers"):
            print(
                f"{result['rb_limit']},{result['status']},{result['rb_used']},"
                f"{result['quality_sum']},{result['product_objective']},"
                f"{result['wall_time_s'] * 1000:.1f}"
            )
        else:
            print(
                f"{result['rb_limit']},{result['status']},,,,,"
                f"{result['wall_time_s'] * 1000:.1f}"
            )


def cpsat_lower_bound_report(
    n=20,
    seed=1,
    difficulty="medium",
    time_limit_s=3.0,
    max_rb=None,
    occupied_fraction=0.0,
):
    dl, ul = 8, 2
    users = generate_random_users(n=n, seed=seed, difficulty=difficulty)
    initial_occupied = generate_initial_occupied(
        dl,
        ul,
        target_fraction=occupied_fraction,
        seed=100000 + seed,
    )
    heuristic_answers, heuristic_detail = run_method(
        "fast-portfolio",
        users,
        dl,
        ul,
        "product",
        initial_occupied,
    )
    heuristic_stats = solution_stats(heuristic_answers, initial_occupied)
    rb_capacity = max(1, popcount(uplink_mask(dl, ul)))
    required_pairs = 0
    for user in users:
        required_pairs += min(
            (popcount(mask) for _, _, mask, _, _ in time_options_for_user(user, dl, ul)),
            default=1,
        )
    initial_pairs = sum(popcount(mask) for mask in initial_occupied)
    capacity_lb = max(1, math.ceil((required_pairs + initial_pairs) / rb_capacity))
    min_rb = max(1, capacity_lb)
    max_rb = heuristic_stats["rb_used"] if max_rb is None else min(max_rb, heuristic_stats["rb_used"])

    print(
        f"cpsat_lower_bound n={n} seed={seed} difficulty={difficulty} "
        f"capacity_lb={capacity_lb} heuristic={heuristic_detail} "
        f"heuristic_rb={heuristic_stats['rb_used']} heuristic_obj={heuristic_stats['objective']}"
    )
    print("rb_limit,status,rb_used,quality,product,solver_time_ms,interpretation")

    certified_infeasible_to = None
    first_feasible = None
    for rb_limit in range(min_rb, max_rb + 1):
        result = cpsat_solve_fixed_rb_budget(
            users=users,
            dl=dl,
            ul=ul,
            rb_limit=rb_limit,
            initial_occupied=initial_occupied,
            time_limit_s=time_limit_s,
            allow_unscheduled=False,
        )
        if result is None:
            continue
        status = result["status"]
        if result.get("answers"):
            first_feasible = first_feasible or rb_limit
            interpretation = "feasible"
            print(
                f"{rb_limit},{status},{result['rb_used']},{result['quality_sum']},"
                f"{result['product_objective']},{result['wall_time_s'] * 1000:.1f},"
                f"{interpretation}"
            )
            break
        interpretation = "certified_infeasible" if status == "INFEASIBLE" else "unknown_or_timeout"
        if status == "INFEASIBLE":
            certified_infeasible_to = rb_limit
        print(
            f"{rb_limit},{status},,,,,{result['wall_time_s'] * 1000:.1f},"
            f"{interpretation}"
        )

    lower = (certified_infeasible_to + 1) if certified_infeasible_to else capacity_lb
    print(
        f"summary certified_min_rb_lower_bound={lower} "
        f"first_feasible_rb={first_feasible if first_feasible is not None else 'not_found'}"
    )


def mckp_quality_lower_bound(users, dl, ul, rb_budget):
    """Optimistic quality bound under total occupied-cell capacity.

    The bound chooses one CSI time mask per user and ignores exact slot/RB
    collisions, so every feasible schedule using rb_budget RBs must have
    quality at least this value.
    """
    uplink_slots = popcount(uplink_mask(dl, ul))
    capacity = rb_budget * uplink_slots
    inf = 10**15
    dp = [inf] * (capacity + 1)
    dp[0] = 0

    for user in users:
        best_by_cells = {}
        for _, _, mask, _, quality in time_options_for_user(user, dl, ul):
            cells = popcount(mask)
            if cells <= capacity:
                best_by_cells[cells] = min(best_by_cells.get(cells, inf), quality)

        next_dp = [inf] * (capacity + 1)
        for used, value in enumerate(dp):
            if value >= inf:
                continue
            for cells, quality in best_by_cells.items():
                new_used = used + cells
                if new_used <= capacity:
                    next_dp[new_used] = min(next_dp[new_used], value + quality)
        dp = next_dp

    lower_bound = min(dp)
    used_cells = min(range(capacity + 1), key=lambda cells: dp[cells])
    return lower_bound, used_cells, capacity


def mckp_lower_bound_report(n=50, seed=1, difficulty="medium", rb_budget=None):
    dl, ul = 8, 2
    users = generate_random_users(n=n, seed=seed, difficulty=difficulty)
    if rb_budget is None:
        rb_budget = math.ceil(n / popcount(uplink_mask(dl, ul)))

    lower_bound, used_cells, capacity = mckp_quality_lower_bound(users, dl, ul, rb_budget)
    print(
        f"mckp_lower_bound n={n} seed={seed} difficulty={difficulty} "
        f"rb_budget={rb_budget} capacity={capacity} used_cells={used_cells} "
        f"quality_lb={lower_bound} product_lb={rb_budget * lower_bound}"
    )


def mckp_lower_bound_jsonl_report(path, rb_budget=None):
    dl, ul = 8, 2
    print("case,rb_budget,quality,quality_lb,gap_q,product,product_lb,gap_product")
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            record = json.loads(line)
            users = record["users"]
            answers = record.get("answers")
            case_id = record.get("case", line_no)
            selected_rb_budget = rb_budget
            stats = None
            if answers:
                stats = solution_stats(answers)
                if selected_rb_budget is None:
                    selected_rb_budget = stats["rb_used"]
            if selected_rb_budget is None:
                selected_rb_budget = math.ceil(len(users) / popcount(uplink_mask(dl, ul)))

            lower_bound, _, _ = mckp_quality_lower_bound(users, dl, ul, selected_rb_budget)
            product_lb = selected_rb_budget * lower_bound
            if stats is None:
                print(f"{case_id},{selected_rb_budget},,{lower_bound},,,{product_lb},")
                continue

            print(
                f"{case_id},{selected_rb_budget},{stats['quality_sum']},{lower_bound},"
                f"{stats['quality_sum'] - lower_bound},{stats['objective']},{product_lb},"
                f"{stats['objective'] - product_lb}"
            )


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def export_instances(path, n_values, seeds, difficulty):
    with open(path, "w", encoding="utf-8") as f:
        for n in n_values:
            for seed in seeds:
                users = generate_random_users(n=n, seed=seed, difficulty=difficulty)
                f.write(json.dumps({
                    "n": n,
                    "seed": seed,
                    "difficulty": difficulty,
                    "users": users,
                }) + "\n")


def main():
    global DEFAULT_TIME_OPTION_CAP
    parser = argparse.ArgumentParser(description="PUCCH CSI scheduling heuristics")
    parser.add_argument("--n", default="50,100", help="comma-separated instance sizes")
    parser.add_argument("--seeds", default="1,2,3", help="comma-separated seeds")
    parser.add_argument("--difficulty", default="medium", choices=["medium", "hard", "large"])
    parser.add_argument("--objective", default="product", choices=["product", "lexicographic"])
    parser.add_argument("--include-hybrid", action="store_true", help="include slower LNS portfolio")
    parser.add_argument("--include-budget-sweep", action="store_true", help="include slower RB-budget greedy sweep")
    parser.add_argument("--include-learned", action="store_true", help="include learned RB-budget helper")
    parser.add_argument("--include-fast-portfolio", action="store_true", help="include tuned fast heuristic portfolio")
    parser.add_argument("--include-compression", action="store_true", help="include slower RB-compression refinement")
    parser.add_argument("--include-rb-eviction", action="store_true", help="include whole-RB evacuation refinement")
    parser.add_argument("--include-adaptive-alns", action="store_true", help="include adaptive multi-operator ALNS refinement")
    parser.add_argument("--include-annealed-alns", action="store_true", help="include annealed ALNS with regret repair")
    parser.add_argument("--include-bandit-alns", action="store_true", help="include bandit-style ALNS over operator, repair, and destroy size")
    parser.add_argument("--include-ucb-bandit-alns", action="store_true", help="include UCB bandit ALNS with priors over operator and destroy size")
    parser.add_argument("--include-ucb-polish-alns", action="store_true", help="include annealed ALNS followed by a short UCB polish")
    parser.add_argument("--include-contextual-bandit-alns", action="store_true", help="include annealed ALNS followed by contextual UCB polish")
    parser.add_argument("--include-contextual-ts-bandit-alns", action="store_true", help="include annealed ALNS followed by faster contextual Thompson polish")
    parser.add_argument("--include-contextual-ts-cpsat", action="store_true", help="include contextual Thompson polish plus CP-SAT guided plateau/RB-compression repair")
    parser.add_argument("--include-cpsat-lns", action="store_true", help="include CP-SAT-guided LNS refinement")
    parser.add_argument("--include-cpsat-rb-eviction", action="store_true", help="include CP-SAT-guided whole-RB eviction")
    parser.add_argument(
        "--occupied-fraction",
        type=float,
        default=0.0,
        help="initial fraction of occupied uplink RB-slot pairs, e.g. 0.1, 0.5, 0.9",
    )
    parser.add_argument("--exact-small", action="store_true", help="run branch-and-bound demo")
    parser.add_argument("--exact-n", type=int, default=8)
    parser.add_argument("--exact-seed", type=int, default=1)
    parser.add_argument("--cpsat-small", action="store_true", help="run CP-SAT RB-budget sweep")
    parser.add_argument("--cpsat-lower-bound", action="store_true", help="report fixed-RB CP-SAT lower-bound evidence")
    parser.add_argument("--mckp-lower-bound", action="store_true", help="report optimistic occupied-cell knapsack quality lower bound")
    parser.add_argument("--mckp-rb-budget", type=int, default=None, help="RB budget for --mckp-lower-bound")
    parser.add_argument("--mckp-input-jsonl", default=None, help="read users or dumped answers from JSONL and report MCKP bounds")
    parser.add_argument("--cpsat-time", type=float, default=5.0, help="CP-SAT time limit per RB budget")
    parser.add_argument("--cpsat-max-rb", type=int, default=None, help="maximum RB budget to sweep")
    parser.add_argument(
        "--time-option-cap",
        type=int,
        default=64,
        help="keep only the best N CSI period/offset candidates per user for faster heuristics",
    )
    parser.add_argument("--export-instances", default=None, help="write generated instances as JSONL")
    args = parser.parse_args()
    DEFAULT_TIME_OPTION_CAP = None if args.time_option_cap <= 0 else args.time_option_cap

    if args.export_instances:
        export_instances(
            args.export_instances,
            n_values=parse_int_list(args.n),
            seeds=parse_int_list(args.seeds),
            difficulty=args.difficulty,
        )
        return

    if args.exact_small:
        exact_demo(n=args.exact_n, seed=args.exact_seed)
        return

    if args.cpsat_small:
        cpsat_demo(
            n=args.exact_n,
            seed=args.exact_seed,
            difficulty=args.difficulty,
            time_limit_s=args.cpsat_time,
            max_rb=args.cpsat_max_rb,
            occupied_fraction=args.occupied_fraction,
        )
        return

    if args.cpsat_lower_bound:
        cpsat_lower_bound_report(
            n=args.exact_n,
            seed=args.exact_seed,
            difficulty=args.difficulty,
            time_limit_s=args.cpsat_time,
            max_rb=args.cpsat_max_rb,
            occupied_fraction=args.occupied_fraction,
        )
        return

    if args.mckp_lower_bound:
        if args.mckp_input_jsonl:
            mckp_lower_bound_jsonl_report(args.mckp_input_jsonl, rb_budget=args.mckp_rb_budget)
        else:
            mckp_lower_bound_report(
                n=args.exact_n,
                seed=args.exact_seed,
                difficulty=args.difficulty,
                rb_budget=args.mckp_rb_budget,
            )
        return

    benchmark(
        n_values=parse_int_list(args.n),
        seeds=parse_int_list(args.seeds),
        difficulty=args.difficulty,
        objective=args.objective,
        include_hybrid=args.include_hybrid,
        include_budget_sweep=args.include_budget_sweep,
        include_learned=args.include_learned,
        include_fast_portfolio=args.include_fast_portfolio,
        include_compression=args.include_compression,
        include_rb_eviction=args.include_rb_eviction,
        include_adaptive_alns=args.include_adaptive_alns,
        include_annealed_alns=args.include_annealed_alns,
        include_bandit_alns=args.include_bandit_alns,
        include_ucb_bandit_alns=args.include_ucb_bandit_alns,
        include_ucb_polish_alns=args.include_ucb_polish_alns,
        include_contextual_bandit_alns=args.include_contextual_bandit_alns,
        include_contextual_ts_bandit_alns=args.include_contextual_ts_bandit_alns,
        include_contextual_ts_cpsat=args.include_contextual_ts_cpsat,
        include_cpsat_lns=args.include_cpsat_lns,
        include_cpsat_rb_eviction=args.include_cpsat_rb_eviction,
        occupied_fraction=args.occupied_fraction,
    )


if __name__ == "__main__":
    main()
