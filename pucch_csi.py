import random
import time

L = 320
R = 58


def is_uplink_slot(slot: int, dl: int, ul: int) -> bool:
    return slot % (dl + ul) >= dl


def periodic_slots(period: int, offset: int) -> list[int]:
    return list(range(offset, L, period))


def distance_between_resources(csi_period, csi_offset, srs_period, srs_offset) -> int:
    csi = periodic_slots(csi_period, csi_offset)
    srs = periodic_slots(srs_period, srs_offset)
    return min(abs(c - s) for c in csi for s in srs)


def distance_to_multiple_srs(csi_period, csi_offset, srs_period, srs_offsets) -> int:
    return min(
        distance_between_resources(csi_period, csi_offset, srs_period, srs_offset)
        for srs_offset in srs_offsets
    )


def can_place_csi(occupied, rb, period, offset, dl, ul) -> bool:
    for slot in periodic_slots(period, offset):
        if not is_uplink_slot(slot, dl, ul):
            return False
        if occupied[rb][slot]:
            return False
    return True


def place_csi(occupied, rb, period, offset) -> None:
    for slot in periodic_slots(period, offset):
        occupied[rb][slot] = True


def remove_csi(occupied, rb, period, offset) -> None:
    if period == 640:
        return
    for slot in periodic_slots(period, offset):
        occupied[rb][slot] = False


def build_occupied_from_answers(answers):
    occupied = [[False for _ in range(L)] for _ in range(R)]
    for a in answers:
        if a["period"] != 640:
            place_csi(occupied, a["rb"], a["period"], a["offset"])
    return occupied


def objective_value(answers) -> int:
    rb_used = len({a["rb"] for a in answers if a["period"] != 640})
    quality_sum = sum(a["quality"] for a in answers)
    return rb_used * quality_sum


def find_best_for_one_user(
    occupied,
    csi_periods,
    srs_period,
    srs_offsets,
    dl,
    ul,
    base_penalty=60,
    quality_weight=1.0,
):
    best_rb = -1
    best_period = 640
    best_offset = 0
    best_distance = 640
    best_score = 10**18

    used_rbs = {rb for rb in range(R) if any(occupied[rb])}
    load_factor = len(used_rbs)

    for period in csi_periods:
        for rb in range(R):
            for offset in range(period):
                if not can_place_csi(occupied, rb, period, offset, dl, ul):
                    continue

                distance = distance_to_multiple_srs(
                    period,
                    offset,
                    srs_period,
                    srs_offsets,
                )

                quality = distance + period
                adaptive_penalty = base_penalty * (1 + load_factor / 5)
                new_rb_penalty = adaptive_penalty if rb not in used_rbs else 0

                score = quality_weight * quality + new_rb_penalty
                candidate = (score, distance, period, rb, offset)
                best = (best_score, best_distance, best_period, best_rb, best_offset)

                if candidate < best:
                    best_score = score
                    best_distance = distance
                    best_period = period
                    best_rb = rb
                    best_offset = offset

    if best_rb == -1:
        return 0, 640, 0, 640

    return best_rb, best_period, best_offset, best_distance


def greedy_solve(users, dl, ul, base_penalty=60, quality_weight=1.0):
    occupied = [[False for _ in range(L)] for _ in range(R)]
    answers = []

    for user_id, user in enumerate(users):
        rb, period, offset, distance = find_best_for_one_user(
            occupied=occupied,
            csi_periods=user["csi_periods"],
            srs_period=user["srs_period"],
            srs_offsets=user["srs_offsets"],
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
        )

        if period != 640:
            place_csi(occupied, rb, period, offset)

        answers.append({
            "user_id": user_id,
            "rb": rb,
            "period": period,
            "offset": offset,
            "distance": distance,
            "quality": distance + period,
        })

    return answers, objective_value(answers)


def multistart_solve(users, dl, ul, base_penalty=60, attempts=50, seed=123, quality_weight=1.0):
    random.seed(seed)

    best_answers = None
    best_score = 10**18
    indexed_users = list(enumerate(users))

    for _ in range(attempts):
        shuffled = indexed_users[:]
        random.shuffle(shuffled)

        shuffled_users = [user for _, user in shuffled]
        original_ids = [original_id for original_id, _ in shuffled]

        answers, score = greedy_solve(
            users=shuffled_users,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            quality_weight=quality_weight,
        )

        restored_answers = []
        for answer, original_id in zip(answers, original_ids):
            fixed = answer.copy()
            fixed["user_id"] = original_id
            restored_answers.append(fixed)

        if score < best_score:
            best_score = score
            best_answers = restored_answers

    return best_answers, best_score


def local_search_one_user_moves(users, answers, dl, ul, max_rounds=2):
    answers = [a.copy() for a in answers]
    best_score = objective_value(answers)

    for _ in range(max_rounds):
        improved = False

        for idx, current_answer in enumerate(answers):
            user_id = current_answer["user_id"]
            user = users[user_id]

            occupied = build_occupied_from_answers(answers)
            remove_csi(
                occupied,
                current_answer["rb"],
                current_answer["period"],
                current_answer["offset"],
            )

            local_best_answer = current_answer.copy()
            local_best_score = best_score

            for period in user["csi_periods"]:
                for rb in range(R):
                    for offset in range(period):
                        if not can_place_csi(occupied, rb, period, offset, dl, ul):
                            continue

                        distance = distance_to_multiple_srs(
                            period,
                            offset,
                            user["srs_period"],
                            user["srs_offsets"],
                        )

                        candidate_answer = {
                            "user_id": user_id,
                            "rb": rb,
                            "period": period,
                            "offset": offset,
                            "distance": distance,
                            "quality": distance + period,
                        }

                        trial_answers = answers[:]
                        trial_answers[idx] = candidate_answer
                        trial_score = objective_value(trial_answers)

                        if trial_score < local_best_score:
                            local_best_score = trial_score
                            local_best_answer = candidate_answer

            if local_best_score < best_score:
                answers[idx] = local_best_answer
                best_score = local_best_score
                improved = True

        if not improved:
            break

    return answers, best_score


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
):
    random.seed(seed)

    base_answers = [a.copy() for a in answers]
    removed_answers = [base_answers[i] for i in removed_indices]

    for i in sorted(removed_indices, reverse=True):
        base_answers.pop(i)

    best_answers = None
    best_score = 10**18

    for _ in range(attempts):
        trial_answers = [a.copy() for a in base_answers]
        occupied = build_occupied_from_answers(trial_answers)

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
            )

            if period == 640:
                success = False
                break

            place_csi(occupied, rb, period, offset)

            trial_answers.append({
                "user_id": user_id,
                "rb": rb,
                "period": period,
                "offset": offset,
                "distance": distance,
                "quality": distance + period,
            })

        if not success:
            continue

        score = objective_value(trial_answers)

        if score < best_score:
            best_score = score
            best_answers = trial_answers

    if best_answers is None:
        return answers, objective_value(answers)

    return best_answers, best_score


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
):
    random.seed(seed)

    answers = [a.copy() for a in answers]
    best_score = objective_value(answers)

    for round_id in range(rounds):
        n = len(answers)

        worst_count = max(1, int(destroy_size * (1 - random_fraction)))
        random_count = destroy_size - worst_count

        ranked = sorted(
            list(enumerate(answers)),
            key=lambda x: x[1]["quality"],
            reverse=True,
        )

        selected_indices = [idx for idx, _ in ranked[:worst_count]]

        remaining_indices = [
            i for i in range(n)
            if i not in selected_indices
        ]

        if random_count > 0 and remaining_indices:
            selected_indices += random.sample(
                remaining_indices,
                k=min(random_count, len(remaining_indices)),
            )

        candidate_answers, candidate_score = repair_removed_users(
            users=users,
            answers=answers,
            removed_indices=selected_indices,
            dl=dl,
            ul=ul,
            base_penalty=base_penalty,
            attempts=repair_attempts,
            seed=seed + round_id,
            quality_weight=quality_weight,
        )

        if candidate_score < best_score:
            answers = candidate_answers
            best_score = candidate_score

            answers, best_score = local_search_one_user_moves(
                users,
                answers,
                dl,
                ul,
                max_rounds=1,
            )

    return answers, best_score


def answers_to_output(answers):
    answers_sorted = sorted(answers, key=lambda x: x["user_id"])
    return [(a["rb"], a["period"], a["offset"]) for a in answers_sorted]


def final_solver(
    users,
    dl,
    ul,
    base_penalty=60,
    attempts=50,
    local_rounds=2,
    seed=123,
    quality_weight=1.0,
    random_destroy_fraction=0.3,
):
    answers, score = multistart_solve(
        users=users,
        dl=dl,
        ul=ul,
        base_penalty=base_penalty,
        attempts=attempts,
        seed=seed,
        quality_weight=quality_weight,
    )

    answers, score = local_search_one_user_moves(
        users,
        answers,
        dl,
        ul,
        max_rounds=local_rounds,
    )

    answers, score = destroy_repair_search(
        users=users,
        answers=answers,
        dl=dl,
        ul=ul,
        base_penalty=base_penalty,
        destroy_size=10,
        rounds=10,
        repair_attempts=20,
        seed=seed + 777,
        quality_weight=quality_weight,
        random_fraction=random_destroy_fraction,
    )

    return answers_to_output(answers), score


def portfolio_solver(users, dl, ul):
    configs = [
        {"base_penalty": 60, "attempts": 50, "local_rounds": 2, "seed_shift": 1000, "quality_weight": 1.0, "random_destroy_fraction": 0.0},
        {"base_penalty": 40, "attempts": 50, "local_rounds": 2, "seed_shift": 4000, "quality_weight": 1.0, "random_destroy_fraction": 0.3},
        {"base_penalty": 80, "attempts": 50, "local_rounds": 2, "seed_shift": 5000, "quality_weight": 1.5, "random_destroy_fraction": 0.3},
        {"base_penalty": 80, "attempts": 50, "local_rounds": 2, "seed_shift": 6000, "quality_weight": 2.0, "random_destroy_fraction": 0.3},
    ]

    best_output = None
    best_score = 10**18
    best_config = None

    for cfg in configs:
        output, score = final_solver(
            users=users,
            dl=dl,
            ul=ul,
            base_penalty=cfg["base_penalty"],
            attempts=cfg["attempts"],
            local_rounds=cfg["local_rounds"],
            seed=cfg["seed_shift"],
            quality_weight=cfg["quality_weight"],
            random_destroy_fraction=cfg["random_destroy_fraction"],
        )

        if score < best_score:
            best_score = score
            best_output = output
            best_config = cfg

    return best_output, best_score, best_config


def generate_random_users(n: int, seed: int = 1):
    random.seed(seed)

    possible_periods = [5, 10, 20, 40, 80, 160, 320]
    users = []

    for _ in range(n):
        start = random.choice([2, 3, 4])
        csi_periods = possible_periods[start:]

        srs_period = random.choice([40, 80])
        srs_offsets = random.sample(
            range(srs_period),
            k=random.randint(1, 3),
        )

        users.append({
            "csi_periods": csi_periods,
            "srs_period": srs_period,
            "srs_offsets": srs_offsets,
        })

    return users


def benchmark():
    dl, ul = 8, 2
    scores = []
    times = []

    for seed in range(1, 6):
        users = generate_random_users(n=100, seed=seed)

        start = time.perf_counter()
        output, score, cfg = portfolio_solver(users, dl, ul)
        elapsed = time.perf_counter() - start

        rb_used = len({rb for rb, period, offset in output if period != 640})

        print(
            f"seed={seed:2d} | "
            f"rb_used={rb_used:2d} | "
            f"objective={score} | "
            f"time={elapsed * 1000:.2f} ms | "
            f"cfg={cfg}"
        )

        scores.append(score)
        times.append(elapsed * 1000)

    print()
    print("Average objective:", sum(scores) / len(scores))
    print("Min objective:", min(scores))
    print("Max objective:", max(scores))
    print("Average time, ms:", sum(times) / len(times))
    print("Min time, ms:", min(times))
    print("Max time, ms:", max(times))


if __name__ == "__main__":
    benchmark()