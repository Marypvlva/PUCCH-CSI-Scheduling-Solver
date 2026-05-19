#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <iostream>
#include <fstream>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <thread>
#include <vector>

namespace {

constexpr int L = 320;
constexpr int R = 58;
constexpr int WORDS = 5;
constexpr int BANDIT_REPAIR_OPTION_LIMIT = 16;

struct Mask {
    std::array<uint64_t, WORDS> w{};
};

struct User {
    std::vector<int> csi_periods;
    int srs_period = 0;
    std::vector<int> srs_offsets;
};

struct TimeOption {
    int period = 0;
    int offset = 0;
    int distance = 0;
    int quality = 0;
    Mask mask;
};

struct BaseTimeOption {
    int period = 0;
    int offset = 0;
    Mask mask;
};

struct Answer {
    int rb = 0;
    int period = 640;
    int offset = 0;
    int distance = 640;
    int quality = 1280;
};

bool is_uplink_slot(int slot, int dl, int ul) {
    return slot % (dl + ul) >= dl;
}

void set_bit(Mask& mask, int slot) {
    mask.w[slot / 64] |= uint64_t{1} << (slot % 64);
}

bool intersects(const Mask& a, const Mask& b) {
    for (int i = 0; i < WORDS; ++i) {
        if (a.w[i] & b.w[i]) return true;
    }
    return false;
}

void add_mask(Mask& a, const Mask& b) {
    for (int i = 0; i < WORDS; ++i) a.w[i] |= b.w[i];
}

void clear_mask(Mask& a) {
    for (uint64_t& word : a.w) word = 0;
}

int popcount(const Mask& a) {
    int total = 0;
    for (uint64_t word : a.w) total += __builtin_popcountll(word);
    return total;
}

int intersection_popcount(const Mask& a, const Mask& b) {
    int total = 0;
    for (int i = 0; i < WORDS; ++i) total += __builtin_popcountll(a.w[i] & b.w[i]);
    return total;
}

bool mask_empty(const Mask& mask) {
    for (uint64_t word : mask.w) {
        if (word) return false;
    }
    return true;
}

Mask periodic_mask(int period, int offset) {
    Mask mask;
    for (int slot = offset; slot < L; slot += period) set_bit(mask, slot);
    return mask;
}

const std::vector<BaseTimeOption>& base_options_for_period(int period, int dl, int ul) {
    static std::array<std::vector<BaseTimeOption>, 321> cache;
    static std::array<uint8_t, 321> ready{};
    if (!ready[period]) {
        for (int offset = 0; offset < period; ++offset) {
            bool uplink = true;
            for (int slot = offset; slot < L; slot += period) {
                if (!is_uplink_slot(slot, dl, ul)) {
                    uplink = false;
                    break;
                }
            }
            if (!uplink) continue;
            BaseTimeOption option;
            option.period = period;
            option.offset = offset;
            option.mask = periodic_mask(period, offset);
            cache[period].push_back(option);
        }
        ready[period] = 1;
    }
    return cache[period];
}

int distance_between_resources(int csi_period, int csi_offset, int srs_period, int srs_offset) {
    if (srs_period >= csi_period) {
        return std::abs(csi_offset - (srs_offset % csi_period));
    }
    int ratio = csi_period / srs_period;
    int best = L * 2;
    for (int i = 0; i < ratio; ++i) {
        int shifted = (srs_offset + i * srs_period) % csi_period;
        best = std::min(best, std::abs(csi_offset - shifted));
    }
    return best;
}

int distance_to_multiple_srs(int csi_period, int csi_offset, int srs_period, const std::vector<int>& srs_offsets) {
    int best = L * 2;
    for (int srs_offset : srs_offsets) {
        best = std::min(best, distance_between_resources(csi_period, csi_offset, srs_period, srs_offset));
    }
    return best;
}

std::vector<TimeOption> build_time_options(const User& user, int dl, int ul, int cap) {
    std::vector<TimeOption> options;
    for (int period : user.csi_periods) {
        for (const BaseTimeOption& base : base_options_for_period(period, dl, ul)) {
            int distance = distance_to_multiple_srs(period, base.offset, user.srs_period, user.srs_offsets);
            TimeOption option;
            option.period = period;
            option.offset = base.offset;
            option.distance = distance;
            option.quality = period + distance;
            option.mask = base.mask;
            options.push_back(option);
        }
    }
    std::sort(options.begin(), options.end(), [](const TimeOption& a, const TimeOption& b) {
        return std::tie(a.quality, a.period, a.offset) < std::tie(b.quality, b.period, b.offset);
    });
    if (cap > 0 && static_cast<int>(options.size()) > cap) options.resize(cap);
    return options;
}

std::vector<int> candidate_rbs(const std::vector<Mask>& occupied, int rb_limit) {
    std::vector<int> rbs;
    for (int rb = 0; rb < R; ++rb) {
        bool used = false;
        for (uint64_t word : occupied[rb].w) {
            if (word) {
                used = true;
                break;
            }
        }
        if (used) rbs.push_back(rb);
    }
    if (rb_limit > 0 && static_cast<int>(rbs.size()) >= rb_limit) return rbs;
    for (int rb = 0; rb < R; ++rb) {
        bool used = false;
        for (uint64_t word : occupied[rb].w) {
            if (word) {
                used = true;
                break;
            }
        }
        if (!used) {
            rbs.push_back(rb);
            break;
        }
    }
    return rbs;
}

int rb_used_count(const std::vector<Mask>& occupied) {
    int count = 0;
    for (const Mask& mask : occupied) {
        for (uint64_t word : mask.w) {
            if (word) {
                ++count;
                break;
            }
        }
    }
    return count;
}

std::vector<std::vector<TimeOption>> prepare_options(const std::vector<User>& users, int dl, int ul, int cap) {
    std::vector<std::vector<TimeOption>> options_by_user;
    options_by_user.reserve(users.size());
    for (const User& user : users) options_by_user.push_back(build_time_options(user, dl, ul, cap));
    return options_by_user;
}

std::vector<Answer> solve_adaptive_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    int rb_limit = 0
) {

    std::vector<Mask> occupied(R);
    std::array<uint8_t, R> active{};
    std::vector<int> active_rbs;
    active_rbs.reserve(R);
    int first_empty = 0;

    std::vector<Answer> answers;
    answers.reserve(options_by_user.size());

    for (size_t user_id = 0; user_id < options_by_user.size(); ++user_id) {
        int used = static_cast<int>(active_rbs.size());

        bool found = false;
        Answer best;
        const TimeOption* best_option = nullptr;
        int64_t best_score_num = std::numeric_limits<int64_t>::max();

        for (const TimeOption& option : options_by_user[user_id]) {
            for (int rb : active_rbs) {
                if (intersects(occupied[rb], option.mask)) continue;
                int64_t score_num = int64_t(option.quality) * 5;
                auto candidate_tuple = std::make_tuple(score_num, option.distance, option.period, rb, option.offset);
                auto best_tuple = std::make_tuple(best_score_num, best.distance, best.period, best.rb, best.offset);
                if (!found || candidate_tuple < best_tuple) {
                    found = true;
                    best_score_num = score_num;
                    best_option = &option;
                    best.rb = rb;
                    best.period = option.period;
                    best.offset = option.offset;
                    best.distance = option.distance;
                    best.quality = option.quality;
                }
            }

            if ((rb_limit == 0 || used < rb_limit) && first_empty < R) {
                int rb = first_empty;
                int64_t penalty_num = int64_t(base_penalty) * (5 + used);
                int64_t score_num = int64_t(option.quality) * 5 + penalty_num;
                auto candidate_tuple = std::make_tuple(score_num, option.distance, option.period, rb, option.offset);
                auto best_tuple = std::make_tuple(best_score_num, best.distance, best.period, best.rb, best.offset);
                if (!found || candidate_tuple < best_tuple) {
                    found = true;
                    best_score_num = score_num;
                    best_option = &option;
                    best.rb = rb;
                    best.period = option.period;
                    best.offset = option.offset;
                    best.distance = option.distance;
                    best.quality = option.quality;
                }
            }
        }

        if (found && best_option != nullptr) {
            if (!active[best.rb]) {
                active[best.rb] = 1;
                active_rbs.push_back(best.rb);
                while (first_empty < R && active[first_empty]) ++first_empty;
            }
            add_mask(occupied[best.rb], best_option->mask);
        }
        answers.push_back(best);
    }
    return answers;
}

std::vector<Answer> solve_adaptive(const std::vector<User>& users, int dl, int ul, int cap, int base_penalty, int rb_limit = 0) {
    auto options_by_user = prepare_options(users, dl, ul, cap);
    return solve_adaptive_prepared(options_by_user, base_penalty, rb_limit);
}

std::tuple<int, int, int> stats(const std::vector<Answer>& answers) {
    std::set<int> used;
    int quality_sum = 0;
    int unscheduled = 0;
    for (const Answer& answer : answers) {
        quality_sum += answer.quality;
        if (answer.period == 640) {
            ++unscheduled;
        } else {
            used.insert(answer.rb);
        }
    }
    return {static_cast<int>(used.size()), quality_sum, unscheduled};
}

int64_t objective_value(const std::vector<Answer>& answers) {
    auto [rb_used, quality_sum, unscheduled] = stats(answers);
    return int64_t(unscheduled) * 1000000000000LL + int64_t(rb_used) * quality_sum;
}

struct SolutionStats {
    int rb_used = 0;
    int quality_sum = 0;
    int unscheduled = 0;
    int occupied_pairs = 0;
    int64_t objective = 0;
};

SolutionStats compact_stats(const std::vector<Answer>& answers) {
    std::array<uint8_t, R> used{};
    SolutionStats result;
    for (const Answer& answer : answers) {
        result.quality_sum += answer.quality;
        if (answer.period == 640) {
            ++result.unscheduled;
            continue;
        }
        used[answer.rb] = 1;
        result.occupied_pairs += L / answer.period;
    }
    for (uint8_t value : used) result.rb_used += value ? 1 : 0;
    result.objective = int64_t(result.unscheduled) * 1000000000000LL
        + int64_t(result.rb_used) * result.quality_sum;
    return result;
}

Mask answer_mask(const Answer& answer) {
    if (answer.period == 640) return Mask{};
    return periodic_mask(answer.period, answer.offset);
}

std::vector<Mask> build_occupied_from_answers(const std::vector<Answer>& answers, const std::vector<int>& skip = {}) {
    std::array<uint8_t, 10000> skipped{};
    for (int idx : skip) {
        if (idx >= 0 && idx < static_cast<int>(skipped.size())) skipped[idx] = 1;
    }
    std::vector<Mask> occupied(R);
    for (int i = 0; i < static_cast<int>(answers.size()); ++i) {
        if (i < static_cast<int>(skipped.size()) && skipped[i]) continue;
        const Answer& answer = answers[i];
        if (answer.period == 640) continue;
        add_mask(occupied[answer.rb], answer_mask(answer));
    }
    return occupied;
}

std::vector<int> active_rbs_from_occupied(const std::vector<Mask>& occupied) {
    std::vector<int> active;
    for (int rb = 0; rb < R; ++rb) {
        if (!mask_empty(occupied[rb])) active.push_back(rb);
    }
    return active;
}

struct Choice {
    bool found = false;
    int64_t score = std::numeric_limits<int64_t>::max();
    int rb = 0;
    int period = 640;
    int offset = 0;
    int distance = 640;
    int quality = 1280;
    const TimeOption* option = nullptr;
};

std::vector<Choice> best_choices_for_user(
    const std::vector<Mask>& occupied,
    const std::vector<TimeOption>& options,
    int base_penalty,
    const std::string& scoring,
    int64_t current_quality_sum,
    int limit,
    int option_limit = 0
) {
    std::vector<int> active_rbs = active_rbs_from_occupied(occupied);
    int used = static_cast<int>(active_rbs.size());
    int first_empty = -1;
    for (int rb = 0; rb < R; ++rb) {
        if (mask_empty(occupied[rb])) {
            first_empty = rb;
            break;
        }
    }

    std::vector<int> rb_candidates = active_rbs;
    if (first_empty >= 0) rb_candidates.push_back(first_empty);

    std::vector<Choice> best;
    best.reserve(limit);
    bool adaptive_scoring = scoring == "adaptive";
    int capped_options = static_cast<int>(options.size());
    if (option_limit > 0) capped_options = std::min(capped_options, option_limit);
    for (int option_idx = 0; option_idx < capped_options; ++option_idx) {
        const TimeOption& option = options[option_idx];
        for (int rb : rb_candidates) {
            if (intersects(occupied[rb], option.mask)) continue;
            bool opens = mask_empty(occupied[rb]);
            int next_used = used + (opens ? 1 : 0);
            int64_t score = 0;
            if (adaptive_scoring) {
                score = int64_t(option.quality) * 5;
                if (opens) score += int64_t(base_penalty) * (5 + used);
            } else {
                int64_t current_obj = int64_t(std::max(1, used)) * current_quality_sum;
                int64_t next_obj = int64_t(next_used) * (current_quality_sum + option.quality);
                score = next_obj - current_obj;
            }

            Choice choice;
            choice.found = true;
            choice.score = score;
            choice.rb = rb;
            choice.period = option.period;
            choice.offset = option.offset;
            choice.distance = option.distance;
            choice.quality = option.quality;
            choice.option = &option;

            auto choice_tuple = std::make_tuple(choice.score, choice.distance, choice.period, choice.rb, choice.offset);
            bool inserted = false;
            for (auto it = best.begin(); it != best.end(); ++it) {
                auto it_tuple = std::make_tuple(it->score, it->distance, it->period, it->rb, it->offset);
                if (choice_tuple < it_tuple) {
                    best.insert(it, choice);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) best.push_back(choice);
            if (static_cast<int>(best.size()) > limit) best.pop_back();
        }
    }
    return best;
}

Answer choice_to_answer(const Choice& choice) {
    Answer answer;
    answer.rb = choice.rb;
    answer.period = choice.period;
    answer.offset = choice.offset;
    answer.distance = choice.distance;
    answer.quality = choice.quality;
    return answer;
}

std::vector<Answer> solve_greedy_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    const std::string& scoring
) {
    std::vector<Mask> occupied(R);
    std::vector<Answer> answers(options_by_user.size());
    int64_t quality_sum = 0;

    for (int user_id = 0; user_id < static_cast<int>(options_by_user.size()); ++user_id) {
        auto choices = best_choices_for_user(
            occupied,
            options_by_user[user_id],
            base_penalty,
            scoring,
            quality_sum,
            1
        );
        if (!choices.empty()) {
            Answer placed = choice_to_answer(choices[0]);
            answers[user_id] = placed;
            add_mask(occupied[placed.rb], choices[0].option->mask);
            quality_sum += placed.quality;
        } else {
            quality_sum += answers[user_id].quality;
        }
    }
    return answers;
}

std::vector<Answer> solve_portfolio_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty
) {
    std::vector<std::vector<Answer>> candidates;
    for (int penalty : {40, 60, 80}) {
        candidates.push_back(solve_greedy_prepared(options_by_user, penalty, "adaptive"));
    }
    candidates.push_back(solve_greedy_prepared(options_by_user, base_penalty, "marginal"));

    int best_idx = 0;
    int64_t best_obj = objective_value(candidates[0]);
    for (int i = 1; i < static_cast<int>(candidates.size()); ++i) {
        int64_t obj = objective_value(candidates[i]);
        if (obj < best_obj) {
            best_obj = obj;
            best_idx = i;
        }
    }
    return candidates[best_idx];
}

int64_t quality_sum_excluding(const std::vector<Answer>& answers, const std::vector<int>& removed) {
    std::vector<uint8_t> skip(answers.size(), 0);
    for (int idx : removed) {
        if (0 <= idx && idx < static_cast<int>(answers.size())) skip[idx] = 1;
    }
    int64_t total = 0;
    for (int i = 0; i < static_cast<int>(answers.size()); ++i) {
        if (!skip[i]) total += answers[i].quality;
    }
    return total;
}

std::vector<Answer> repair_random(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    std::vector<int> removed,
    int base_penalty,
    const std::string& scoring,
    std::mt19937& rng,
    int option_limit = 0
) {
    std::shuffle(removed.begin(), removed.end(), rng);
    std::vector<Answer> trial = answers;
    std::vector<Mask> occupied = build_occupied_from_answers(answers, removed);
    int64_t quality_sum = quality_sum_excluding(answers, removed);

    for (int user_id : removed) {
        auto choices = best_choices_for_user(
            occupied, options_by_user[user_id], base_penalty, scoring, quality_sum, 1, option_limit
        );
        if (choices.empty()) return answers;
        Answer placed = choice_to_answer(choices[0]);
        trial[user_id] = placed;
        add_mask(occupied[placed.rb], choices[0].option->mask);
        quality_sum += placed.quality;
    }
    return trial;
}

std::vector<Answer> repair_regret(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    std::vector<int> removed,
    int base_penalty,
    const std::string& scoring,
    int option_limit = 0
) {
    std::vector<Answer> trial = answers;
    std::vector<Mask> occupied = build_occupied_from_answers(answers, removed);
    int64_t quality_sum = quality_sum_excluding(answers, removed);

    while (!removed.empty()) {
        int selected_pos = -1;
        Choice selected_choice;
        int64_t selected_regret = std::numeric_limits<int64_t>::min();
        int64_t selected_score = 0;

        for (int pos = 0; pos < static_cast<int>(removed.size()); ++pos) {
            int user_id = removed[pos];
            auto choices = best_choices_for_user(
                occupied, options_by_user[user_id], base_penalty, scoring, quality_sum, 2, option_limit
            );
            if (choices.empty()) return answers;
            int64_t regret = choices.size() == 1 ? 1000000000LL : choices[1].score - choices[0].score;
            auto cand_tuple = std::make_tuple(regret, -choices[0].score, answers[user_id].quality);
            auto best_tuple = std::make_tuple(selected_regret, -selected_score, selected_pos < 0 ? -1 : answers[removed[selected_pos]].quality);
            if (selected_pos < 0 || cand_tuple > best_tuple) {
                selected_pos = pos;
                selected_regret = regret;
                selected_score = choices[0].score;
                selected_choice = choices[0];
            }
        }

        int user_id = removed[selected_pos];
        Answer placed = choice_to_answer(selected_choice);
        trial[user_id] = placed;
        add_mask(occupied[placed.rb], selected_choice.option->mask);
        quality_sum += placed.quality;
        removed.erase(removed.begin() + selected_pos);
    }
    return trial;
}

std::vector<Answer> repair_regret_fast(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    const std::vector<int>& removed,
    int base_penalty,
    const std::string& scoring,
    int option_limit = 0
) {
    std::vector<Answer> trial = answers;
    std::vector<Mask> occupied = build_occupied_from_answers(answers, removed);
    int64_t quality_sum = quality_sum_excluding(answers, removed);

    struct RepairOrderItem {
        int user_id = 0;
        int64_t regret = 0;
        int64_t score = 0;
        int old_quality = 0;
    };

    std::vector<RepairOrderItem> order;
    order.reserve(removed.size());
    for (int user_id : removed) {
        auto choices = best_choices_for_user(
            occupied, options_by_user[user_id], base_penalty, scoring, quality_sum, 2, option_limit
        );
        if (choices.empty()) return answers;
        int64_t regret = choices.size() == 1 ? 1000000000LL : choices[1].score - choices[0].score;
        order.push_back({user_id, regret, choices[0].score, answers[user_id].quality});
    }

    std::sort(order.begin(), order.end(), [](const RepairOrderItem& a, const RepairOrderItem& b) {
        return std::make_tuple(a.regret, -a.score, a.old_quality)
            > std::make_tuple(b.regret, -b.score, b.old_quality);
    });

    for (const RepairOrderItem& item : order) {
        auto choices = best_choices_for_user(
            occupied, options_by_user[item.user_id], base_penalty, scoring, quality_sum, 1, option_limit
        );
        if (choices.empty()) return answers;
        Answer placed = choice_to_answer(choices[0]);
        trial[item.user_id] = placed;
        add_mask(occupied[placed.rb], choices[0].option->mask);
        quality_sum += placed.quality;
    }
    return trial;
}

std::vector<int> rb_loads_from_answers(const std::vector<Answer>& answers) {
    std::vector<int> loads(R, 0);
    for (const Answer& answer : answers) {
        if (answer.period == 640) continue;
        loads[answer.rb] += popcount(answer_mask(answer));
    }
    return loads;
}

std::vector<int> destroy_indices(
    const std::vector<Answer>& answers,
    const std::string& op,
    int destroy_size,
    std::mt19937& rng
) {
    std::vector<int> scheduled;
    for (int i = 0; i < static_cast<int>(answers.size()); ++i) {
        if (answers[i].period != 640) scheduled.push_back(i);
    }
    int count = std::min(destroy_size, static_cast<int>(scheduled.size()));
    if (count <= 0) return {};
    auto loads = rb_loads_from_answers(answers);

    if (op == "random") {
        std::shuffle(scheduled.begin(), scheduled.end(), rng);
        scheduled.resize(count);
        return scheduled;
    }

    if (op == "worst-quality") {
        auto comp = [&](int a, int b) {
            return std::tie(answers[a].quality, loads[answers[a].rb]) > std::tie(answers[b].quality, loads[answers[b].rb]);
        };
        std::partial_sort(scheduled.begin(), scheduled.begin() + count, scheduled.end(), comp);
        scheduled.resize(count);
        return scheduled;
    }

    if (op == "crowded-rb") {
        int target = int(std::max_element(loads.begin(), loads.end()) - loads.begin());
        std::vector<int> selected;
        for (int idx : scheduled) if (answers[idx].rb == target) selected.push_back(idx);
        std::sort(scheduled.begin(), scheduled.end(), [&](int a, int b) {
            return std::tie(answers[a].quality, loads[answers[a].rb]) > std::tie(answers[b].quality, loads[answers[b].rb]);
        });
        for (int idx : scheduled) {
            if (static_cast<int>(selected.size()) >= count) break;
            if (std::find(selected.begin(), selected.end(), idx) == selected.end()) selected.push_back(idx);
        }
        selected.resize(count);
        return selected;
    }

    if (op == "light-rb" || op == "evacuation-related") {
        int target = -1;
        for (int rb = 0; rb < R; ++rb) {
            if (loads[rb] == 0) continue;
            if (target < 0 || std::tie(loads[rb], rb) < std::tie(loads[target], target)) target = rb;
        }
        Mask target_mask;
        std::vector<int> selected;
        for (int idx : scheduled) {
            if (answers[idx].rb == target) {
                selected.push_back(idx);
                add_mask(target_mask, answer_mask(answers[idx]));
            }
        }
        if (op == "evacuation-related") {
            std::sort(scheduled.begin(), scheduled.end(), [&](int a, int b) {
                return std::make_tuple(intersection_popcount(answer_mask(answers[a]), target_mask), loads[answers[a].rb], answers[a].quality)
                    > std::make_tuple(intersection_popcount(answer_mask(answers[b]), target_mask), loads[answers[b].rb], answers[b].quality);
            });
        } else {
            std::sort(scheduled.begin(), scheduled.end(), [&](int a, int b) {
                return std::make_tuple(loads[answers[a].rb], answers[a].quality) > std::make_tuple(loads[answers[b].rb], answers[b].quality);
            });
        }
        for (int idx : scheduled) {
            if (static_cast<int>(selected.size()) >= count) break;
            if (std::find(selected.begin(), selected.end(), idx) == selected.end()) selected.push_back(idx);
        }
        selected.resize(count);
        return selected;
    }

    if (op == "conflict-graph") {
        std::array<int, L> pressure{};
        for (int idx : scheduled) {
            for (int slot = answers[idx].offset; slot < L; slot += answers[idx].period) ++pressure[slot];
        }
        auto comp = [&](int a, int b) {
            int score_a = 0;
            int score_b = 0;
            for (int slot = answers[a].offset; slot < L; slot += answers[a].period) score_a += pressure[slot];
            for (int slot = answers[b].offset; slot < L; slot += answers[b].period) score_b += pressure[slot];
            return std::make_tuple(score_a, loads[answers[a].rb], answers[a].quality)
                > std::make_tuple(score_b, loads[answers[b].rb], answers[b].quality);
        };
        std::partial_sort(scheduled.begin(), scheduled.begin() + count, scheduled.end(), comp);
        scheduled.resize(count);
        return scheduled;
    }

    if (op == "related-time") {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(scheduled.size()) - 1);
        int seed_idx = scheduled[pick(rng)];
        Mask seed_mask = answer_mask(answers[seed_idx]);
        auto comp = [&](int a, int b) {
            return std::make_tuple(
                a == seed_idx,
                answers[a].rb == answers[seed_idx].rb,
                intersection_popcount(answer_mask(answers[a]), seed_mask),
                -std::abs(answers[a].period - answers[seed_idx].period),
                answers[a].quality
            ) > std::make_tuple(
                b == seed_idx,
                answers[b].rb == answers[seed_idx].rb,
                intersection_popcount(answer_mask(answers[b]), seed_mask),
                -std::abs(answers[b].period - answers[seed_idx].period),
                answers[b].quality
            );
        };
        std::partial_sort(scheduled.begin(), scheduled.begin() + count, scheduled.end(), comp);
        scheduled.resize(count);
        return scheduled;
    }

    auto comp = [&](int a, int b) {
        return std::make_tuple(loads[answers[a].rb], popcount(answer_mask(answers[a])), answers[a].quality)
            > std::make_tuple(loads[answers[b].rb], popcount(answer_mask(answers[b])), answers[b].quality);
    };
    std::partial_sort(scheduled.begin(), scheduled.begin() + count, scheduled.end(), comp);
    scheduled.resize(count);
    return scheduled;
}

std::vector<Answer> solve_annealed_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    int rounds,
    int seed
) {
    std::vector<Answer> current = solve_portfolio_prepared(options_by_user, base_penalty);
    std::vector<Answer> best = current;
    int64_t current_obj = objective_value(current);
    int64_t best_obj = current_obj;
    std::vector<std::string> operators{
        "random", "worst-quality", "crowded-rb", "light-rb", "blockers", "related-time", "evacuation-related"
    };
    std::vector<std::string> repair_modes{"random", "regret"};
    std::vector<double> weights(operators.size() * repair_modes.size(), 1.0);
    std::mt19937 rng(seed);
    int n = static_cast<int>(options_by_user.size());
    int base_destroy = std::max(8, std::min(40, n / 9));
    double temperature = std::max(100.0, double(current_obj) * 0.015);

    for (int round = 0; round < rounds; ++round) {
        std::discrete_distribution<int> chooser(weights.begin(), weights.end());
        int op_mode = chooser(rng);
        int op_id = op_mode / static_cast<int>(repair_modes.size());
        int mode_id = op_mode % static_cast<int>(repair_modes.size());
        std::uniform_int_distribution<int> jitter_pick(0, 3);
        int jitter_values[4] = {-4, 0, 4, 8};
        int destroy_size = std::max(6, std::min(n, base_destroy + jitter_values[jitter_pick(rng)]));
        auto removed = destroy_indices(current, operators[op_id], destroy_size, rng);
        if (removed.empty()) continue;

        std::string scoring = (operators[op_id] == "light-rb" || operators[op_id] == "blockers" || operators[op_id] == "evacuation-related")
            ? "adaptive" : "marginal";
        int penalty = scoring == "adaptive" ? 90 : base_penalty;
        std::vector<Answer> candidate = repair_modes[mode_id] == "regret"
            ? repair_regret(options_by_user, current, removed, penalty, scoring)
            : repair_random(options_by_user, current, removed, penalty, scoring, rng);

        int64_t candidate_obj = objective_value(candidate);
        int64_t delta = candidate_obj - current_obj;
        bool accept = delta <= 0;
        if (!accept && delta < 100000000000LL) {
            std::uniform_real_distribution<double> unit(0.0, 1.0);
            accept = unit(rng) < std::exp(-double(delta) / std::max(temperature, 1.0));
        }
        if (accept) {
            current = std::move(candidate);
            current_obj = candidate_obj;
            weights[op_mode] += 0.4;
        } else {
            weights[op_mode] = std::max(0.2, weights[op_mode] * 0.9);
        }
        if (candidate_obj < best_obj) {
            if (accept) {
                best = current;
            } else {
                best = candidate;
            }
            best_obj = candidate_obj;
            weights[op_mode] += 4.0;
        }
        temperature *= 0.90;
    }
    return best;
}

enum class BanditContext {
    RbCompression = 0,
    Crowding = 1,
    QualityPolish = 2,
    Plateau = 3,
};

struct BanditArm {
    std::string op;
    std::string repair;
    std::string size_name;
    double multiplier = 1.0;
    int count = 1;
    double mean = 0.0;
};

BanditContext contextual_bandit_context_cpp(
    const std::vector<Answer>& answers,
    const SolutionStats& stats_value,
    int no_improve_rounds,
    int last_gain_type
) {
    if (no_improve_rounds >= 5) return BanditContext::Plateau;
    if (last_gain_type == 1 && no_improve_rounds <= 2) return BanditContext::QualityPolish;

    auto loads = rb_loads_from_answers(answers);
    std::vector<int> active_loads;
    for (int load : loads) {
        if (load > 0) active_loads.push_back(load);
    }
    if (active_loads.empty()) return BanditContext::QualityPolish;

    int rb_capacity = 0;
    for (int slot = 0; slot < L; ++slot) {
        if (is_uplink_slot(slot, 8, 2)) ++rb_capacity;
    }
    int rb_lower_bound = std::max(1, (stats_value.occupied_pairs + rb_capacity - 1) / rb_capacity);
    double avg_load = double(std::accumulate(active_loads.begin(), active_loads.end(), 0)) / active_loads.size();
    int max_load = *std::max_element(active_loads.begin(), active_loads.end());
    int light_rbs = 0;
    for (int load : active_loads) {
        if (load <= std::max(1.0, 0.35 * avg_load)) ++light_rbs;
    }
    int rb_slack = stats_value.rb_used - rb_lower_bound;
    double imbalance = double(max_load) / std::max(1.0, avg_load);

    if (rb_slack >= 2 || (rb_slack >= 1 && light_rbs > 0)) return BanditContext::RbCompression;
    if (imbalance >= 1.55) return BanditContext::Crowding;
    return BanditContext::QualityPolish;
}

std::array<std::vector<BanditArm>, 4> make_contextual_bandit_arms() {
    std::array<std::vector<BanditArm>, 4> arms;
    arms[0] = {
        {"evacuation-related", "regret", "large", 1.35, 4, 1.40},
        {"light-rb", "regret", "large", 1.30, 3, 1.10},
        {"conflict-graph", "regret", "large", 1.25, 2, 0.90},
        {"crowded-rb", "regret", "base", 1.00, 2, 0.75},
        {"blockers", "regret", "large", 1.25, 2, 0.70},
    };
    arms[1] = {
        {"crowded-rb", "regret", "base", 1.00, 4, 1.20},
        {"conflict-graph", "regret", "base", 1.00, 3, 1.05},
        {"blockers", "regret", "base", 1.00, 3, 0.95},
        {"related-time", "regret", "base", 1.00, 2, 0.75},
        {"evacuation-related", "regret", "large", 1.30, 2, 0.65},
    };
    arms[2] = {
        {"related-time", "regret", "base", 1.00, 4, 1.25},
        {"worst-quality", "regret", "base", 1.00, 3, 1.00},
        {"conflict-graph", "regret", "base", 1.00, 2, 0.70},
        {"related-time", "random", "base", 1.00, 2, 0.65},
        {"crowded-rb", "regret", "base", 1.00, 2, 0.55},
    };
    arms[3] = {
        {"evacuation-related", "regret", "xlarge", 1.65, 3, 1.20},
        {"conflict-graph", "regret", "large", 1.35, 3, 1.10},
        {"related-time", "regret", "large", 1.35, 3, 1.05},
        {"random", "regret", "large", 1.35, 2, 0.75},
        {"blockers", "regret", "large", 1.35, 2, 0.70},
    };
    return arms;
}

void adjust_contextual_bandit_arms(
    std::array<std::vector<BanditArm>, 4>& arms,
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    double avg_options = 0.0;
    int min_pairs = 0;
    for (const auto& options : options_by_user) {
        avg_options += static_cast<double>(options.size());
        int best_pairs = L;
        for (const TimeOption& option : options) best_pairs = std::min(best_pairs, popcount(option.mask));
        min_pairs += best_pairs == L ? 1 : best_pairs;
    }
    avg_options /= std::max<size_t>(1, options_by_user.size());
    int rb_capacity = 0;
    for (int slot = 0; slot < L; ++slot) if (is_uplink_slot(slot, 8, 2)) ++rb_capacity;
    int rb_lower_bound = std::max(1, (min_pairs + rb_capacity - 1) / rb_capacity);
    double pressure = double(rb_lower_bound) / R;

    auto bump = [&](BanditContext context, const std::string& op, double mean_delta, int count_delta) {
        for (BanditArm& arm : arms[static_cast<int>(context)]) {
            if (arm.op == op) {
                arm.mean += mean_delta;
                arm.count += count_delta;
            }
        }
    };

    if (options_by_user.size() >= 300 || pressure > 0.35) {
        bump(BanditContext::RbCompression, "evacuation-related", 0.15, 1);
        bump(BanditContext::RbCompression, "conflict-graph", 0.20, 1);
        bump(BanditContext::Plateau, "conflict-graph", 0.15, 1);
    }
    if (avg_options > 40.0) {
        bump(BanditContext::QualityPolish, "related-time", 0.10, 1);
        bump(BanditContext::Crowding, "conflict-graph", 0.10, 1);
    }
}

int thompson_arm_choice(std::vector<BanditArm>& arms, std::mt19937& rng, double exploration, double epsilon) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(rng) < epsilon) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(arms.size()) - 1);
        return pick(rng);
    }
    int best_idx = 0;
    double best_score = -1e100;
    for (int i = 0; i < static_cast<int>(arms.size()); ++i) {
        double sigma = exploration / std::sqrt(std::max(1, arms[i].count));
        std::normal_distribution<double> normal(0.0, sigma);
        double score = arms[i].mean + normal(rng);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return best_idx;
}

void update_bandit_arm(BanditArm& arm, double reward) {
    ++arm.count;
    arm.mean += (reward - arm.mean) / arm.count;
}

std::vector<Answer> solve_contextual_bandit_polish_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& start_answers,
    int base_penalty,
    int polish_rounds,
    int seed
) {
    std::vector<Answer> current = start_answers;
    std::vector<Answer> best = current;
    SolutionStats current_stats = compact_stats(current);
    SolutionStats best_stats = current_stats;
    int64_t current_obj = current_stats.objective;
    int64_t best_obj = best_stats.objective;

    auto arms_by_context = make_contextual_bandit_arms();
    adjust_contextual_bandit_arms(arms_by_context, options_by_user);
    std::mt19937 rng(seed + 91000);
    int n = static_cast<int>(options_by_user.size());
    int base_destroy = std::max(8, std::min(30, n / 13));
    double temperature = std::max(50.0, double(current_obj) * 0.005);
    int no_improve_rounds = 0;
    int last_gain_type = 0;  // 0 none/objective, 1 rb, 2 quality

    for (int round = 0; round < polish_rounds; ++round) {
        if (round >= 8 && no_improve_rounds >= 7) break;

        BanditContext context = contextual_bandit_context_cpp(current, current_stats, no_improve_rounds, last_gain_type);
        int context_idx = static_cast<int>(context);
        double progress = double(round) / std::max(1, polish_rounds);
        double epsilon = std::max(0.015, 0.07 * (1.0 - progress));
        double exploration = context == BanditContext::Plateau ? 0.80 : 0.55;
        int arm_idx = thompson_arm_choice(arms_by_context[context_idx], rng, exploration, epsilon);
        BanditArm& arm = arms_by_context[context_idx][arm_idx];

        double multiplier = arm.multiplier;
        if (context == BanditContext::QualityPolish) multiplier = std::min(multiplier, 1.0);
        if (context == BanditContext::Plateau) multiplier *= 1.1;
        std::uniform_int_distribution<int> jitter_pick(-1, 1);
        int destroy_size = std::max(6, std::min(n, int(std::round(base_destroy * multiplier)) + jitter_pick(rng)));
        auto removed = destroy_indices(current, arm.op, destroy_size, rng);
        if (removed.empty()) {
            update_bandit_arm(arm, 0.0);
            ++no_improve_rounds;
            temperature *= 0.88;
            continue;
        }

        std::string scoring = (arm.op == "light-rb" || arm.op == "blockers" || arm.op == "evacuation-related" || arm.op == "conflict-graph")
            ? "adaptive" : "marginal";
        int penalty = scoring == "adaptive" ? 105 : base_penalty;
        std::vector<Answer> candidate = arm.repair == "regret"
            ? repair_regret_fast(options_by_user, current, removed, penalty, scoring, BANDIT_REPAIR_OPTION_LIMIT)
            : repair_random(options_by_user, current, removed, penalty, scoring, rng, BANDIT_REPAIR_OPTION_LIMIT);
        SolutionStats candidate_stats = compact_stats(candidate);
        int64_t candidate_obj = candidate_stats.objective;
        int64_t delta = candidate_obj - current_obj;
        bool accept = delta <= 0;
        if (!accept && delta < 100000000000LL) {
            std::uniform_real_distribution<double> unit(0.0, 1.0);
            accept = unit(rng) < std::exp(-double(delta) / std::max(temperature, 1.0));
        }

        double objective_gain = std::max(0.0, double(current_obj - candidate_obj) / std::max<int64_t>(1, current_obj));
        double best_gain = std::max(0.0, double(best_obj - candidate_obj) / std::max<int64_t>(1, best_obj));
        int rb_gain = std::max(0, current_stats.rb_used - candidate_stats.rb_used);
        int quality_gain = std::max(0, current_stats.quality_sum - candidate_stats.quality_sum);
        int rb_penalty = std::max(0, candidate_stats.rb_used - current_stats.rb_used);
        double reward = 0.02 + 55.0 * objective_gain + 100.0 * best_gain
            + 1.1 * rb_gain + std::min(1.6, quality_gain / 100.0) - 0.25 * rb_penalty;

        if (accept) {
            current = std::move(candidate);
            current_stats = candidate_stats;
            current_obj = candidate_obj;
        }

        if (candidate_obj < best_obj) {
            SolutionStats old_best = best_stats;
            if (accept) {
                best = current;
            } else {
                best = candidate;
            }
            best_stats = candidate_stats;
            best_obj = candidate_obj;
            no_improve_rounds = 0;
            if (best_stats.rb_used < old_best.rb_used) last_gain_type = 1;
            else if (best_stats.quality_sum < old_best.quality_sum) last_gain_type = 2;
            else last_gain_type = 0;
            reward += 2.5;
        } else {
            ++no_improve_rounds;
        }
        update_bandit_arm(arm, reward);
        temperature *= 0.88;
    }
    return best;
}

std::vector<Answer> solve_contextual_bandit_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    int rounds,
    int seed
) {
    std::vector<Answer> start = solve_annealed_prepared(options_by_user, base_penalty, rounds, seed);
    return solve_contextual_bandit_polish_prepared(
        options_by_user,
        start,
        base_penalty,
        std::max(8, rounds / 2),
        seed
    );
}

std::vector<Answer> solve_parallel_contextual_bandit_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    int rounds,
    int seed,
    int workers
) {
    workers = std::max(1, workers);
    if (workers == 1) return solve_contextual_bandit_prepared(options_by_user, base_penalty, rounds, seed);

    std::vector<std::future<std::vector<Answer>>> initial_futures;
    initial_futures.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
        initial_futures.push_back(std::async(
            std::launch::async,
            [&options_by_user, base_penalty, rounds, seed, worker]() {
                return solve_contextual_bandit_prepared(
                    options_by_user,
                    base_penalty,
                    rounds,
                    seed + 1009 * worker
                );
            }
        ));
    }

    std::vector<Answer> incumbent;
    int64_t incumbent_obj = std::numeric_limits<int64_t>::max();
    for (auto& future : initial_futures) {
        std::vector<Answer> candidate = future.get();
        int64_t candidate_obj = objective_value(candidate);
        if (candidate_obj < incumbent_obj) {
            incumbent_obj = candidate_obj;
            incumbent = std::move(candidate);
        }
    }

    int shared_rounds = std::max(4, rounds / 4);
    std::vector<std::future<std::vector<Answer>>> polish_futures;
    polish_futures.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
        std::vector<Answer> shared_start = incumbent;
        polish_futures.push_back(std::async(
            std::launch::async,
            [&options_by_user, shared_start, base_penalty, shared_rounds, seed, worker]() {
                return solve_contextual_bandit_polish_prepared(
                    options_by_user,
                    shared_start,
                    base_penalty,
                    shared_rounds,
                    seed + 20011 + 1009 * worker
                );
            }
        ));
    }
    for (auto& future : polish_futures) {
        std::vector<Answer> candidate = future.get();
        int64_t candidate_obj = objective_value(candidate);
        if (candidate_obj < incumbent_obj) {
            incumbent_obj = candidate_obj;
            incumbent = std::move(candidate);
        }
    }
    return incumbent;
}

std::vector<Answer> solve_independent_parallel_contextual_bandit_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    int rounds,
    int seed,
    int workers
) {
    workers = std::max(1, workers);
    std::vector<std::future<std::vector<Answer>>> futures;
    futures.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
        futures.push_back(std::async(
            std::launch::async,
            [&options_by_user, base_penalty, rounds, seed, worker]() {
                return solve_contextual_bandit_prepared(
                    options_by_user,
                    base_penalty,
                    rounds,
                    seed + 1009 * worker
                );
            }
        ));
    }

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    for (auto& future : futures) {
        std::vector<Answer> candidate = future.get();
        int64_t candidate_obj = objective_value(candidate);
        if (candidate_obj < best_obj) {
            best_obj = candidate_obj;
            best = std::move(candidate);
        }
    }
    return best;
}

std::vector<User> generate_random_users(int n, int seed, const std::string& difficulty) {
    std::mt19937 rng(seed);
    std::vector<int> possible_periods{5, 10, 20, 40, 80, 160, 320};
    std::vector<User> users;
    users.reserve(n);

    for (int i = 0; i < n; ++i) {
        User user;
        int start = 2;
        int srs_count = 1;
        if (difficulty == "hard") {
            std::uniform_int_distribution<int> start_dist(0, 3);
            std::uniform_int_distribution<int> srs_period_idx(0, 2);
            std::uniform_int_distribution<int> count_dist(1, 4);
            start = start_dist(rng);
            user.srs_period = std::vector<int>{20, 40, 80}[srs_period_idx(rng)];
            srs_count = count_dist(rng);
        } else if (difficulty == "large") {
            std::uniform_int_distribution<int> start_dist(2, 5);
            std::uniform_int_distribution<int> srs_period_idx(0, 2);
            std::uniform_int_distribution<int> count_dist(1, 3);
            start = start_dist(rng);
            user.srs_period = std::vector<int>{40, 80, 160}[srs_period_idx(rng)];
            srs_count = count_dist(rng);
        } else {
            std::uniform_int_distribution<int> start_choice(0, 2);
            std::uniform_int_distribution<int> srs_period_idx(0, 1);
            std::uniform_int_distribution<int> count_dist(1, 3);
            start = std::vector<int>{2, 3, 4}[start_choice(rng)];
            user.srs_period = std::vector<int>{40, 80}[srs_period_idx(rng)];
            srs_count = count_dist(rng);
        }

        for (size_t p = start; p < possible_periods.size(); ++p) user.csi_periods.push_back(possible_periods[p]);

        std::vector<int> offsets(user.srs_period);
        std::iota(offsets.begin(), offsets.end(), 0);
        std::shuffle(offsets.begin(), offsets.end(), rng);
        offsets.resize(std::min<int>(srs_count, offsets.size()));
        user.srs_offsets = offsets;
        users.push_back(user);
    }
    return users;
}

std::vector<int> parse_int_array(const std::string& text, size_t& pos) {
    std::vector<int> values;
    pos = text.find('[', pos);
    if (pos == std::string::npos) return values;
    ++pos;
    while (pos < text.size() && text[pos] != ']') {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == ',')) ++pos;
        if (pos < text.size() && (std::isdigit(text[pos]) || text[pos] == '-')) {
            size_t next = pos;
            int value = std::stoi(text.substr(pos), &next);
            values.push_back(value);
            pos += next;
        } else {
            ++pos;
        }
    }
    if (pos < text.size() && text[pos] == ']') ++pos;
    return values;
}

int parse_int_field(const std::string& text, const std::string& key, size_t start = 0) {
    size_t pos = text.find("\"" + key + "\"", start);
    if (pos == std::string::npos) return 0;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return 0;
    ++pos;
    while (pos < text.size() && text[pos] == ' ') ++pos;
    return std::stoi(text.substr(pos));
}

std::vector<User> parse_users_json_line(const std::string& line) {
    std::vector<User> users;
    size_t users_pos = line.find("\"users\"");
    if (users_pos == std::string::npos) return users;
    size_t pos = line.find('{', users_pos);
    while (pos != std::string::npos) {
        size_t end = line.find('}', pos);
        if (end == std::string::npos) break;
        std::string object = line.substr(pos, end - pos + 1);
        if (object.find("csi_periods") != std::string::npos) {
            User user;
            size_t csi_pos = object.find("\"csi_periods\"");
            user.csi_periods = parse_int_array(object, csi_pos);
            user.srs_period = parse_int_field(object, "srs_period");
            size_t srs_pos = object.find("\"srs_offsets\"");
            user.srs_offsets = parse_int_array(object, srs_pos);
            users.push_back(user);
        }
        pos = line.find('{', end + 1);
    }
    return users;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 450;
    int seeds = 3;
    int cap = 24;
    int base_penalty = 60;
    int rounds = 28;
    int workers = 2;
    std::string difficulty = "large";
    std::string input_jsonl;
    std::string mode = "adaptive";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--seeds" && i + 1 < argc) seeds = std::stoi(argv[++i]);
        else if (arg == "--cap" && i + 1 < argc) cap = std::stoi(argv[++i]);
        else if (arg == "--base-penalty" && i + 1 < argc) base_penalty = std::stoi(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc) rounds = std::stoi(argv[++i]);
        else if (arg == "--workers" && i + 1 < argc) workers = std::stoi(argv[++i]);
        else if (arg == "--difficulty" && i + 1 < argc) difficulty = argv[++i];
        else if (arg == "--input-jsonl" && i + 1 < argc) input_jsonl = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    }

    double total_ms = 0.0;
    int64_t total_obj = 0;
    int cases = 0;

    double total_prepare_ms = 0.0;

    auto run_case = [&](const std::vector<User>& users, int case_id) {
        auto prepare_start = std::chrono::steady_clock::now();
        auto options_by_user = prepare_options(users, 8, 2, cap);
        auto prepare_end = std::chrono::steady_clock::now();
        auto start = std::chrono::steady_clock::now();
        std::vector<Answer> answers;
        if (mode == "annealed") {
            answers = solve_annealed_prepared(options_by_user, base_penalty, rounds, 7000 + case_id);
        } else if (mode == "bandit") {
            answers = solve_contextual_bandit_prepared(options_by_user, base_penalty, rounds, 7000 + case_id);
        } else if (mode == "pbandit" || mode == "parallel-bandit") {
            answers = solve_parallel_contextual_bandit_prepared(
                options_by_user,
                base_penalty,
                rounds,
                7000 + case_id,
                workers
            );
        } else if (mode == "ipbandit" || mode == "independent-parallel-bandit") {
            answers = solve_independent_parallel_contextual_bandit_prepared(
                options_by_user,
                base_penalty,
                rounds,
                7000 + case_id,
                workers
            );
        } else {
            answers = solve_adaptive_prepared(options_by_user, base_penalty);
        }
        auto end = std::chrono::steady_clock::now();
        auto [rb_used, quality_sum, unscheduled] = stats(answers);
        int64_t objective = int64_t(rb_used) * quality_sum;
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double prepare_ms = std::chrono::duration<double, std::milli>(prepare_end - prepare_start).count();
        total_ms += ms;
        total_prepare_ms += prepare_ms;
        total_obj += objective;
        ++cases;
        std::cout << "case=" << case_id
                  << " obj=" << objective
                  << " rb=" << rb_used
                  << " q=" << quality_sum
                  << " uns=" << unscheduled
                  << " prepare_ms=" << prepare_ms
                  << " solve_ms=" << ms << "\n";
    };

    if (!input_jsonl.empty()) {
        std::ifstream in(input_jsonl);
        std::string line;
        int case_id = 0;
        while (std::getline(in, line)) {
            auto users = parse_users_json_line(line);
            run_case(users, ++case_id);
        }
    } else {
        for (int seed = 1; seed <= seeds; ++seed) {
            auto users = generate_random_users(n, seed, difficulty);
            run_case(users, seed);
        }
    }
    if (cases > 0) {
        std::cout << "avg_obj=" << (total_obj / cases)
                  << " avg_prepare_ms=" << (total_prepare_ms / cases)
                  << " avg_solve_ms=" << (total_ms / cases) << "\n";
    }
    return 0;
}
