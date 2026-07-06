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
#include <queue>
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

struct BaseOptionCacheEntry {
    int period = 0;
    int dl = 0;
    int ul = 0;
    std::vector<BaseTimeOption> options;
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

void remove_mask(Mask& a, const Mask& b) {
    for (int i = 0; i < WORDS; ++i) a.w[i] &= ~b.w[i];
}

int popcount(const Mask& a) {
    int total = 0;
    for (uint64_t word : a.w) total += __builtin_popcountll(word);
    return total;
}

int first_set_slot(const Mask& a) {
    for (int word_idx = 0; word_idx < WORDS; ++word_idx) {
        uint64_t word = a.w[word_idx];
        if (word) return word_idx * 64 + __builtin_ctzll(word);
    }
    return L;
}

template <class Fn>
void for_each_slot_in_mask(const Mask& mask, Fn&& fn) {
    for (int word_idx = 0; word_idx < WORDS; ++word_idx) {
        uint64_t word = mask.w[word_idx];
        while (word) {
            int bit = __builtin_ctzll(word);
            fn(word_idx * 64 + bit);
            word &= word - 1;
        }
    }
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
    static std::vector<BaseOptionCacheEntry> cache;
    for (const BaseOptionCacheEntry& entry : cache) {
        if (entry.period == period && entry.dl == dl && entry.ul == ul) return entry.options;
    }

    BaseOptionCacheEntry entry;
    entry.period = period;
    entry.dl = dl;
    entry.ul = ul;
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
        if (mask_empty(option.mask)) continue;
        entry.options.push_back(option);
    }
    cache.push_back(std::move(entry));
    return cache.back().options;
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
    if (cap > 0 && static_cast<int>(options.size()) > cap) {
        std::vector<TimeOption> capped;
        capped.reserve(cap);

        std::vector<TimeOption> sparse_options;
        for (const TimeOption& option : options) {
            int cells = popcount(option.mask);
            if (cells == 1 || (cap <= 32 && cells == 2)) sparse_options.push_back(option);
        }
        std::sort(sparse_options.begin(), sparse_options.end(), [](const TimeOption& a, const TimeOption& b) {
            return std::make_tuple(popcount(a.mask), a.offset, a.quality, a.period)
                < std::make_tuple(popcount(b.mask), b.offset, b.quality, b.period);
        });

        int sparse_budget = static_cast<int>(sparse_options.size());
        int target_size = cap + sparse_budget;
        sparse_budget = std::min<int>(sparse_options.size(), sparse_budget);

        capped.reserve(target_size);
        auto add_unique = [&](const TimeOption& option) {
            for (const TimeOption& existing : capped) {
                if (existing.period == option.period && existing.offset == option.offset) return;
            }
            if (static_cast<int>(capped.size()) < target_size) capped.push_back(option);
        };

        for (int i = 0; i < sparse_budget; ++i) add_unique(sparse_options[i]);
        for (const TimeOption& option : options) {
            if (static_cast<int>(capped.size()) >= target_size) break;
            add_unique(option);
        }
        std::sort(capped.begin(), capped.end(), [](const TimeOption& a, const TimeOption& b) {
            return std::tie(a.quality, a.period, a.offset) < std::tie(b.quality, b.period, b.offset);
        });
        options = std::move(capped);
    }
    return options;
}

std::vector<std::vector<TimeOption>> prepare_options(const std::vector<User>& users, int dl, int ul, int cap) {
    std::vector<std::vector<TimeOption>> options_by_user;
    options_by_user.reserve(users.size());
    for (const User& user : users) options_by_user.push_back(build_time_options(user, dl, ul, cap));
    return options_by_user;
}

std::vector<std::vector<TimeOption>> prepare_sparse_online_options(
    const std::vector<User>& users,
    int dl,
    int ul
) {
    std::vector<std::vector<TimeOption>> options_by_user;
    options_by_user.reserve(users.size());
    for (const User& user : users) {
        std::vector<TimeOption> options;
        int best_cells = L + 1;
        for (int period : user.csi_periods) {
            for (const BaseTimeOption& base : base_options_for_period(period, dl, ul)) {
                int cells = popcount(base.mask);
                if (cells > best_cells) continue;
                if (cells < best_cells) {
                    best_cells = cells;
                    options.clear();
                }
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
        if (options.empty()) options = build_time_options(user, dl, ul, 0);
        options_by_user.push_back(std::move(options));
    }
    return options_by_user;
}

std::vector<std::vector<TimeOption>> limited_options_by_user(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int limit
) {
    std::vector<std::vector<TimeOption>> limited;
    limited.reserve(options_by_user.size());
    for (const auto& options : options_by_user) {
        int count = std::min<int>(limit, options.size());
        limited.emplace_back(options.begin(), options.begin() + count);
    }
    return limited;
}

int seen_uplink_slot_count(const std::vector<std::vector<TimeOption>>& options_by_user) {
    std::array<uint8_t, L> seen{};
    for (const auto& options : options_by_user) {
        for (const TimeOption& option : options) {
            for_each_slot_in_mask(option.mask, [&](int slot) {
                if (slot >= 0 && slot < L) seen[slot] = 1;
            });
        }
    }

    int count = 0;
    for (uint8_t value : seen) count += value ? 1 : 0;
    return std::max(1, count);
}

int occupied_cell_lower_bound_rb(const std::vector<std::vector<TimeOption>>& options_by_user) {
    int required_cells = 0;
    for (const auto& options : options_by_user) {
        int best_cells = L + 1;
        for (const TimeOption& option : options) best_cells = std::min(best_cells, popcount(option.mask));
        required_cells += best_cells == L + 1 ? 1 : best_cells;
    }
    int uplink_slots = seen_uplink_slot_count(options_by_user);
    return std::max(1, (required_cells + uplink_slots - 1) / uplink_slots);
}

std::pair<int, int> rb_budget_window(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int extra_budgets,
    int large_extra_budgets
) {
    int lower = occupied_cell_lower_bound_rb(options_by_user);
    int n = static_cast<int>(options_by_user.size());
    int extra = n > 2000 ? large_extra_budgets : extra_budgets;
    int upper = std::min(R, lower + std::max(0, extra));
    return {lower, upper};
}

struct InstanceFeatures {
    int n = 0;
    int uplink_slots = 0;
    int rb_cell_lb = 1;
    double pressure = 0.0;
    int one_cell_users = 0;
    double one_cell_ratio = 0.0;
    bool all_one_cell_feasible = false;
    double avg_options = 0.0;
    int min_options = 0;
    double avg_best_cells = 0.0;
    double avg_regret = 0.0;
    int p95_regret = 0;
};

InstanceFeatures compute_instance_features(const std::vector<std::vector<TimeOption>>& options_by_user) {
    InstanceFeatures features;
    features.n = static_cast<int>(options_by_user.size());
    features.uplink_slots = seen_uplink_slot_count(options_by_user);
    features.rb_cell_lb = occupied_cell_lower_bound_rb(options_by_user);
    features.pressure = double(features.rb_cell_lb) / R;
    features.min_options = features.n == 0 ? 0 : std::numeric_limits<int>::max();

    std::vector<int> regrets;
    regrets.reserve(options_by_user.size());
    for (const auto& options : options_by_user) {
        features.avg_options += static_cast<double>(options.size());
        features.min_options = std::min<int>(features.min_options, options.size());

        int one_cell_count = 0;
        int best_cells = L + 1;
        int best_quality = std::numeric_limits<int>::max();
        int second_quality = std::numeric_limits<int>::max();
        for (const TimeOption& option : options) {
            int cells = popcount(option.mask);
            if (cells == 1) ++one_cell_count;
            best_cells = std::min(best_cells, cells);
            if (option.quality < best_quality) {
                second_quality = best_quality;
                best_quality = option.quality;
            } else if (option.quality < second_quality) {
                second_quality = option.quality;
            }
        }
        if (one_cell_count > 0) ++features.one_cell_users;
        features.avg_best_cells += best_cells == L + 1 ? 0.0 : static_cast<double>(best_cells);
        int regret = second_quality == std::numeric_limits<int>::max()
            ? 1000000
            : std::max(0, second_quality - best_quality);
        regrets.push_back(regret);
        features.avg_regret += regret;
    }

    if (features.n > 0) {
        features.avg_options /= features.n;
        features.avg_best_cells /= features.n;
        features.avg_regret /= features.n;
        features.one_cell_ratio = double(features.one_cell_users) / features.n;
        features.all_one_cell_feasible = features.one_cell_users == features.n;
    }
    if (!regrets.empty()) {
        std::sort(regrets.begin(), regrets.end());
        int idx = std::min<int>(regrets.size() - 1, static_cast<int>(std::ceil(0.95 * regrets.size())) - 1);
        features.p95_regret = regrets[std::max(0, idx)];
    }
    return features;
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

const TimeOption* find_time_option(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int user_id,
    int period,
    int offset
) {
    if (user_id < 0 || user_id >= static_cast<int>(options_by_user.size())) return nullptr;
    for (const TimeOption& option : options_by_user[user_id]) {
        if (option.period == period && option.offset == offset) return &option;
    }
    return nullptr;
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
        result.occupied_pairs += popcount(periodic_mask(answer.period, answer.offset));
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
    std::vector<uint8_t> skipped(answers.size(), 0);
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

std::vector<Answer> solve_saturation_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    std::vector<Mask> occupied(R);
    std::array<uint8_t, R> active{};
    std::vector<int> active_rbs;
    active_rbs.reserve(R);
    int first_empty = 0;
    std::vector<Answer> answers(options_by_user.size());

    for (int user_id = 0; user_id < static_cast<int>(options_by_user.size()); ++user_id) {
        bool found = false;
        Answer best;
        const TimeOption* best_option = nullptr;
        std::tuple<int, int, int, int, int, int> best_tuple{};

        std::vector<int> rb_candidates = active_rbs;
        if (first_empty < R) rb_candidates.push_back(first_empty);

        for (const TimeOption& option : options_by_user[user_id]) {
            int occupancy = popcount(option.mask);
            for (int rb : rb_candidates) {
                if (intersects(occupied[rb], option.mask)) continue;
                bool opens = !active[rb];
                auto candidate_tuple = std::make_tuple(
                    occupancy,
                    opens ? 1 : 0,
                    option.quality,
                    -option.period,
                    rb,
                    option.offset
                );
                if (!found || candidate_tuple < best_tuple) {
                    found = true;
                    best_tuple = candidate_tuple;
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
            answers[user_id] = best;
            if (!active[best.rb]) {
                active[best.rb] = 1;
                active_rbs.push_back(best.rb);
                while (first_empty < R && active[first_empty]) ++first_empty;
            }
            add_mask(occupied[best.rb], best_option->mask);
        }
    }
    return answers;
}

std::vector<Answer> solve_sparse_capacity_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool balance_load_first
) {
    struct UserOrder {
        int user_id = 0;
        int sparse_count = 0;
        int best_quality = std::numeric_limits<int>::max();
        int regret = 0;
    };

    std::vector<UserOrder> order;
    order.reserve(options_by_user.size());
    for (int user_id = 0; user_id < static_cast<int>(options_by_user.size()); ++user_id) {
        std::vector<int> qualities;
        for (const TimeOption& option : options_by_user[user_id]) {
            if (popcount(option.mask) != 1) continue;
            qualities.push_back(option.quality);
        }
        std::sort(qualities.begin(), qualities.end());
        UserOrder item;
        item.user_id = user_id;
        item.sparse_count = static_cast<int>(qualities.size());
        if (!qualities.empty()) {
            item.best_quality = qualities[0];
            item.regret = qualities.size() == 1 ? 1000000 : qualities[1] - qualities[0];
        }
        order.push_back(item);
    }

    std::sort(order.begin(), order.end(), [](const UserOrder& a, const UserOrder& b) {
        bool a_has = a.sparse_count > 0;
        bool b_has = b.sparse_count > 0;
        return std::make_tuple(!a_has, a.sparse_count == 0 ? 1000000 : a.sparse_count, -a.regret, a.best_quality, a.user_id)
            < std::make_tuple(!b_has, b.sparse_count == 0 ? 1000000 : b.sparse_count, -b.regret, b.best_quality, b.user_id);
    });

    std::vector<Answer> answers(options_by_user.size());
    std::array<int, L> slot_load{};

    for (const UserOrder& item : order) {
        if (item.sparse_count == 0) continue;
        bool found = false;
        Answer best;
        const TimeOption* best_option = nullptr;
        std::tuple<int, int, int, int> best_tuple{};

        for (const TimeOption& option : options_by_user[item.user_id]) {
            if (popcount(option.mask) != 1) continue;
            int slot = first_set_slot(option.mask);
            if (slot >= L || slot_load[slot] >= R) continue;
            auto candidate_tuple = balance_load_first
                ? std::make_tuple(slot_load[slot], option.quality, slot, option.offset)
                : std::make_tuple(option.quality, slot_load[slot], slot, option.offset);
            if (!found || candidate_tuple < best_tuple) {
                found = true;
                best_tuple = candidate_tuple;
                best_option = &option;
                best.rb = slot_load[slot];
                best.period = option.period;
                best.offset = option.offset;
                best.distance = option.distance;
                best.quality = option.quality;
            }
        }

        if (found && best_option != nullptr) {
            int slot = first_set_slot(best_option->mask);
            answers[item.user_id] = best;
            ++slot_load[slot];
        }
    }

    return answers;
}

struct FlowEdge {
    int to = 0;
    int rev = 0;
    int cap = 0;
    int cost = 0;
};

struct MinCostFlow {
    std::vector<std::vector<FlowEdge>> graph;

    explicit MinCostFlow(int n) : graph(n) {}

    int add_edge(int from, int to, int cap, int cost) {
        FlowEdge forward{to, static_cast<int>(graph[to].size()), cap, cost};
        FlowEdge backward{from, static_cast<int>(graph[from].size()), 0, -cost};
        graph[from].push_back(forward);
        graph[to].push_back(backward);
        return static_cast<int>(graph[from].size()) - 1;
    }

    std::pair<int, int64_t> min_cost_flow(int source, int sink, int max_flow) {
        int n = static_cast<int>(graph.size());
        const int64_t INF = std::numeric_limits<int64_t>::max() / 4;
        std::vector<int64_t> potential(n, 0), dist(n);
        std::vector<int> prev_node(n), prev_edge(n);
        int flow = 0;
        int64_t cost = 0;

        while (flow < max_flow) {
            std::fill(dist.begin(), dist.end(), INF);
            dist[source] = 0;
            using State = std::pair<int64_t, int>;
            std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
            pq.push({0, source});

            while (!pq.empty()) {
                auto [current_dist, node] = pq.top();
                pq.pop();
                if (current_dist != dist[node]) continue;
                for (int edge_idx = 0; edge_idx < static_cast<int>(graph[node].size()); ++edge_idx) {
                    const FlowEdge& edge = graph[node][edge_idx];
                    if (edge.cap <= 0) continue;
                    int64_t next_dist = current_dist + edge.cost + potential[node] - potential[edge.to];
                    if (next_dist < dist[edge.to]) {
                        dist[edge.to] = next_dist;
                        prev_node[edge.to] = node;
                        prev_edge[edge.to] = edge_idx;
                        pq.push({next_dist, edge.to});
                    }
                }
            }

            if (dist[sink] == INF) break;
            for (int node = 0; node < n; ++node) {
                if (dist[node] < INF) potential[node] += dist[node];
            }

            int add = max_flow - flow;
            for (int node = sink; node != source; node = prev_node[node]) {
                add = std::min(add, graph[prev_node[node]][prev_edge[node]].cap);
            }
            for (int node = sink; node != source; node = prev_node[node]) {
                FlowEdge& edge = graph[prev_node[node]][prev_edge[node]];
                edge.cap -= add;
                graph[node][edge.rev].cap += add;
                cost += int64_t(add) * edge.cost;
            }
            flow += add;
        }
        return {flow, cost};
    }
};

std::vector<Answer> solve_single_report_assignment_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int rb_budget
) {
    int n = static_cast<int>(options_by_user.size());
    int source = 0;
    int user_base = 1;
    int slot_base = user_base + n;
    int sink = slot_base + L;
    MinCostFlow flow(sink + 1);

    struct AssignmentArc {
        int user_id = 0;
        int edge_idx = 0;
        const TimeOption* option = nullptr;
    };
    std::vector<AssignmentArc> arcs;
    arcs.reserve(static_cast<size_t>(n) * 16);

    for (int user_id = 0; user_id < n; ++user_id) {
        flow.add_edge(source, user_base + user_id, 1, 0);
        std::array<const TimeOption*, L> best_by_slot{};
        for (const TimeOption& option : options_by_user[user_id]) {
            if (popcount(option.mask) != 1) continue;
            int slot = first_set_slot(option.mask);
            if (slot >= L) continue;
            const TimeOption* current = best_by_slot[slot];
            if (current == nullptr || std::tie(option.quality, option.period, option.offset)
                    < std::tie(current->quality, current->period, current->offset)) {
                best_by_slot[slot] = &option;
            }
        }
        for (int slot = 0; slot < L; ++slot) {
            const TimeOption* option = best_by_slot[slot];
            if (option == nullptr) continue;
            int edge_idx = flow.add_edge(user_base + user_id, slot_base + slot, 1, option->quality);
            arcs.push_back({user_id, edge_idx, option});
        }
    }
    for (int slot = 0; slot < L; ++slot) {
        flow.add_edge(slot_base + slot, sink, rb_budget, 0);
    }

    auto [scheduled, ignored_cost] = flow.min_cost_flow(source, sink, n);
    (void)ignored_cost;

    std::vector<Answer> answers(n);
    if (scheduled < n) return answers;

    std::array<std::vector<int>, L> users_by_slot;
    std::vector<const TimeOption*> assigned_options(n, nullptr);
    for (const AssignmentArc& arc : arcs) {
        const FlowEdge& edge = flow.graph[user_base + arc.user_id][arc.edge_idx];
        if (edge.cap != 0) continue;
        int slot = first_set_slot(arc.option->mask);
        users_by_slot[slot].push_back(arc.user_id);
        assigned_options[arc.user_id] = arc.option;
    }

    for (int slot = 0; slot < L; ++slot) {
        std::sort(users_by_slot[slot].begin(), users_by_slot[slot].end(), [&](int a, int b) {
            const TimeOption* option_a = assigned_options[a];
            const TimeOption* option_b = assigned_options[b];
            return std::tie(option_a->quality, a) < std::tie(option_b->quality, b);
        });
        for (int i = 0; i < static_cast<int>(users_by_slot[slot].size()); ++i) {
            int user_id = users_by_slot[slot][i];
            const TimeOption* option = assigned_options[user_id];
            answers[user_id] = Answer{i, option->period, option->offset, option->distance, option->quality};
        }
    }
    return answers;
}

std::vector<Answer> solve_single_cell_flow_budget_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int rb_budget
) {
    return solve_single_report_assignment_prepared(options_by_user, rb_budget);
}

std::vector<Answer> solve_single_cell_flow_sweep_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int extra_budgets = 5,
    int large_extra_budgets = 0
) {
    auto [lower, upper] = rb_budget_window(options_by_user, extra_budgets, large_extra_budgets);

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> candidate = solve_single_cell_flow_budget_prepared(options_by_user, rb_budget);
        if (compact_stats(candidate).unscheduled > 0) continue;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    }
    return best;
}

std::vector<Answer> solve_single_cell_auction_budget_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int rb_budget
) {
    int n = static_cast<int>(options_by_user.size());
    if (rb_budget <= 0 || rb_budget > R) return {};

    struct SlotOption {
        int slot = 0;
        const TimeOption* option = nullptr;
    };
    std::vector<std::vector<SlotOption>> arcs(n);
    for (int user_id = 0; user_id < n; ++user_id) {
        std::array<const TimeOption*, L> best_by_slot{};
        for (const TimeOption& option : options_by_user[user_id]) {
            if (popcount(option.mask) != 1) continue;
            int slot = first_set_slot(option.mask);
            if (slot < 0 || slot >= L) continue;
            const TimeOption* current = best_by_slot[slot];
            if (current == nullptr || std::tie(option.quality, option.period, option.offset)
                    < std::tie(current->quality, current->period, current->offset)) {
                best_by_slot[slot] = &option;
            }
        }
        for (int slot = 0; slot < L; ++slot) {
            if (best_by_slot[slot] != nullptr) arcs[user_id].push_back({slot, best_by_slot[slot]});
        }
        if (arcs[user_id].empty()) return {};
    }

    std::vector<double> price(L, 0.0);
    std::vector<int> assigned_slot(n, -1);
    std::vector<const TimeOption*> assigned_option(n, nullptr);
    std::array<std::vector<int>, L> slot_users;
    std::queue<int> pending;
    std::vector<uint8_t> queued(n, 1);
    for (int user_id = 0; user_id < n; ++user_id) pending.push(user_id);

    auto remove_from_slot = [&](int user_id, int slot) {
        auto& users = slot_users[slot];
        auto it = std::find(users.begin(), users.end(), user_id);
        if (it != users.end()) {
            *it = users.back();
            users.pop_back();
        }
    };

    int max_iterations = std::max(2000, n * 160);
    int assigned_count = 0;
    for (int iter = 0; iter < max_iterations && !pending.empty(); ++iter) {
        int user_id = pending.front();
        pending.pop();
        queued[user_id] = 0;

        int best_idx = -1;
        double best_score = std::numeric_limits<double>::infinity();
        double second_score = std::numeric_limits<double>::infinity();
        for (int idx = 0; idx < static_cast<int>(arcs[user_id].size()); ++idx) {
            int slot = arcs[user_id][idx].slot;
            double score = static_cast<double>(arcs[user_id][idx].option->quality) + price[slot];
            if (score < best_score) {
                second_score = best_score;
                best_score = score;
                best_idx = idx;
            } else if (score < second_score) {
                second_score = score;
            }
        }
        if (best_idx < 0) return {};

        int new_slot = arcs[user_id][best_idx].slot;
        const TimeOption* new_option = arcs[user_id][best_idx].option;
        double gap = std::isfinite(second_score) ? second_score - best_score : 1.0;
        price[new_slot] += std::max(0.25, gap + 0.25);

        if (assigned_slot[user_id] != -1) {
            remove_from_slot(user_id, assigned_slot[user_id]);
        } else {
            ++assigned_count;
        }
        assigned_slot[user_id] = new_slot;
        assigned_option[user_id] = new_option;
        slot_users[new_slot].push_back(user_id);

        while (static_cast<int>(slot_users[new_slot].size()) > rb_budget) {
            auto& users = slot_users[new_slot];
            int evict_pos = 0;
            for (int pos = 1; pos < static_cast<int>(users.size()); ++pos) {
                int a = users[pos];
                int b = users[evict_pos];
                if (std::tie(assigned_option[a]->quality, a) > std::tie(assigned_option[b]->quality, b)) {
                    evict_pos = pos;
                }
            }
            int evicted = users[evict_pos];
            users[evict_pos] = users.back();
            users.pop_back();
            assigned_slot[evicted] = -1;
            assigned_option[evicted] = nullptr;
            --assigned_count;
            if (!queued[evicted]) {
                pending.push(evicted);
                queued[evicted] = 1;
            }
        }
    }

    if (assigned_count < n) return {};

    std::vector<Answer> answers(n);
    for (int slot = 0; slot < L; ++slot) {
        auto& users = slot_users[slot];
        std::sort(users.begin(), users.end(), [&](int a, int b) {
            return std::tie(assigned_option[a]->quality, a) < std::tie(assigned_option[b]->quality, b);
        });
        if (static_cast<int>(users.size()) > rb_budget) return {};
        for (int rb = 0; rb < static_cast<int>(users.size()); ++rb) {
            int user_id = users[rb];
            const TimeOption* option = assigned_option[user_id];
            if (option == nullptr) return {};
            answers[user_id] = Answer{rb, option->period, option->offset, option->distance, option->quality};
        }
    }
    return answers;
}

std::vector<Answer> solve_single_cell_auction_sweep_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int extra_budgets = 5,
    int large_extra_budgets = 0
) {
    auto [lower, upper] = rb_budget_window(options_by_user, extra_budgets, large_extra_budgets);

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> candidate = solve_single_cell_auction_budget_prepared(options_by_user, rb_budget);
        if (candidate.empty() || compact_stats(candidate).unscheduled > 0) continue;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    }
    return best;
}

std::vector<Answer> polish_fixed_rb_quality_dynamic(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int passes,
    int order_mode,
    int seed
);

std::vector<Answer> polish_ejection_chain_quality(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int passes,
    int max_blockers
);

std::vector<Answer> polish_tabu_fixed_rb_quality(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int iterations,
    int tenure,
    int option_limit,
    int user_limit
);

std::vector<Answer> solve_single_cell_auction_upgrade_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool strong = false
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, ignored_upper] = rb_budget_window(options_by_user, strong ? 8 : 5, 0);
    (void)ignored_upper;
    int upper = std::min(R, lower + (strong ? 8 : 5));

    std::vector<Answer> best = solve_single_cell_auction_sweep_prepared(options_by_user, strong ? 8 : 5, 0);
    int64_t best_obj = best.empty() ? std::numeric_limits<int64_t>::max() : objective_value(best);
    auto consider = [&](std::vector<Answer> candidate) {
        if (candidate.empty()) return;
        if (compact_stats(candidate).unscheduled > 0) return;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> start = solve_single_cell_auction_budget_prepared(options_by_user, rb_budget);
        if (start.empty() || compact_stats(start).unscheduled > 0) continue;
        consider(start);

        std::vector<Answer> current = polish_fixed_rb_quality_dynamic(
            options_by_user,
            start,
            rb_budget,
            n <= 1500 ? 5 : 2,
            1,
            43000 + rb_budget
        );
        consider(current);

        current = polish_ejection_chain_quality(
            options_by_user,
            current,
            rb_budget,
            n <= 1500 ? 3 : 1,
            n <= 1500 ? 3 : 2
        );
        consider(current);

        if (strong && n <= 1500) {
            current = polish_tabu_fixed_rb_quality(
                options_by_user,
                current,
                rb_budget,
                40,
                7,
                18,
                420
            );
            consider(current);
        }
    }
    return best;
}

std::vector<Answer> solve_flow320_budget_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int rb_budget
) {
    int n = static_cast<int>(options_by_user.size());
    if (rb_budget <= 0 || rb_budget > R) return {};

    int source = 0;
    int user_base = 1;
    int slot_base = user_base + n;
    int sink = slot_base + L;
    MinCostFlow flow(sink + 1);

    struct AssignmentArc {
        int user_id = 0;
        int edge_idx = 0;
        const TimeOption* option = nullptr;
    };
    std::vector<AssignmentArc> arcs;
    arcs.reserve(static_cast<size_t>(n) * 64);

    for (int user_id = 0; user_id < n; ++user_id) {
        flow.add_edge(source, user_base + user_id, 1, 0);
        std::array<const TimeOption*, L> best_by_slot{};
        for (const TimeOption& option : options_by_user[user_id]) {
            if (option.period != 320 || popcount(option.mask) != 1) continue;
            int slot = first_set_slot(option.mask);
            if (slot < 0 || slot >= L) continue;
            const TimeOption* current = best_by_slot[slot];
            if (current == nullptr || std::tie(option.quality, option.offset)
                    < std::tie(current->quality, current->offset)) {
                best_by_slot[slot] = &option;
            }
        }
        for (int slot = 0; slot < L; ++slot) {
            const TimeOption* option = best_by_slot[slot];
            if (option == nullptr) continue;
            int edge_idx = flow.add_edge(user_base + user_id, slot_base + slot, 1, option->quality);
            arcs.push_back({user_id, edge_idx, option});
        }
    }
    for (int slot = 0; slot < L; ++slot) {
        flow.add_edge(slot_base + slot, sink, rb_budget, 0);
    }

    auto [scheduled, ignored_cost] = flow.min_cost_flow(source, sink, n);
    (void)ignored_cost;
    if (scheduled < n) return {};

    std::array<std::vector<int>, L> users_by_slot;
    std::vector<const TimeOption*> assigned_options(n, nullptr);
    for (const AssignmentArc& arc : arcs) {
        const FlowEdge& edge = flow.graph[user_base + arc.user_id][arc.edge_idx];
        if (edge.cap != 0) continue;
        int slot = first_set_slot(arc.option->mask);
        if (slot < 0 || slot >= L) return {};
        users_by_slot[slot].push_back(arc.user_id);
        assigned_options[arc.user_id] = arc.option;
    }

    std::vector<Answer> answers(n);
    for (int slot = 0; slot < L; ++slot) {
        std::sort(users_by_slot[slot].begin(), users_by_slot[slot].end(), [&](int a, int b) {
            const TimeOption* option_a = assigned_options[a];
            const TimeOption* option_b = assigned_options[b];
            return std::tie(option_a->quality, a) < std::tie(option_b->quality, b);
        });
        if (static_cast<int>(users_by_slot[slot].size()) > rb_budget) return {};
        for (int rb = 0; rb < static_cast<int>(users_by_slot[slot].size()); ++rb) {
            int user_id = users_by_slot[slot][rb];
            const TimeOption* option = assigned_options[user_id];
            if (option == nullptr) return {};
            answers[user_id] = Answer{rb, option->period, option->offset, option->distance, option->quality};
        }
    }
    return answers;
}

std::vector<Answer> solve_flow320_sweep_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int extra_budgets = 8
) {
    auto [lower, ignored_upper] = rb_budget_window(options_by_user, extra_budgets, 0);
    (void)ignored_upper;
    int upper = std::min(R, lower + std::max(0, extra_budgets));

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> candidate = solve_flow320_budget_prepared(options_by_user, rb_budget);
        if (candidate.empty()) continue;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    }
    return best;
}

std::vector<Answer> solve_assignment_sweep_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    return solve_single_cell_flow_sweep_prepared(options_by_user, 5, 0);
}

std::vector<Answer> polish_fixed_rb_quality(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int passes
) {
    std::vector<Mask> occupied(R);
    for (const Answer& answer : answers) {
        if (answer.period == 640) continue;
        add_mask(occupied[answer.rb], answer_mask(answer));
    }

    struct Proposal {
        int user_id = 0;
        int gain = 0;
        Answer answer;
        const TimeOption* option = nullptr;
    };

    int n = static_cast<int>(answers.size());
    for (int pass = 0; pass < passes; ++pass) {
        std::vector<Proposal> proposals;
        proposals.reserve(n);

        for (int user_id = 0; user_id < n; ++user_id) {
            if (answers[user_id].period == 640) continue;
            Mask old_mask = answer_mask(answers[user_id]);
            remove_mask(occupied[answers[user_id].rb], old_mask);

            bool found = false;
            Answer best_answer;
            const TimeOption* best_option = nullptr;
            std::tuple<int, int, int, int, int> best_tuple{};

            for (const TimeOption& option : options_by_user[user_id]) {
                if (option.quality >= answers[user_id].quality) continue;
                for (int rb = 0; rb < rb_budget; ++rb) {
                    if (intersects(occupied[rb], option.mask)) continue;
                    auto candidate_tuple = std::make_tuple(
                        option.quality,
                        popcount(option.mask),
                        rb == answers[user_id].rb ? 0 : 1,
                        rb,
                        option.offset
                    );
                    if (!found || candidate_tuple < best_tuple) {
                        found = true;
                        best_tuple = candidate_tuple;
                        best_option = &option;
                        best_answer.rb = rb;
                        best_answer.period = option.period;
                        best_answer.offset = option.offset;
                        best_answer.distance = option.distance;
                        best_answer.quality = option.quality;
                    }
                }
            }

            add_mask(occupied[answers[user_id].rb], old_mask);
            if (found && best_option != nullptr) {
                proposals.push_back({user_id, answers[user_id].quality - best_answer.quality, best_answer, best_option});
            }
        }

        if (proposals.empty()) break;
        std::sort(proposals.begin(), proposals.end(), [](const Proposal& a, const Proposal& b) {
            return std::make_tuple(a.gain, -popcount(a.option->mask), -a.answer.quality)
                > std::make_tuple(b.gain, -popcount(b.option->mask), -b.answer.quality);
        });

        int applied = 0;
        for (const Proposal& proposal : proposals) {
            const Answer& old_answer = answers[proposal.user_id];
            if (old_answer.period == 640) continue;
            Mask old_mask = answer_mask(old_answer);
            remove_mask(occupied[old_answer.rb], old_mask);
            if (!intersects(occupied[proposal.answer.rb], proposal.option->mask)
                && proposal.answer.quality < old_answer.quality) {
                answers[proposal.user_id] = proposal.answer;
                add_mask(occupied[proposal.answer.rb], proposal.option->mask);
                ++applied;
            } else {
                add_mask(occupied[old_answer.rb], old_mask);
            }
        }
        if (applied == 0) break;
    }
    return answers;
}

std::vector<Answer> polish_fixed_rb_quality_dynamic(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int passes,
    int order_mode,
    int seed
) {
    std::vector<Mask> occupied(R);
    for (const Answer& answer : answers) {
        if (answer.period == 640) continue;
        add_mask(occupied[answer.rb], answer_mask(answer));
    }
    std::mt19937 rng(seed);
    int n = static_cast<int>(answers.size());

    for (int pass = 0; pass < passes; ++pass) {
        std::vector<int> order;
        order.reserve(n);
        for (int user_id = 0; user_id < n; ++user_id) {
            if (answers[user_id].period != 640) order.push_back(user_id);
        }

        if (order_mode == 0) {
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return std::tie(answers[a].quality, a) > std::tie(answers[b].quality, b);
            });
        } else if (order_mode == 1) {
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                auto potential = [&](int user_id) {
                    int best_quality = answers[user_id].quality;
                    for (const TimeOption& option : options_by_user[user_id]) {
                        best_quality = std::min(best_quality, option.quality);
                    }
                    return answers[user_id].quality - best_quality;
                };
                return std::make_tuple(potential(a), answers[a].quality, a)
                    > std::make_tuple(potential(b), answers[b].quality, b);
            });
        } else {
            std::shuffle(order.begin(), order.end(), rng);
        }

        int applied = 0;
        for (int user_id : order) {
            if (answers[user_id].period == 640) continue;
            Answer old_answer = answers[user_id];
            Mask old_mask = answer_mask(old_answer);
            remove_mask(occupied[old_answer.rb], old_mask);

            bool found = false;
            Answer best_answer;
            const TimeOption* best_option = nullptr;
            std::tuple<int, int, int, int, int> best_tuple{};

            for (const TimeOption& option : options_by_user[user_id]) {
                if (option.quality >= old_answer.quality) continue;
                for (int rb = 0; rb < rb_budget; ++rb) {
                    if (intersects(occupied[rb], option.mask)) continue;
                    auto candidate_tuple = std::make_tuple(
                        option.quality,
                        popcount(option.mask),
                        rb == old_answer.rb ? 0 : 1,
                        rb,
                        option.offset
                    );
                    if (!found || candidate_tuple < best_tuple) {
                        found = true;
                        best_tuple = candidate_tuple;
                        best_option = &option;
                        best_answer.rb = rb;
                        best_answer.period = option.period;
                        best_answer.offset = option.offset;
                        best_answer.distance = option.distance;
                        best_answer.quality = option.quality;
                    }
                }
            }

            if (found && best_option != nullptr) {
                answers[user_id] = best_answer;
                add_mask(occupied[best_answer.rb], best_option->mask);
                ++applied;
            } else {
                add_mask(occupied[old_answer.rb], old_mask);
            }
        }
        if (applied == 0) break;
    }
    return answers;
}

std::vector<Answer> polish_ejection_chain_quality(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int passes,
    int max_blockers
) {
    int n = static_cast<int>(answers.size());
    std::vector<std::array<int, L>> owner(rb_budget);
    auto rebuild_owner = [&]() {
        for (int rb = 0; rb < rb_budget; ++rb) owner[rb].fill(-1);
        for (int user_id = 0; user_id < n; ++user_id) {
            if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
            for_each_slot_in_mask(answer_mask(answers[user_id]), [&](int slot) {
                owner[answers[user_id].rb][slot] = user_id;
            });
        }
    };
    rebuild_owner();

    auto best_repair_choice = [&](const std::vector<Mask>& occupied, int user_id) {
        Choice best;
        for (const TimeOption& option : options_by_user[user_id]) {
            for (int rb = 0; rb < rb_budget; ++rb) {
                if (intersects(occupied[rb], option.mask)) continue;
                auto candidate_tuple = std::make_tuple(option.quality, popcount(option.mask), rb, option.offset);
                auto best_tuple = std::make_tuple(best.quality, best.option ? popcount(best.option->mask) : L, best.rb, best.offset);
                if (!best.found || candidate_tuple < best_tuple) {
                    best.found = true;
                    best.rb = rb;
                    best.period = option.period;
                    best.offset = option.offset;
                    best.distance = option.distance;
                    best.quality = option.quality;
                    best.option = &option;
                }
            }
        }
        return best;
    };

    for (int pass = 0; pass < passes; ++pass) {
        struct Move {
            int user_id = 0;
            int rb = 0;
            const TimeOption* option = nullptr;
            std::vector<int> blockers;
            int optimistic_gain = 0;
        };
        std::vector<Move> moves;
        moves.reserve(n * 4);

        for (int user_id = 0; user_id < n; ++user_id) {
            if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
            for (const TimeOption& option : options_by_user[user_id]) {
                if (option.quality >= answers[user_id].quality) continue;
                int direct_gain = answers[user_id].quality - option.quality;
                if (direct_gain <= 0) continue;
                for (int rb = 0; rb < rb_budget; ++rb) {
                    std::vector<int> blockers;
                    bool own_overlap_only = true;
                    for_each_slot_in_mask(option.mask, [&](int slot) {
                        int blocker = owner[rb][slot];
                        if (blocker >= 0 && blocker != user_id) {
                            own_overlap_only = false;
                            if (std::find(blockers.begin(), blockers.end(), blocker) == blockers.end()) {
                                blockers.push_back(blocker);
                            }
                        }
                    });
                    (void)own_overlap_only;
                    if (static_cast<int>(blockers.size()) > max_blockers) continue;
                    int blocker_quality = 0;
                    for (int blocker : blockers) blocker_quality += answers[blocker].quality;
                    int optimistic_gain = direct_gain - static_cast<int>(blockers.size()) * 8;
                    if (optimistic_gain <= 0 && !blockers.empty()) continue;
                    moves.push_back({user_id, rb, &option, blockers, optimistic_gain + blocker_quality / 10000});
                    if (static_cast<int>(moves.size()) > n * 24) break;
                }
                if (static_cast<int>(moves.size()) > n * 24) break;
            }
        }
        if (moves.empty()) break;

        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return std::make_tuple(a.optimistic_gain, -static_cast<int>(a.blockers.size()), -a.option->quality)
                > std::make_tuple(b.optimistic_gain, -static_cast<int>(b.blockers.size()), -b.option->quality);
        });

        bool improved = false;
        int attempts = std::min<int>(moves.size(), 2500);
        for (int move_idx = 0; move_idx < attempts && !improved; ++move_idx) {
            const Move& move = moves[move_idx];
            std::vector<int> removed = move.blockers;
            removed.push_back(move.user_id);
            std::sort(removed.begin(), removed.end());
            removed.erase(std::unique(removed.begin(), removed.end()), removed.end());

            std::vector<Mask> occupied(rb_budget);
            for (int user_id = 0; user_id < n; ++user_id) {
                if (std::binary_search(removed.begin(), removed.end(), user_id)) continue;
                if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
                add_mask(occupied[answers[user_id].rb], answer_mask(answers[user_id]));
            }
            if (intersects(occupied[move.rb], move.option->mask)) continue;

            std::vector<Answer> trial = answers;
            int old_quality = 0;
            for (int user_id : removed) old_quality += answers[user_id].quality;

            trial[move.user_id] = Answer{
                move.rb,
                move.option->period,
                move.option->offset,
                move.option->distance,
                move.option->quality
            };
            add_mask(occupied[move.rb], move.option->mask);
            int new_quality = move.option->quality;

            std::vector<int> repair_users = move.blockers;
            std::sort(repair_users.begin(), repair_users.end(), [&](int a, int b) {
                return std::tie(answers[a].quality, a) > std::tie(answers[b].quality, b);
            });

            bool ok = true;
            for (int blocker : repair_users) {
                Choice choice = best_repair_choice(occupied, blocker);
                if (!choice.found) {
                    ok = false;
                    break;
                }
                trial[blocker] = choice_to_answer(choice);
                add_mask(occupied[choice.rb], choice.option->mask);
                new_quality += choice.quality;
                if (new_quality >= old_quality) {
                    ok = false;
                    break;
                }
            }
            if (!ok || new_quality >= old_quality) continue;

            answers = std::move(trial);
            rebuild_owner();
            improved = true;
        }
        if (!improved) break;
    }
    return answers;
}

std::vector<Answer> polish_tabu_fixed_rb_quality(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int iterations,
    int tenure,
    int option_limit,
    int user_limit
) {
    int n = static_cast<int>(answers.size());
    if (rb_budget <= 0 || rb_budget > R || n == 0) return answers;

    std::vector<Mask> occupied(rb_budget);
    int current_quality = 0;
    for (const Answer& answer : answers) {
        current_quality += answer.quality;
        if (answer.period == 640 || answer.rb >= rb_budget) continue;
        add_mask(occupied[answer.rb], answer_mask(answer));
    }

    std::vector<Answer> best = answers;
    int best_quality = current_quality;

    struct RankedUser {
        int user_id = 0;
        int potential = 0;
        int quality = 0;
    };
    std::vector<RankedUser> ranked;
    ranked.reserve(n);
    for (int user_id = 0; user_id < n; ++user_id) {
        if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
        int best_option_quality = answers[user_id].quality;
        int inspected = 0;
        for (const TimeOption& option : options_by_user[user_id]) {
            best_option_quality = std::min(best_option_quality, option.quality);
            if (++inspected >= option_limit) break;
        }
        ranked.push_back({user_id, answers[user_id].quality - best_option_quality, answers[user_id].quality});
    }
    std::sort(ranked.begin(), ranked.end(), [](const RankedUser& a, const RankedUser& b) {
        return std::make_tuple(a.potential, a.quality, -a.user_id)
            > std::make_tuple(b.potential, b.quality, -b.user_id);
    });
    if (static_cast<int>(ranked.size()) > user_limit) ranked.resize(user_limit);

    std::vector<int> tabu_until(n, 0);
    int no_improve = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        int best_user = -1;
        Answer best_answer;
        const TimeOption* best_option = nullptr;
        int best_delta = std::numeric_limits<int>::max();
        std::tuple<int, int, int, int, int> best_tuple{};

        for (const RankedUser& ranked_user : ranked) {
            int user_id = ranked_user.user_id;
            if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;

            Answer old_answer = answers[user_id];
            Mask old_mask = answer_mask(old_answer);
            remove_mask(occupied[old_answer.rb], old_mask);

            int inspected = 0;
            for (const TimeOption& option : options_by_user[user_id]) {
                if (++inspected > option_limit) break;
                for (int rb = 0; rb < rb_budget; ++rb) {
                    if (rb == old_answer.rb && option.period == old_answer.period && option.offset == old_answer.offset) {
                        continue;
                    }
                    if (intersects(occupied[rb], option.mask)) continue;

                    int delta = option.quality - old_answer.quality;
                    bool aspiration = current_quality + delta < best_quality;
                    if (iter < tabu_until[user_id] && !aspiration) continue;
                    int allowed_worsening = 8 + std::min(20, no_improve / 2);
                    if (delta > allowed_worsening && !aspiration) continue;

                    auto candidate_tuple = std::make_tuple(
                        delta,
                        option.quality,
                        rb == old_answer.rb ? 0 : 1,
                        rb,
                        option.offset
                    );
                    if (best_user < 0 || candidate_tuple < best_tuple) {
                        best_user = user_id;
                        best_answer = Answer{rb, option.period, option.offset, option.distance, option.quality};
                        best_option = &option;
                        best_delta = delta;
                        best_tuple = candidate_tuple;
                    }
                }
            }

            add_mask(occupied[old_answer.rb], old_mask);
        }

        if (best_user < 0 || best_option == nullptr) break;

        Answer old_answer = answers[best_user];
        remove_mask(occupied[old_answer.rb], answer_mask(old_answer));
        answers[best_user] = best_answer;
        add_mask(occupied[best_answer.rb], best_option->mask);
        current_quality += best_delta;
        tabu_until[best_user] = iter + tenure + (best_user % 5);

        if (current_quality < best_quality) {
            best_quality = current_quality;
            best = answers;
            no_improve = 0;
        } else {
            ++no_improve;
        }

        if (no_improve > std::max(12, iterations / 3)) break;
    }

    return objective_value(best) < objective_value(answers) ? best : answers;
}

std::vector<Answer> polish_beam_fixed_set_quality(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    int rb_budget,
    int neighborhood_size,
    int beam_width,
    int option_limit
) {
    int n = static_cast<int>(answers.size());
    struct CandidateUser {
        int user_id = 0;
        int potential = 0;
        int feasible_count = 0;
    };
    std::vector<CandidateUser> candidates;
    candidates.reserve(n);
    for (int user_id = 0; user_id < n; ++user_id) {
        if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
        int best_quality = answers[user_id].quality;
        int feasible_count = 0;
        for (const TimeOption& option : options_by_user[user_id]) {
            if (option.quality < answers[user_id].quality) {
                best_quality = std::min(best_quality, option.quality);
                ++feasible_count;
            }
        }
        int potential = answers[user_id].quality - best_quality;
        if (potential > 0) candidates.push_back({user_id, potential, feasible_count});
    }
    if (candidates.empty()) return answers;
    std::sort(candidates.begin(), candidates.end(), [](const CandidateUser& a, const CandidateUser& b) {
        return std::make_tuple(a.potential, -a.feasible_count, a.user_id)
            > std::make_tuple(b.potential, -b.feasible_count, b.user_id);
    });
    std::vector<std::array<int, L>> current_owner(rb_budget);
    for (int rb = 0; rb < rb_budget; ++rb) current_owner[rb].fill(-1);
    for (int user_id = 0; user_id < n; ++user_id) {
        if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
        for_each_slot_in_mask(answer_mask(answers[user_id]), [&](int slot) {
            current_owner[answers[user_id].rb][slot] = user_id;
        });
    }

    std::vector<int> selected;
    selected.reserve(neighborhood_size);
    auto add_selected = [&](int user_id) {
        if (user_id < 0) return;
        if (std::find(selected.begin(), selected.end(), user_id) != selected.end()) return;
        if (static_cast<int>(selected.size()) < neighborhood_size) selected.push_back(user_id);
    };

    int seed_count = std::min<int>(candidates.size(), std::max(8, neighborhood_size / 3));
    for (int i = 0; i < seed_count; ++i) add_selected(candidates[i].user_id);
    for (int i = 0; i < seed_count && static_cast<int>(selected.size()) < neighborhood_size; ++i) {
        int user_id = candidates[i].user_id;
        int inspected_options = 0;
        for (const TimeOption& option : options_by_user[user_id]) {
            if (option.quality >= answers[user_id].quality) continue;
            for (int rb = 0; rb < rb_budget; ++rb) {
                for_each_slot_in_mask(option.mask, [&](int slot) {
                    int blocker = current_owner[rb][slot];
                    if (blocker >= 0 && blocker != user_id) add_selected(blocker);
                });
                if (static_cast<int>(selected.size()) >= neighborhood_size) break;
            }
            if (++inspected_options >= 4 || static_cast<int>(selected.size()) >= neighborhood_size) break;
        }
    }
    for (const CandidateUser& candidate : candidates) {
        if (static_cast<int>(selected.size()) >= neighborhood_size) break;
        add_selected(candidate.user_id);
    }
    std::sort(selected.begin(), selected.end());

    std::vector<Mask> fixed_occupied(rb_budget);
    int old_quality = 0;
    for (int user_id = 0; user_id < n; ++user_id) {
        if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
        if (std::binary_search(selected.begin(), selected.end(), user_id)) {
            old_quality += answers[user_id].quality;
            continue;
        }
        add_mask(fixed_occupied[answers[user_id].rb], answer_mask(answers[user_id]));
    }

    struct UserChoices {
        int user_id = 0;
        std::vector<Choice> choices;
    };
    std::vector<UserChoices> ordered;
    ordered.reserve(selected.size());
    for (int user_id : selected) {
        std::vector<Choice> choices;
        for (const TimeOption& option : options_by_user[user_id]) {
            for (int rb = 0; rb < rb_budget; ++rb) {
                if (intersects(fixed_occupied[rb], option.mask)) continue;
                Choice choice;
                choice.found = true;
                choice.rb = rb;
                choice.period = option.period;
                choice.offset = option.offset;
                choice.distance = option.distance;
                choice.quality = option.quality;
                choice.option = &option;
                choices.push_back(choice);
            }
        }
        std::sort(choices.begin(), choices.end(), [](const Choice& a, const Choice& b) {
            return std::make_tuple(a.quality, popcount(a.option->mask), a.rb, a.offset)
                < std::make_tuple(b.quality, popcount(b.option->mask), b.rb, b.offset);
        });
        if (static_cast<int>(choices.size()) > option_limit) choices.resize(option_limit);
        if (choices.empty()) return answers;
        ordered.push_back({user_id, std::move(choices)});
    }
    std::sort(ordered.begin(), ordered.end(), [](const UserChoices& a, const UserChoices& b) {
        return std::make_tuple(a.choices.size(), a.choices.front().quality, a.user_id)
            < std::make_tuple(b.choices.size(), b.choices.front().quality, b.user_id);
    });

    struct BeamState {
        std::vector<Mask> occupied;
        int quality = 0;
        std::vector<Answer> placed;
    };
    std::vector<BeamState> beam;
    beam.push_back({fixed_occupied, 0, {}});

    for (const UserChoices& user_choices : ordered) {
        std::vector<BeamState> next;
        next.reserve(static_cast<size_t>(beam_width) * std::min<int>(option_limit, user_choices.choices.size()));
        for (const BeamState& state : beam) {
            for (const Choice& choice : user_choices.choices) {
                if (intersects(state.occupied[choice.rb], choice.option->mask)) continue;
                BeamState child;
                child.occupied = state.occupied;
                add_mask(child.occupied[choice.rb], choice.option->mask);
                child.quality = state.quality + choice.quality;
                child.placed = state.placed;
                child.placed.push_back(choice_to_answer(choice));
                next.push_back(std::move(child));
            }
        }
        if (next.empty()) return answers;
        std::sort(next.begin(), next.end(), [](const BeamState& a, const BeamState& b) {
            return a.quality < b.quality;
        });
        if (static_cast<int>(next.size()) > beam_width) next.resize(beam_width);
        beam = std::move(next);
    }

    const BeamState& best_state = beam.front();
    if (best_state.quality >= old_quality) return answers;

    std::vector<Answer> trial = answers;
    for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
        trial[ordered[i].user_id] = best_state.placed[i];
    }
    return trial;
}

std::vector<int> hungarian_min_assignment(const std::vector<std::vector<int>>& cost, int impossible_cost) {
    int n = static_cast<int>(cost.size());
    if (n == 0) return {};
    int m = static_cast<int>(cost[0].size());
    if (n > m) return {};

    const int64_t INF = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> u(n + 1), v(m + 1);
    std::vector<int> p(m + 1), way(m + 1);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<int64_t> minv(m + 1, INF);
        std::vector<uint8_t> used(m + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0];
            int64_t delta = INF;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                int64_t cur = int64_t(cost[i0 - 1][j - 1]) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if (delta >= INF / 2) return {};
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= m; ++j) {
        if (p[j] > 0) assignment[p[j] - 1] = j - 1;
    }
    for (int i = 0; i < n; ++i) {
        if (assignment[i] < 0 || cost[i][assignment[i]] >= impossible_cost) return {};
    }
    return assignment;
}

std::vector<Answer> polish_fixed_pattern_assignment(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    int max_users = 900
) {
    int n = static_cast<int>(answers.size());
    if (n == 0 || n > max_users) return answers;

    struct Position {
        int rb = 0;
        int period = 640;
        int offset = 0;
    };

    std::vector<Position> positions;
    positions.reserve(n);
    for (const Answer& answer : answers) {
        positions.push_back({answer.rb, answer.period, answer.offset});
    }

    constexpr int IMPOSSIBLE = 100000000;
    std::vector<std::vector<int>> cost(n, std::vector<int>(n, IMPOSSIBLE));
    std::vector<std::vector<int>> distance(n, std::vector<int>(n, 640));
    for (int user_id = 0; user_id < n; ++user_id) {
        for (int pos_id = 0; pos_id < n; ++pos_id) {
            const Position& position = positions[pos_id];
            if (position.period == 640) {
                cost[user_id][pos_id] = 1280;
                distance[user_id][pos_id] = 640;
                continue;
            }
            const TimeOption* option = find_time_option(
                options_by_user,
                user_id,
                position.period,
                position.offset
            );
            if (option == nullptr) continue;
            cost[user_id][pos_id] = option->quality;
            distance[user_id][pos_id] = option->distance;
        }
    }

    std::vector<int> assignment = hungarian_min_assignment(cost, IMPOSSIBLE);
    if (assignment.empty()) return answers;

    std::vector<Answer> trial(n);
    for (int user_id = 0; user_id < n; ++user_id) {
        const Position& position = positions[assignment[user_id]];
        trial[user_id].rb = position.rb;
        trial[user_id].period = position.period;
        trial[user_id].offset = position.offset;
        trial[user_id].distance = distance[user_id][assignment[user_id]];
        trial[user_id].quality = cost[user_id][assignment[user_id]];
    }
    return objective_value(trial) < objective_value(answers) ? trial : answers;
}

std::vector<Answer> polish_fixed_pattern_flow(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    int max_users = 10000
) {
    int n = static_cast<int>(answers.size());
    if (n == 0 || n > max_users) return answers;

    struct PatternCategory {
        int period = 640;
        int offset = 0;
        std::vector<Answer> positions;
    };

    std::vector<PatternCategory> categories;
    categories.reserve(128);
    for (const Answer& answer : answers) {
        int category_id = -1;
        for (int i = 0; i < static_cast<int>(categories.size()); ++i) {
            if (categories[i].period == answer.period && categories[i].offset == answer.offset) {
                category_id = i;
                break;
            }
        }
        if (category_id < 0) {
            PatternCategory category;
            category.period = answer.period;
            category.offset = answer.offset;
            categories.push_back(std::move(category));
            category_id = static_cast<int>(categories.size()) - 1;
        }
        categories[category_id].positions.push_back(answer);
    }

    int source = 0;
    int user_base = 1;
    int category_base = user_base + n;
    int sink = category_base + static_cast<int>(categories.size());
    MinCostFlow flow(sink + 1);

    struct AssignmentArc {
        int user_id = 0;
        int category_id = 0;
        int edge_idx = 0;
        int distance = 640;
        int quality = 1280;
    };
    std::vector<AssignmentArc> arcs;
    arcs.reserve(static_cast<size_t>(n) * categories.size());

    for (int user_id = 0; user_id < n; ++user_id) {
        flow.add_edge(source, user_base + user_id, 1, 0);
        for (int category_id = 0; category_id < static_cast<int>(categories.size()); ++category_id) {
            const PatternCategory& category = categories[category_id];
            int distance = 640;
            int quality = 1280;
            if (category.period == 640) {
                distance = 640;
                quality = 1280;
            } else {
                const TimeOption* option = find_time_option(
                    options_by_user,
                    user_id,
                    category.period,
                    category.offset
                );
                if (option == nullptr) continue;
                distance = option->distance;
                quality = option->quality;
            }
            int edge_idx = flow.add_edge(
                user_base + user_id,
                category_base + category_id,
                1,
                quality
            );
            arcs.push_back({user_id, category_id, edge_idx, distance, quality});
        }
    }

    for (int category_id = 0; category_id < static_cast<int>(categories.size()); ++category_id) {
        flow.add_edge(
            category_base + category_id,
            sink,
            static_cast<int>(categories[category_id].positions.size()),
            0
        );
    }

    auto [assigned, total_cost] = flow.min_cost_flow(source, sink, n);
    (void)total_cost;
    if (assigned < n) return answers;

    std::vector<Answer> trial(n);
    std::vector<uint8_t> user_assigned(n, 0);
    std::vector<int> next_position(categories.size(), 0);
    for (const AssignmentArc& arc : arcs) {
        const FlowEdge& edge = flow.graph[user_base + arc.user_id][arc.edge_idx];
        if (edge.cap != 0) continue;
        int pos_idx = next_position[arc.category_id]++;
        if (pos_idx >= static_cast<int>(categories[arc.category_id].positions.size())) return answers;
        const Answer& position = categories[arc.category_id].positions[pos_idx];
        Answer assigned_answer;
        assigned_answer.rb = position.rb;
        assigned_answer.period = position.period;
        assigned_answer.offset = position.offset;
        assigned_answer.distance = arc.distance;
        assigned_answer.quality = arc.quality;
        trial[arc.user_id] = assigned_answer;
        user_assigned[arc.user_id] = 1;
    }

    for (uint8_t value : user_assigned) {
        if (!value) return answers;
    }
    return objective_value(trial) < objective_value(answers) ? trial : answers;
}

struct BinUsers {
    int rb = 0;
    int offset = 0;
    std::vector<int> users;
};

std::vector<BinUsers> collect_period_bins(const std::vector<Answer>& answers, int target_period) {
    std::array<std::vector<int>, R * 160> bins;
    std::array<uint8_t, R * 160> has_other{};
    for (int user_id = 0; user_id < static_cast<int>(answers.size()); ++user_id) {
        const Answer& answer = answers[user_id];
        if (answer.period != 160 && answer.period != 320) continue;
        int offset = answer.offset % 160;
        int key = answer.rb * 160 + offset;
        if (answer.period == target_period) {
            bins[key].push_back(user_id);
        } else {
            has_other[key] = 1;
        }
    }

    std::vector<BinUsers> result;
    for (int key = 0; key < R * 160; ++key) {
        if (bins[key].empty() || has_other[key]) continue;
        if (target_period == 160 && bins[key].size() != 1) continue;
        if (target_period == 320 && bins[key].size() != 2) continue;
        BinUsers item;
        item.rb = key / 160;
        item.offset = key % 160;
        item.users = bins[key];
        std::sort(item.users.begin(), item.users.end(), [&](int a, int b) {
            return answers[a].offset < answers[b].offset;
        });
        result.push_back(std::move(item));
    }
    return result;
}

bool best_three_user_role_assignment(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::array<int, 3>& users,
    const std::array<Answer, 3>& positions,
    std::array<Answer, 3>& assigned,
    int& best_quality
) {
    std::array<int, 3> perm{0, 1, 2};
    bool found = false;
    best_quality = std::numeric_limits<int>::max();

    do {
        int total = 0;
        std::array<Answer, 3> candidate;
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            int user_id = users[i];
            const Answer& position = positions[perm[i]];
            const TimeOption* option = find_time_option(
                options_by_user,
                user_id,
                position.period,
                position.offset
            );
            if (option == nullptr) {
                ok = false;
                break;
            }
            candidate[i].rb = position.rb;
            candidate[i].period = position.period;
            candidate[i].offset = position.offset;
            candidate[i].distance = option->distance;
            candidate[i].quality = option->quality;
            total += option->quality;
        }
        if (ok && total < best_quality) {
            found = true;
            best_quality = total;
            assigned = candidate;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    return found;
}

bool best_small_role_assignment(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<int>& users,
    const std::vector<Answer>& positions,
    std::vector<Answer>& assigned,
    int& best_quality
) {
    int n = static_cast<int>(users.size());
    if (n == 0 || n != static_cast<int>(positions.size()) || n > 10) return false;

    int states = 1 << n;
    const int INF = std::numeric_limits<int>::max() / 4;
    std::vector<std::vector<int>> cost(n, std::vector<int>(n, INF));
    std::vector<std::vector<int>> distance(n, std::vector<int>(n, 640));
    for (int user_pos = 0; user_pos < n; ++user_pos) {
        int user_id = users[user_pos];
        for (int pos_id = 0; pos_id < n; ++pos_id) {
            const Answer& position = positions[pos_id];
            const TimeOption* option = find_time_option(
                options_by_user,
                user_id,
                position.period,
                position.offset
            );
            if (option == nullptr) continue;
            cost[user_pos][pos_id] = option->quality;
            distance[user_pos][pos_id] = option->distance;
        }
    }

    std::vector<int> dp(states, INF), prev_state(states, -1), prev_pos(states, -1);
    dp[0] = 0;

    for (int mask = 0; mask < states; ++mask) {
        if (dp[mask] >= INF) continue;
        int user_pos = __builtin_popcount(static_cast<unsigned>(mask));
        if (user_pos >= n) continue;
        for (int pos_id = 0; pos_id < n; ++pos_id) {
            if (mask & (1 << pos_id)) continue;
            if (cost[user_pos][pos_id] >= INF) continue;
            int next_mask = mask | (1 << pos_id);
            int next_quality = dp[mask] + cost[user_pos][pos_id];
            if (next_quality < dp[next_mask]) {
                dp[next_mask] = next_quality;
                prev_state[next_mask] = mask;
                prev_pos[next_mask] = pos_id;
            }
        }
    }

    int full = states - 1;
    if (dp[full] >= INF) return false;
    best_quality = dp[full];

    assigned.assign(n, Answer{});
    int mask = full;
    for (int user_pos = n - 1; user_pos >= 0; --user_pos) {
        int pos_id = prev_pos[mask];
        if (pos_id < 0) return false;
        const Answer& position = positions[pos_id];
        Answer answer;
        answer.rb = position.rb;
        answer.period = position.period;
        answer.offset = position.offset;
        answer.distance = distance[user_pos][pos_id];
        answer.quality = cost[user_pos][pos_id];
        assigned[user_pos] = answer;
        mask = prev_state[mask];
    }
    return true;
}

std::vector<Answer> polish_period160_role_swaps(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    int max_passes = 12
) {
    if (answers.size() > 10000) return answers;
    for (const Answer& answer : answers) {
        if (answer.period != 160 && answer.period != 320 && answer.period != 640) return answers;
    }

    std::vector<Answer> current = answers;
    for (int pass = 0; pass < max_passes; ++pass) {
        std::vector<BinUsers> p160_bins = collect_period_bins(current, 160);
        std::vector<BinUsers> p320_bins = collect_period_bins(current, 320);
        int best_gain = 0;
        std::array<int, 3> best_users{};
        std::array<Answer, 3> best_assigned{};

        for (const BinUsers& source : p160_bins) {
            int source_user = source.users[0];
            for (const BinUsers& target : p320_bins) {
                std::array<int, 3> users{source_user, target.users[0], target.users[1]};
                int old_quality = current[users[0]].quality
                    + current[users[1]].quality
                    + current[users[2]].quality;

                std::array<Answer, 3> positions{};
                positions[0].rb = source.rb;
                positions[0].period = 320;
                positions[0].offset = source.offset;
                positions[1].rb = source.rb;
                positions[1].period = 320;
                positions[1].offset = source.offset + 160;
                positions[2].rb = target.rb;
                positions[2].period = 160;
                positions[2].offset = target.offset;

                std::array<Answer, 3> assigned{};
                int new_quality = 0;
                if (!best_three_user_role_assignment(options_by_user, users, positions, assigned, new_quality)) {
                    continue;
                }
                int gain = old_quality - new_quality;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_users = users;
                    best_assigned = assigned;
                }
            }
        }

        if (best_gain <= 0) break;
        for (int i = 0; i < 3; ++i) current[best_users[i]] = best_assigned[i];
    }

    return objective_value(current) < objective_value(answers) ? current : answers;
}

int best_quality_for_period(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int user_id,
    int period
) {
    int best = std::numeric_limits<int>::max() / 4;
    for (const TimeOption& option : options_by_user[user_id]) {
        if (option.period == period) best = std::min(best, option.quality);
    }
    return best;
}

std::vector<Answer> polish_period160_double_role_swaps(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    int max_passes = 3,
    int target_limit = 40
) {
    if (answers.size() <= 1000 || answers.size() > 2500) return answers;
    for (const Answer& answer : answers) {
        if (answer.period != 160 && answer.period != 320 && answer.period != 640) return answers;
    }

    std::vector<Answer> current = answers;
    for (int pass = 0; pass < max_passes; ++pass) {
        std::vector<BinUsers> p160_bins = collect_period_bins(current, 160);
        std::vector<BinUsers> p320_bins = collect_period_bins(current, 320);
        if (p160_bins.size() < 2 || p320_bins.size() < 2) break;

        struct RankedBin {
            int score = 0;
            BinUsers bin;
        };
        std::vector<RankedBin> ranked_targets;
        ranked_targets.reserve(p320_bins.size());
        for (const BinUsers& bin : p320_bins) {
            int score = std::numeric_limits<int>::min() / 4;
            for (int user_id : bin.users) {
                int best160 = best_quality_for_period(options_by_user, user_id, 160);
                if (best160 < std::numeric_limits<int>::max() / 8) {
                    score = std::max(score, current[user_id].quality - best160);
                }
            }
            if (score > std::numeric_limits<int>::min() / 8) {
                ranked_targets.push_back({score, bin});
            }
        }
        std::sort(ranked_targets.begin(), ranked_targets.end(), [](const RankedBin& a, const RankedBin& b) {
            if (a.score != b.score) return a.score > b.score;
            return std::tie(a.bin.rb, a.bin.offset) < std::tie(b.bin.rb, b.bin.offset);
        });
        if (static_cast<int>(ranked_targets.size()) > target_limit) {
            ranked_targets.resize(target_limit);
        }

        int best_gain = 0;
        std::vector<int> best_users;
        std::vector<Answer> best_assigned;

        for (int i = 0; i < static_cast<int>(p160_bins.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(p160_bins.size()); ++j) {
                const BinUsers& source_a = p160_bins[i];
                const BinUsers& source_b = p160_bins[j];
                for (int a = 0; a < static_cast<int>(ranked_targets.size()); ++a) {
                    for (int b = a + 1; b < static_cast<int>(ranked_targets.size()); ++b) {
                        const BinUsers& target_a = ranked_targets[a].bin;
                        const BinUsers& target_b = ranked_targets[b].bin;

                        std::vector<int> users{
                            source_a.users[0],
                            source_b.users[0],
                            target_a.users[0],
                            target_a.users[1],
                            target_b.users[0],
                            target_b.users[1],
                        };
                        int old_quality = 0;
                        for (int user_id : users) old_quality += current[user_id].quality;

                        std::vector<Answer> positions;
                        positions.reserve(6);
                        for (const BinUsers* source : {&source_a, &source_b}) {
                            Answer first;
                            first.rb = source->rb;
                            first.period = 320;
                            first.offset = source->offset;
                            positions.push_back(first);
                            Answer second = first;
                            second.offset = source->offset + 160;
                            positions.push_back(second);
                        }
                        for (const BinUsers* target : {&target_a, &target_b}) {
                            Answer upgrade;
                            upgrade.rb = target->rb;
                            upgrade.period = 160;
                            upgrade.offset = target->offset;
                            positions.push_back(upgrade);
                        }

                        std::vector<Answer> assigned;
                        int new_quality = 0;
                        if (!best_small_role_assignment(options_by_user, users, positions, assigned, new_quality)) {
                            continue;
                        }
                        int gain = old_quality - new_quality;
                        if (gain > best_gain) {
                            best_gain = gain;
                            best_users = users;
                            best_assigned = assigned;
                        }
                    }
                }
            }
        }

        if (best_gain <= 0) break;
        for (int i = 0; i < static_cast<int>(best_users.size()); ++i) {
            current[best_users[i]] = best_assigned[i];
        }
    }

    return objective_value(current) < objective_value(answers) ? current : answers;
}

[[maybe_unused]] std::vector<Answer> polish_period160_triple_role_swaps(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    int max_passes = 1,
    int target_limit = 14
) {
    if (answers.size() <= 1000 || answers.size() > 10000) return answers;
    for (const Answer& answer : answers) {
        if (answer.period != 160 && answer.period != 320 && answer.period != 640) return answers;
    }

    std::vector<Answer> current = answers;
    for (int pass = 0; pass < max_passes; ++pass) {
        std::vector<BinUsers> p160_bins = collect_period_bins(current, 160);
        std::vector<BinUsers> p320_bins = collect_period_bins(current, 320);
        if (p160_bins.size() < 3 || p320_bins.size() < 3) break;

        struct RankedBin {
            int score = 0;
            BinUsers bin;
        };
        std::vector<RankedBin> ranked_targets;
        ranked_targets.reserve(p320_bins.size());
        for (const BinUsers& bin : p320_bins) {
            int score = std::numeric_limits<int>::min() / 4;
            for (int user_id : bin.users) {
                int best160 = best_quality_for_period(options_by_user, user_id, 160);
                if (best160 < std::numeric_limits<int>::max() / 8) {
                    score = std::max(score, current[user_id].quality - best160);
                }
            }
            if (score > std::numeric_limits<int>::min() / 8) ranked_targets.push_back({score, bin});
        }
        std::sort(ranked_targets.begin(), ranked_targets.end(), [](const RankedBin& a, const RankedBin& b) {
            if (a.score != b.score) return a.score > b.score;
            return std::tie(a.bin.rb, a.bin.offset) < std::tie(b.bin.rb, b.bin.offset);
        });
        if (static_cast<int>(ranked_targets.size()) > target_limit) {
            ranked_targets.resize(target_limit);
        }
        if (ranked_targets.size() < 3) break;

        int best_gain = 0;
        std::vector<int> best_users;
        std::vector<Answer> best_assigned;

        for (int i = 0; i < static_cast<int>(p160_bins.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(p160_bins.size()); ++j) {
                for (int k = j + 1; k < static_cast<int>(p160_bins.size()); ++k) {
                    const std::array<const BinUsers*, 3> sources{
                        &p160_bins[i],
                        &p160_bins[j],
                        &p160_bins[k],
                    };
                    for (int a = 0; a < static_cast<int>(ranked_targets.size()); ++a) {
                        for (int b = a + 1; b < static_cast<int>(ranked_targets.size()); ++b) {
                            for (int c = b + 1; c < static_cast<int>(ranked_targets.size()); ++c) {
                                const std::array<const BinUsers*, 3> targets{
                                    &ranked_targets[a].bin,
                                    &ranked_targets[b].bin,
                                    &ranked_targets[c].bin,
                                };

                                std::vector<int> users;
                                users.reserve(9);
                                for (const BinUsers* source : sources) users.push_back(source->users[0]);
                                for (const BinUsers* target : targets) {
                                    users.push_back(target->users[0]);
                                    users.push_back(target->users[1]);
                                }

                                int old_quality = 0;
                                for (int user_id : users) old_quality += current[user_id].quality;

                                std::vector<Answer> positions;
                                positions.reserve(9);
                                for (const BinUsers* source : sources) {
                                    Answer first;
                                    first.rb = source->rb;
                                    first.period = 320;
                                    first.offset = source->offset;
                                    positions.push_back(first);
                                    Answer second = first;
                                    second.offset = source->offset + 160;
                                    positions.push_back(second);
                                }
                                for (const BinUsers* target : targets) {
                                    Answer upgrade;
                                    upgrade.rb = target->rb;
                                    upgrade.period = 160;
                                    upgrade.offset = target->offset;
                                    positions.push_back(upgrade);
                                }

                                std::vector<Answer> assigned;
                                int new_quality = 0;
                                if (!best_small_role_assignment(options_by_user, users, positions, assigned, new_quality)) {
                                    continue;
                                }
                                int gain = old_quality - new_quality;
                                if (gain > best_gain) {
                                    best_gain = gain;
                                    best_users = users;
                                    best_assigned = assigned;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (best_gain <= 0) break;
        for (int i = 0; i < static_cast<int>(best_users.size()); ++i) {
            current[best_users[i]] = best_assigned[i];
        }
    }

    return objective_value(current) < objective_value(answers) ? current : answers;
}

std::vector<Answer> polish_fixed_period160_global_p320(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& answers,
    int rb_budget
) {
    if (rb_budget <= 0 || rb_budget > R || answers.empty()) return answers;
    for (const Answer& answer : answers) {
        if (answer.period != 160 && answer.period != 320 && answer.period != 640) return answers;
    }

    int n = static_cast<int>(answers.size());
    std::vector<int> remaining_users;
    remaining_users.reserve(n);
    std::array<int, L> slot_capacity{};
    slot_capacity.fill(rb_budget);
    std::array<std::array<uint8_t, R>, L> used_rb_by_slot{};
    std::vector<Answer> fixed_answers;
    fixed_answers.reserve(n);

    for (int user_id = 0; user_id < n; ++user_id) {
        const Answer& answer = answers[user_id];
        if (answer.period == 160) {
            if (answer.rb < 0 || answer.rb >= rb_budget) return answers;
            fixed_answers.push_back(answer);
            const TimeOption* option = find_time_option(options_by_user, user_id, answer.period, answer.offset);
            if (option == nullptr) return answers;
            for_each_slot_in_mask(option->mask, [&](int slot) {
                --slot_capacity[slot];
                used_rb_by_slot[slot][answer.rb] = 1;
            });
        } else {
            remaining_users.push_back(user_id);
        }
    }

    for (int slot = 0; slot < L; ++slot) {
        if (slot_capacity[slot] < 0) return answers;
    }

    int m = static_cast<int>(remaining_users.size());
    int source = 0;
    int user_base = 1;
    int slot_base = user_base + m;
    int sink = slot_base + L;
    MinCostFlow flow(sink + 1);

    struct ArcInfo {
        int local_user = 0;
        int user_id = 0;
        int edge_idx = 0;
        const TimeOption* option = nullptr;
    };
    std::vector<ArcInfo> arcs;
    arcs.reserve(static_cast<size_t>(m) * 64);

    for (int local_user = 0; local_user < m; ++local_user) {
        int user_id = remaining_users[local_user];
        flow.add_edge(source, user_base + local_user, 1, 0);
        std::array<const TimeOption*, L> best_by_slot{};
        for (const TimeOption& option : options_by_user[user_id]) {
            if (option.period != 320 || popcount(option.mask) != 1) continue;
            int slot = first_set_slot(option.mask);
            if (slot < 0 || slot >= L || slot_capacity[slot] <= 0) continue;
            const TimeOption* current = best_by_slot[slot];
            if (current == nullptr || std::tie(option.quality, option.offset)
                    < std::tie(current->quality, current->offset)) {
                best_by_slot[slot] = &option;
            }
        }
        for (int slot = 0; slot < L; ++slot) {
            const TimeOption* option = best_by_slot[slot];
            if (option == nullptr) continue;
            int edge_idx = flow.add_edge(user_base + local_user, slot_base + slot, 1, option->quality);
            arcs.push_back({local_user, user_id, edge_idx, option});
        }
    }

    for (int slot = 0; slot < L; ++slot) {
        if (slot_capacity[slot] > 0) {
            flow.add_edge(slot_base + slot, sink, slot_capacity[slot], 0);
        }
    }

    auto [assigned, total_cost] = flow.min_cost_flow(source, sink, m);
    (void)total_cost;
    if (assigned < m) return answers;

    std::array<std::vector<Answer>, L> assigned_by_slot;
    for (const ArcInfo& arc : arcs) {
        const FlowEdge& edge = flow.graph[user_base + arc.local_user][arc.edge_idx];
        if (edge.cap != 0) continue;
        Answer answer;
        answer.period = arc.option->period;
        answer.offset = arc.option->offset;
        answer.distance = arc.option->distance;
        answer.quality = arc.option->quality;
        assigned_by_slot[arc.option->offset].push_back(answer);
        assigned_by_slot[arc.option->offset].back().rb = arc.user_id;
    }

    std::vector<Answer> trial(n);
    std::vector<uint8_t> filled(n, 0);
    for (const Answer& answer : fixed_answers) {
        int user_id = -1;
        for (int i = 0; i < n; ++i) {
            if (!filled[i] && answers[i].period == 160 && answers[i].rb == answer.rb
                    && answers[i].offset == answer.offset && answers[i].quality == answer.quality) {
                user_id = i;
                break;
            }
        }
        if (user_id < 0) return answers;
        trial[user_id] = answer;
        filled[user_id] = 1;
    }

    for (int slot = 0; slot < L; ++slot) {
        int next_rb = 0;
        for (Answer assigned_answer : assigned_by_slot[slot]) {
            int user_id = assigned_answer.rb;
            while (next_rb < rb_budget && used_rb_by_slot[slot][next_rb]) ++next_rb;
            if (next_rb >= rb_budget) return answers;
            assigned_answer.rb = next_rb;
            trial[user_id] = assigned_answer;
            filled[user_id] = 1;
            used_rb_by_slot[slot][next_rb] = 1;
        }
    }

    for (uint8_t value : filled) {
        if (!value) return answers;
    }
    return objective_value(trial) < objective_value(answers) ? trial : answers;
}

std::vector<Answer> polish_sparse_two_cell_upgrade(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int passes
) {
    int n = static_cast<int>(answers.size());
    if (rb_budget <= 0 || rb_budget > R || n == 0) return answers;

    std::vector<Mask> occupied(rb_budget);
    for (const Answer& answer : answers) {
        if (answer.period == 640 || answer.rb >= rb_budget) continue;
        add_mask(occupied[answer.rb], answer_mask(answer));
    }

    for (int pass = 0; pass < passes; ++pass) {
        struct RankedUser {
            int user_id = 0;
            int potential = 0;
            int quality = 0;
        };
        std::vector<RankedUser> order;
        order.reserve(n);
        for (int user_id = 0; user_id < n; ++user_id) {
            if (answers[user_id].period == 640 || answers[user_id].rb >= rb_budget) continue;
            int best_two_cell = answers[user_id].quality;
            for (const TimeOption& option : options_by_user[user_id]) {
                if (popcount(option.mask) == 2) best_two_cell = std::min(best_two_cell, option.quality);
            }
            int potential = answers[user_id].quality - best_two_cell;
            if (potential > 0) order.push_back({user_id, potential, answers[user_id].quality});
        }
        if (order.empty()) break;
        std::sort(order.begin(), order.end(), [](const RankedUser& a, const RankedUser& b) {
            return std::make_tuple(a.potential, a.quality, -a.user_id)
                > std::make_tuple(b.potential, b.quality, -b.user_id);
        });

        int applied = 0;
        for (const RankedUser& item : order) {
            int user_id = item.user_id;
            Answer old_answer = answers[user_id];
            if (old_answer.period == 640 || old_answer.rb >= rb_budget) continue;
            Mask old_mask = answer_mask(old_answer);
            remove_mask(occupied[old_answer.rb], old_mask);

            bool found = false;
            Answer best_answer;
            const TimeOption* best_option = nullptr;
            std::tuple<int, int, int, int> best_tuple{};
            for (const TimeOption& option : options_by_user[user_id]) {
                if (popcount(option.mask) != 2) continue;
                if (option.quality >= old_answer.quality) continue;
                for (int rb = 0; rb < rb_budget; ++rb) {
                    if (intersects(occupied[rb], option.mask)) continue;
                    auto candidate_tuple = std::make_tuple(
                        option.quality,
                        rb == old_answer.rb ? 0 : 1,
                        rb,
                        option.offset
                    );
                    if (!found || candidate_tuple < best_tuple) {
                        found = true;
                        best_tuple = candidate_tuple;
                        best_option = &option;
                        best_answer = Answer{rb, option.period, option.offset, option.distance, option.quality};
                    }
                }
            }

            if (found && best_option != nullptr) {
                answers[user_id] = best_answer;
                add_mask(occupied[best_answer.rb], best_option->mask);
                ++applied;
            } else {
                add_mask(occupied[old_answer.rb], old_mask);
            }
        }
        if (applied == 0) break;
    }

    return answers;
}

struct PlacementChoice {
    bool found = false;
    Answer answer;
    const TimeOption* option = nullptr;
};

PlacementChoice best_sparse_cell_placement(
    const std::vector<TimeOption>& options,
    const std::vector<Mask>& occupied,
    int rb_budget,
    int required_cells,
    int max_matching_options,
    int quality_limit_exclusive = std::numeric_limits<int>::max()
) {
    PlacementChoice best;
    int inspected = 0;
    for (const TimeOption& option : options) {
        if (popcount(option.mask) != required_cells) continue;
        ++inspected;
        if (option.quality >= quality_limit_exclusive) {
            if (inspected >= max_matching_options) break;
            continue;
        }
        for (int rb = 0; rb < rb_budget; ++rb) {
            if (intersects(occupied[rb], option.mask)) continue;
            best.found = true;
            best.option = &option;
            best.answer = Answer{rb, option.period, option.offset, option.distance, option.quality};
            return best;
        }
        if (inspected >= max_matching_options) break;
    }
    return best;
}

std::vector<Answer> polish_dense_two_cell_exchange(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> answers,
    int rb_budget,
    int max_passes,
    int max_add_options,
    int max_drop_options,
    int add_candidate_limit
) {
    int n = static_cast<int>(answers.size());
    if (rb_budget <= 0 || rb_budget > R || n == 0) return answers;
    if (compact_stats(answers).unscheduled > 0) return answers;

    for (int pass = 0; pass < max_passes; ++pass) {
        std::vector<int> two_cell_users;
        two_cell_users.reserve(32);
        struct AddCandidate {
            int user_id = 0;
            int potential = 0;
            int best_two_quality = 0;
            int current_quality = 0;
        };
        std::vector<AddCandidate> add_candidates;
        add_candidates.reserve(n);

        for (int user_id = 0; user_id < n; ++user_id) {
            const Answer& answer = answers[user_id];
            if (answer.period == 640 || answer.rb < 0 || answer.rb >= rb_budget) return answers;
            int cells = popcount(answer_mask(answer));
            if (cells == 2) {
                two_cell_users.push_back(user_id);
                continue;
            }
            if (cells != 1) continue;
            int best_two_quality = answer.quality;
            int inspected = 0;
            for (const TimeOption& option : options_by_user[user_id]) {
                if (popcount(option.mask) != 2) continue;
                best_two_quality = std::min(best_two_quality, option.quality);
                if (++inspected >= max_add_options) break;
            }
            int potential = answer.quality - best_two_quality;
            if (potential > 0) {
                add_candidates.push_back({user_id, potential, best_two_quality, answer.quality});
            }
        }

        if (two_cell_users.empty() || add_candidates.empty()) break;
        std::sort(add_candidates.begin(), add_candidates.end(), [](const AddCandidate& a, const AddCandidate& b) {
            return std::make_tuple(a.potential, -a.best_two_quality, a.current_quality, -a.user_id)
                > std::make_tuple(b.potential, -b.best_two_quality, b.current_quality, -b.user_id);
        });
        if (static_cast<int>(add_candidates.size()) > add_candidate_limit) {
            add_candidates.resize(add_candidate_limit);
        }

        std::vector<Mask> occupied(rb_budget);
        for (const Answer& answer : answers) {
            add_mask(occupied[answer.rb], answer_mask(answer));
        }

        struct ExchangeMove {
            int delta = 0;
            int drop_user = -1;
            int add_user = -1;
            Answer drop_answer;
            Answer add_answer;
            const TimeOption* drop_option = nullptr;
            const TimeOption* add_option = nullptr;
        };
        ExchangeMove best_move;

        for (int drop_user : two_cell_users) {
            const Answer old_drop = answers[drop_user];
            const Mask old_drop_mask = answer_mask(old_drop);
            remove_mask(occupied[old_drop.rb], old_drop_mask);

            for (const AddCandidate& candidate : add_candidates) {
                int add_user = candidate.user_id;
                const Answer old_add = answers[add_user];
                if (old_add.period == 640 || old_add.rb < 0 || old_add.rb >= rb_budget) continue;
                if (popcount(answer_mask(old_add)) != 1) continue;

                const Mask old_add_mask = answer_mask(old_add);
                remove_mask(occupied[old_add.rb], old_add_mask);

                PlacementChoice add_choice = best_sparse_cell_placement(
                    options_by_user[add_user],
                    occupied,
                    rb_budget,
                    2,
                    max_add_options,
                    old_add.quality
                );

                if (add_choice.found) {
                    add_mask(occupied[add_choice.answer.rb], add_choice.option->mask);
                    PlacementChoice drop_choice = best_sparse_cell_placement(
                        options_by_user[drop_user],
                        occupied,
                        rb_budget,
                        1,
                        max_drop_options
                    );

                    if (drop_choice.found) {
                        int delta = add_choice.answer.quality + drop_choice.answer.quality
                            - old_add.quality - old_drop.quality;
                        if (delta < best_move.delta) {
                            best_move.delta = delta;
                            best_move.drop_user = drop_user;
                            best_move.add_user = add_user;
                            best_move.drop_answer = drop_choice.answer;
                            best_move.add_answer = add_choice.answer;
                            best_move.drop_option = drop_choice.option;
                            best_move.add_option = add_choice.option;
                        }
                    }
                    remove_mask(occupied[add_choice.answer.rb], add_choice.option->mask);
                }

                add_mask(occupied[old_add.rb], old_add_mask);
            }

            add_mask(occupied[old_drop.rb], old_drop_mask);
        }

        if (best_move.delta >= 0 || best_move.drop_user < 0 || best_move.add_user < 0) break;
        answers[best_move.drop_user] = best_move.drop_answer;
        answers[best_move.add_user] = best_move.add_answer;
    }

    return answers;
}

std::vector<Answer> solve_dense_two_cell_exchange_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    int n = static_cast<int>(options_by_user.size());
    int rb_budget = occupied_cell_lower_bound_rb(options_by_user);
    if (rb_budget <= 0 || rb_budget > R) return {};

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    auto consider = [&](std::vector<Answer> candidate) {
        if (candidate.empty() || compact_stats(candidate).unscheduled > 0) return;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    std::vector<Answer> start = solve_single_cell_flow_budget_prepared(options_by_user, rb_budget);
    if (compact_stats(start).unscheduled > 0) return start;
    consider(start);

    std::vector<Answer> current = polish_sparse_two_cell_upgrade(
        options_by_user,
        start,
        rb_budget,
        n > 2500 ? 2 : 4
    );
    consider(current);

    current = polish_fixed_period160_global_p320(options_by_user, current, rb_budget);
    consider(current);

    current = polish_dense_two_cell_exchange(
        options_by_user,
        current,
        rb_budget,
        n > 2500 ? 4 : 6,
        n > 2500 ? 24 : 20,
        n > 2500 ? 24 : 20,
        n > 2500 ? 1400 : 900
    );
    consider(current);

    current = polish_fixed_period160_global_p320(options_by_user, current, rb_budget);
    consider(current);

    return best;
}

std::vector<Answer> solve_sparse_two_cell_upgrade_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    int n = static_cast<int>(options_by_user.size());
    int uplink_slots = seen_uplink_slot_count(options_by_user);
    int lower = occupied_cell_lower_bound_rb(options_by_user);
    int two_cell_budget = std::max(lower, (2 * n + uplink_slots - 1) / uplink_slots);
    int start_budget = std::max(lower, two_cell_budget - 2);
    int upper = std::min(R, two_cell_budget + 1);

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    auto consider = [&](std::vector<Answer> candidate) {
        if (candidate.empty() || compact_stats(candidate).unscheduled > 0) return;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    for (int rb_budget = start_budget; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> start = solve_single_cell_flow_budget_prepared(options_by_user, rb_budget);
        if (compact_stats(start).unscheduled > 0) continue;
        consider(start);

        std::vector<Answer> current = polish_sparse_two_cell_upgrade(
            options_by_user,
            start,
            rb_budget,
            n <= 700 ? 4 : 2
        );
        consider(current);

        current = polish_fixed_period160_global_p320(options_by_user, current, rb_budget);
        consider(current);

        current = polish_fixed_pattern_flow(options_by_user, current);
        current = polish_fixed_pattern_assignment(options_by_user, current);
        consider(current);
    }
    return best;
}

std::vector<Answer> solve_assignment_upgrade_sweep_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool use_pattern_flow = false
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, upper] = rb_budget_window(options_by_user, 5, 0);

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> assignment = solve_single_report_assignment_prepared(options_by_user, rb_budget);
        std::vector<std::vector<Answer>> variants;
        variants.push_back(polish_fixed_rb_quality(options_by_user, assignment, rb_budget, 4));
        variants.push_back(polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 0, 1100 + rb_budget));
        variants.push_back(polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 1, 2100 + rb_budget));
        variants.push_back(polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 2, 3100 + rb_budget));
        variants.push_back(polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 2, 4100 + rb_budget));
        variants.push_back(polish_ejection_chain_quality(options_by_user, variants[0], rb_budget, 5, 4));
        variants.push_back(polish_ejection_chain_quality(options_by_user, variants[1], rb_budget, 5, 4));
        if (n <= 1500) {
            variants.push_back(polish_tabu_fixed_rb_quality(
                options_by_user, variants[0], rb_budget, 90, 9, 22, 700
            ));
            variants.push_back(polish_tabu_fixed_rb_quality(
                options_by_user, variants[1], rb_budget, 80, 9, 22, 700
            ));
        } else {
            variants.push_back(polish_tabu_fixed_rb_quality(
                options_by_user, variants[0], rb_budget, 24, 7, 14, 320
            ));
        }
        variants.push_back(polish_beam_fixed_set_quality(options_by_user, variants[0], rb_budget, 48, 96, 18));
        variants.push_back(polish_beam_fixed_set_quality(options_by_user, variants[1], rb_budget, 64, 128, 20));

        std::vector<Answer> budget_best;
        int64_t budget_best_obj = std::numeric_limits<int64_t>::max();
        for (std::vector<Answer>& candidate : variants) {
            int64_t obj = objective_value(candidate);
            if (obj < budget_best_obj) {
                budget_best_obj = obj;
                budget_best = candidate;
            }
            if (obj < best_obj) {
                best_obj = obj;
                best = std::move(candidate);
            }
        }
        if (!budget_best.empty()) {
            std::vector<Answer> polished = polish_fixed_pattern_assignment(options_by_user, budget_best);
            if (use_pattern_flow) {
                polished = polish_fixed_pattern_flow(options_by_user, polished);
            }
            polished = polish_period160_role_swaps(options_by_user, polished, 8);
            polished = polish_period160_double_role_swaps(options_by_user, polished, 2, 40);
            polished = polish_period160_role_swaps(options_by_user, polished, 4);
            polished = polish_fixed_period160_global_p320(options_by_user, polished, rb_budget);
            if (use_pattern_flow) {
                polished = polish_fixed_pattern_flow(options_by_user, polished);
            }
            polished = polish_fixed_pattern_assignment(options_by_user, polished);
            int64_t obj = objective_value(polished);
            if (obj < best_obj) {
                best_obj = obj;
                best = std::move(polished);
            }
        }
    }
    return best;
}

std::vector<Answer> solve_assignment_upgrade_exchange_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool use_pattern_flow = false
) {
    int n = static_cast<int>(options_by_user.size());
    std::vector<Answer> best = solve_assignment_upgrade_sweep_prepared(options_by_user, use_pattern_flow);
    if (best.empty() || compact_stats(best).unscheduled > 0) return best;

    SolutionStats best_stats = compact_stats(best);
    int rb_budget = best_stats.rb_used;
    if (rb_budget <= 0 || rb_budget > R) return best;

    std::vector<Answer> current = polish_dense_two_cell_exchange(
        options_by_user,
        best,
        rb_budget,
        n > 2500 ? 2 : 4,
        n > 2500 ? 18 : 20,
        n > 2500 ? 18 : 20,
        n > 2500 ? 550 : 900
    );

    if (objective_value(current) < best_stats.objective) {
        best = std::move(current);
        best_stats = compact_stats(best);
    }
    return best;
}

std::vector<Answer> solve_assignment_upgrade_lean_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    auto [lower, upper] = rb_budget_window(options_by_user, 5, 0);
    (void)upper;
    int rb_budget = lower;

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    auto consider = [&](std::vector<Answer> candidate) {
        if (candidate.empty() || compact_stats(candidate).unscheduled > 0) return;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    std::vector<Answer> assignment = solve_single_report_assignment_prepared(options_by_user, rb_budget);
    consider(assignment);
    if (compact_stats(assignment).unscheduled > 0) return best;

    std::vector<Answer> current = polish_fixed_rb_quality(options_by_user, assignment, rb_budget, 4);
    consider(current);
    std::vector<Answer> after_first_swaps = polish_period160_role_swaps(options_by_user, current, 8);
    consider(after_first_swaps);

    std::vector<Answer> without_double = polish_period160_role_swaps(options_by_user, after_first_swaps, 4);
    consider(without_double);

    current = polish_dense_two_cell_exchange(
        options_by_user,
        best,
        rb_budget,
        1,
        16,
        16,
        400
    );
    consider(current);
    return best;
}

std::vector<Answer> solve_assignment_upgrade_diagnostics_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, upper] = rb_budget_window(options_by_user, 5, 0);
    (void)upper;
    int rb_budget = lower;

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();
    auto consider = [&](const std::string& name, std::vector<Answer> candidate, double ms) {
        int64_t obj = objective_value(candidate);
        SolutionStats st = compact_stats(candidate);
        std::cerr << "diag " << name
                  << " obj=" << obj
                  << " rb=" << st.rb_used
                  << " q=" << st.quality_sum
                  << " uns=" << st.unscheduled
                  << " ms=" << ms << "\n";
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    auto timed = [&](const std::string& name, auto&& fn) -> std::vector<Answer> {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Answer> candidate = fn();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        consider(name, candidate, ms);
        return candidate;
    };

    std::vector<Answer> assignment = timed("assignment", [&]() {
        return solve_single_report_assignment_prepared(options_by_user, rb_budget);
    });
    std::vector<Answer> v0 = timed("fixed_rb", [&]() {
        return polish_fixed_rb_quality(options_by_user, assignment, rb_budget, 4);
    });
    std::vector<Answer> v1 = timed("dynamic0", [&]() {
        return polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 0, 1100 + rb_budget);
    });
    std::vector<Answer> v2 = timed("dynamic1", [&]() {
        return polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 1, 2100 + rb_budget);
    });
    std::vector<Answer> v3 = timed("dynamic2a", [&]() {
        return polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 2, 3100 + rb_budget);
    });
    std::vector<Answer> v4 = timed("dynamic2b", [&]() {
        return polish_fixed_rb_quality_dynamic(options_by_user, assignment, rb_budget, 5, 2, 4100 + rb_budget);
    });
    timed("ejection_v0", [&]() {
        return polish_ejection_chain_quality(options_by_user, v0, rb_budget, 5, 4);
    });
    timed("ejection_v1", [&]() {
        return polish_ejection_chain_quality(options_by_user, v1, rb_budget, 5, 4);
    });
    timed("tabu_v0", [&]() {
        return polish_tabu_fixed_rb_quality(options_by_user, v0, rb_budget, n <= 1500 ? 90 : 24, 7, 14, 320);
    });
    timed("beam_v0", [&]() {
        return polish_beam_fixed_set_quality(options_by_user, v0, rb_budget, 48, 96, 18);
    });
    timed("beam_v1", [&]() {
        return polish_beam_fixed_set_quality(options_by_user, v1, rb_budget, 64, 128, 20);
    });
    std::vector<Answer> polished = timed("pattern_assignment", [&]() {
        return polish_fixed_pattern_assignment(options_by_user, best);
    });
    polished = timed("role_swaps8", [&]() {
        return polish_period160_role_swaps(options_by_user, polished, 8);
    });
    polished = timed("double_swaps", [&]() {
        return polish_period160_double_role_swaps(options_by_user, polished, 2, 40);
    });
    polished = timed("role_swaps4", [&]() {
        return polish_period160_role_swaps(options_by_user, polished, 4);
    });
    polished = timed("global_p320", [&]() {
        return polish_fixed_period160_global_p320(options_by_user, polished, rb_budget);
    });
    polished = timed("final_pattern_assignment", [&]() {
        return polish_fixed_pattern_assignment(options_by_user, polished);
    });
    (void)polished;
    return best;
}

double price_mask_cost(const Mask& mask, const std::array<double, L>& prices) {
    double total = 0.0;
    for_each_slot_in_mask(mask, [&](int slot) { total += prices[slot]; });
    return total;
}

std::vector<Answer> color_selected_options(
    const std::vector<const TimeOption*>& selected,
    int rb_budget
) {
    int n = static_cast<int>(selected.size());
    if (rb_budget <= 0 || rb_budget > R) return {};

    std::array<std::vector<int>, L> users_by_slot;
    std::vector<int> degree(n, 0);
    std::vector<uint8_t> active(n, 0);
    int active_count = 0;
    for (int user_id = 0; user_id < n; ++user_id) {
        if (selected[user_id] == nullptr) continue;
        active[user_id] = 1;
        ++active_count;
        for_each_slot_in_mask(selected[user_id]->mask, [&](int slot) {
            users_by_slot[slot].push_back(user_id);
        });
    }
    for (const auto& users : users_by_slot) {
        int local_degree = std::max(0, static_cast<int>(users.size()) - 1);
        for (int user_id : users) degree[user_id] += local_degree;
    }

    std::vector<Mask> occupied(rb_budget);
    std::vector<Answer> answers(n);
    std::vector<int> color(n, -1);
    std::vector<uint64_t> forbidden(n, 0);
    std::vector<int> rb_load(rb_budget, 0);
    uint64_t all_colors = rb_budget == 64 ? ~uint64_t{0} : ((uint64_t{1} << rb_budget) - 1);

    int remaining = active_count;
    while (remaining > 0) {
        int user_id = -1;
        std::tuple<int, int, int, int, int> best_tuple{};
        for (int candidate = 0; candidate < n; ++candidate) {
            if (!active[candidate] || color[candidate] >= 0) continue;
            int saturation = __builtin_popcountll(forbidden[candidate] & all_colors);
            int cells = popcount(selected[candidate]->mask);
            auto candidate_tuple = std::make_tuple(
                saturation,
                degree[candidate],
                cells,
                -selected[candidate]->quality,
                -candidate
            );
            if (user_id < 0 || candidate_tuple > best_tuple) {
                user_id = candidate;
                best_tuple = candidate_tuple;
            }
        }
        if (user_id < 0) return {};

        const TimeOption* option = selected[user_id];
        int best_rb = -1;
        int best_load = std::numeric_limits<int>::max();
        uint64_t allowed = all_colors & ~forbidden[user_id];
        for (int rb = 0; rb < rb_budget; ++rb) {
            if ((allowed & (uint64_t{1} << rb)) == 0) continue;
            if (intersects(occupied[rb], option->mask)) continue;
            int load = rb_load[rb];
            if (load < best_load) {
                best_load = load;
                best_rb = rb;
            }
        }
        if (best_rb < 0) return {};
        Answer answer;
        answer.rb = best_rb;
        answer.period = option->period;
        answer.offset = option->offset;
        answer.distance = option->distance;
        answer.quality = option->quality;
        answers[user_id] = answer;
        color[user_id] = best_rb;
        add_mask(occupied[best_rb], option->mask);
        rb_load[best_rb] += popcount(option->mask);
        --remaining;

        uint64_t color_bit = uint64_t{1} << best_rb;
        for_each_slot_in_mask(option->mask, [&](int slot) {
            for (int neighbor : users_by_slot[slot]) {
                if (neighbor != user_id && active[neighbor] && color[neighbor] < 0) {
                    forbidden[neighbor] |= color_bit;
                }
            }
        });
    }
    return answers;
}

int slot_overload_total(const std::array<int, L>& loads, int rb_budget);

bool repair_selected_slot_capacity(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<const TimeOption*>& selected,
    std::array<int, L>& loads,
    int rb_budget,
    int max_moves
);

std::vector<Answer> solve_price_relaxation_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, upper] = rb_budget_window(options_by_user, 5, 0);

    std::vector<Answer> best = solve_assignment_upgrade_sweep_prepared(options_by_user);
    int64_t best_obj = objective_value(best);

    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::array<double, L> prices{};
        double step = 35.0;
        int iterations = n <= 800 ? 60 : 30;
        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<const TimeOption*> selected(n, nullptr);
            std::array<int, L> loads{};

            for (int user_id = 0; user_id < n; ++user_id) {
                const TimeOption* best_option = nullptr;
                std::tuple<int, int, int> best_tuple{};
                for (const TimeOption& option : options_by_user[user_id]) {
                    double price_cost = price_mask_cost(option.mask, prices);
                    auto candidate_tuple = std::make_tuple(
                        option.quality + price_cost,
                        option.quality,
                        popcount(option.mask)
                    );
                    if (best_option == nullptr || candidate_tuple < best_tuple) {
                        best_tuple = candidate_tuple;
                        best_option = &option;
                    }
                }
                selected[user_id] = best_option;
                if (best_option != nullptr) {
                    for_each_slot_in_mask(best_option->mask, [&](int slot) { ++loads[slot]; });
                }
            }

            bool capacity_ok = slot_overload_total(loads, rb_budget) == 0;
            if (!capacity_ok && slot_overload_total(loads, rb_budget) <= std::max(80, n / 2)) {
                capacity_ok = repair_selected_slot_capacity(
                    options_by_user,
                    selected,
                    loads,
                    rb_budget,
                    std::max(120, n * 2)
                );
            }
            if (capacity_ok) {
                std::vector<Answer> candidate = color_selected_options(selected, rb_budget);
                if (!candidate.empty()) {
                    candidate = polish_fixed_rb_quality_dynamic(
                        options_by_user, std::move(candidate), rb_budget, 3, iter % 3, 9000 + iter + rb_budget
                    );
                    int64_t obj = objective_value(candidate);
                    if (obj < best_obj) {
                        best_obj = obj;
                        best = std::move(candidate);
                    }
                }
            }

            for (int slot = 0; slot < L; ++slot) {
                double violation = static_cast<double>(loads[slot] - rb_budget);
                prices[slot] = std::max(0.0, prices[slot] + step * violation);
            }
            step *= 0.985;
        }
    }
    return best;
}

bool option_fits_slot_load(const TimeOption& option, const std::array<int, L>& loads, int rb_budget) {
    bool fits = true;
    for_each_slot_in_mask(option.mask, [&](int slot) {
        if (loads[slot] >= rb_budget) fits = false;
    });
    return fits;
}

void add_option_to_slot_load(const TimeOption& option, std::array<int, L>& loads) {
    for_each_slot_in_mask(option.mask, [&](int slot) { ++loads[slot]; });
}

void remove_option_from_slot_load(const TimeOption& option, std::array<int, L>& loads) {
    for_each_slot_in_mask(option.mask, [&](int slot) { --loads[slot]; });
}

int slot_overload_total(const std::array<int, L>& loads, int rb_budget) {
    int total = 0;
    for (int load : loads) total += std::max(0, load - rb_budget);
    return total;
}

bool repair_selected_slot_capacity(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<const TimeOption*>& selected,
    std::array<int, L>& loads,
    int rb_budget,
    int max_moves
) {
    int overload = slot_overload_total(loads, rb_budget);
    int n = static_cast<int>(selected.size());

    for (int move = 0; move < max_moves && overload > 0; ++move) {
        int best_user = -1;
        const TimeOption* best_option = nullptr;
        int best_reduction = 0;
        std::tuple<int, int, int, int, int> best_tuple{};

        for (int user_id = 0; user_id < n; ++user_id) {
            const TimeOption* current = selected[user_id];
            if (current == nullptr) continue;

            int current_relief = 0;
            for_each_slot_in_mask(current->mask, [&](int slot) {
                current_relief += std::max(0, loads[slot] - rb_budget)
                    - std::max(0, loads[slot] - 1 - rb_budget);
            });
            if (current_relief <= 0) continue;

            remove_option_from_slot_load(*current, loads);
            for (const TimeOption& option : options_by_user[user_id]) {
                if (&option == current) continue;

                int added_excess = 0;
                for_each_slot_in_mask(option.mask, [&](int slot) {
                    added_excess += std::max(0, loads[slot] + 1 - rb_budget)
                        - std::max(0, loads[slot] - rb_budget);
                });
                int reduction = current_relief - added_excess;
                if (reduction <= 0) continue;

                int quality_delta = option.quality - current->quality;
                auto candidate_tuple = std::make_tuple(
                    reduction,
                    -quality_delta,
                    -added_excess,
                    -option.quality,
                    -popcount(option.mask)
                );
                if (best_user < 0 || candidate_tuple > best_tuple) {
                    best_user = user_id;
                    best_option = &option;
                    best_reduction = reduction;
                    best_tuple = candidate_tuple;
                }
            }
            add_option_to_slot_load(*current, loads);
        }

        if (best_user < 0 || best_option == nullptr || best_reduction <= 0) break;
        remove_option_from_slot_load(*selected[best_user], loads);
        selected[best_user] = best_option;
        add_option_to_slot_load(*best_option, loads);
        overload = slot_overload_total(loads, rb_budget);
    }

    return overload == 0;
}

std::vector<Answer> solve_slot_capacity_regret_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, upper] = rb_budget_window(options_by_user, 5, 0);

    std::vector<Answer> best = solve_assignment_upgrade_sweep_prepared(options_by_user);
    int64_t best_obj = objective_value(best);

    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::array<int, L> loads{};
        std::vector<const TimeOption*> selected(n, nullptr);
        std::vector<uint8_t> remaining(n, 1);
        int remaining_count = n;
        bool failed = false;

        while (remaining_count > 0 && !failed) {
            int selected_user = -1;
            const TimeOption* selected_option = nullptr;
            int selected_regret = std::numeric_limits<int>::min();
            int selected_score = std::numeric_limits<int>::max();

            for (int user_id = 0; user_id < n; ++user_id) {
                if (!remaining[user_id]) continue;
                const TimeOption* best_option = nullptr;
                int best_quality = std::numeric_limits<int>::max();
                int second_quality = std::numeric_limits<int>::max();

                for (const TimeOption& option : options_by_user[user_id]) {
                    if (!option_fits_slot_load(option, loads, rb_budget)) continue;
                    if (option.quality < best_quality) {
                        second_quality = best_quality;
                        best_quality = option.quality;
                        best_option = &option;
                    } else if (option.quality < second_quality) {
                        second_quality = option.quality;
                    }
                }
                if (best_option == nullptr) {
                    failed = true;
                    break;
                }
                int regret = second_quality == std::numeric_limits<int>::max()
                    ? 1000000
                    : second_quality - best_quality;
                auto candidate_tuple = std::make_tuple(regret, -best_quality, -popcount(best_option->mask), user_id);
                auto best_tuple = std::make_tuple(selected_regret, -selected_score, selected_option ? -popcount(selected_option->mask) : 0, selected_user);
                if (selected_user < 0 || candidate_tuple > best_tuple) {
                    selected_user = user_id;
                    selected_option = best_option;
                    selected_regret = regret;
                    selected_score = best_quality;
                }
            }

            if (failed || selected_user < 0 || selected_option == nullptr) {
                failed = true;
                break;
            }
            selected[selected_user] = selected_option;
            remaining[selected_user] = 0;
            --remaining_count;
            add_option_to_slot_load(*selected_option, loads);
        }
        if (failed) continue;

        std::vector<Answer> colored = color_selected_options(selected, rb_budget);
        if (colored.empty()) continue;
        colored = polish_fixed_rb_quality_dynamic(options_by_user, std::move(colored), rb_budget, 4, 1, 12000 + rb_budget);
        int64_t obj = objective_value(colored);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(colored);
        }
    }
    return best;
}

std::vector<Answer> polish_classical_fixed_budget(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    std::vector<Answer> candidate,
    int rb_budget,
    int seed,
    bool strong = true
) {
    if (candidate.empty()) return candidate;
    int n = static_cast<int>(candidate.size());

    std::vector<Answer> best = candidate;
    int64_t best_obj = objective_value(best);
    auto consider = [&](std::vector<Answer> trial) {
        if (trial.empty()) return;
        int64_t obj = objective_value(trial);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(trial);
        }
    };

    std::vector<Answer> current = polish_fixed_rb_quality_dynamic(
        options_by_user,
        candidate,
        rb_budget,
        n <= 1500 ? 6 : 3,
        1,
        seed
    );
    consider(current);

    current = polish_ejection_chain_quality(
        options_by_user,
        current,
        rb_budget,
        n <= 1500 ? 5 : 2,
        n <= 1500 ? 4 : 2
    );
    consider(current);

    current = polish_tabu_fixed_rb_quality(
        options_by_user,
        current,
        rb_budget,
        strong ? (n <= 1500 ? 80 : 20) : (n <= 1500 ? 28 : 8),
        9,
        strong ? (n <= 1500 ? 22 : 14) : 12,
        strong ? (n <= 1500 ? 700 : 320) : (n <= 1500 ? 280 : 120)
    );
    consider(current);

    if (strong && n <= 1400) {
        consider(polish_beam_fixed_set_quality(options_by_user, current, rb_budget, 56, 128, 20));
    }

    if (!strong) return best;

    current = polish_fixed_pattern_assignment(options_by_user, best);
    current = polish_fixed_pattern_flow(options_by_user, current);
    current = polish_period160_role_swaps(options_by_user, current, 8);
    current = polish_period160_double_role_swaps(options_by_user, current, 2, 40);
    current = polish_period160_role_swaps(options_by_user, current, 4);
    current = polish_fixed_period160_global_p320(options_by_user, current, rb_budget);
    current = polish_fixed_pattern_flow(options_by_user, current);
    current = polish_fixed_pattern_assignment(options_by_user, current);
    consider(current);

    return best;
}

std::vector<Answer> solve_priced_coloring_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool full_search = false
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, upper] = rb_budget_window(options_by_user, 6, 1);

    std::vector<Answer> best;
    int64_t best_obj = std::numeric_limits<int64_t>::max();

    auto consider = [&](std::vector<Answer> candidate) {
        if (candidate.empty()) return;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    if (!full_search && n > 2500) {
        best = solve_assignment_upgrade_sweep_prepared(options_by_user, false);
        best_obj = objective_value(best);
        if (compact_stats(best).rb_used >= R) return best;
        consider(solve_assignment_upgrade_sweep_prepared(options_by_user, true));
        return best;
    }

    if (!full_search && n > 600) {
        best = solve_assignment_upgrade_sweep_prepared(options_by_user, n > 800);
        best_obj = objective_value(best);
        return best;
    }

    best = solve_assignment_upgrade_sweep_prepared(options_by_user, true);
    best_obj = objective_value(best);
    if (n > 2500 && compact_stats(best).rb_used >= R) return best;

    if (full_search && n <= 1800) {
        consider(solve_slot_capacity_regret_prepared(options_by_user));
    }

    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> assignment = solve_single_report_assignment_prepared(options_by_user, rb_budget);
        consider(polish_classical_fixed_budget(options_by_user, std::move(assignment), rb_budget, 17000 + rb_budget));

        std::array<double, L> prices{};
        double step = n <= 900 ? 48.0 : (n <= 2200 ? 38.0 : 30.0);
        int iterations = full_search
            ? (n <= 900 ? 42 : (n <= 2200 ? 30 : 20))
            : (n <= 900 ? 24 : (n <= 2200 ? 18 : 12));

        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<const TimeOption*> selected(n, nullptr);
            std::array<int, L> loads{};

            for (int user_id = 0; user_id < n; ++user_id) {
                const TimeOption* best_option = nullptr;
                std::tuple<double, int, int, int, int> best_tuple{};
                int inspected = 0;
                for (const TimeOption& option : options_by_user[user_id]) {
                    double price_cost = price_mask_cost(option.mask, prices);
                    double cell_bias = (iter % 5 == 1) ? 0.35 * popcount(option.mask) : 0.0;
                    double score = option.quality + price_cost + cell_bias;
                    auto candidate_tuple = std::make_tuple(
                        score,
                        option.quality,
                        popcount(option.mask),
                        option.period,
                        option.offset
                    );
                    if (best_option == nullptr || candidate_tuple < best_tuple) {
                        best_tuple = candidate_tuple;
                        best_option = &option;
                    }
                    if (n > 2500 && ++inspected >= 48) break;
                }
                selected[user_id] = best_option;
                if (best_option != nullptr) add_option_to_slot_load(*best_option, loads);
            }

            std::array<int, L> dual_loads = loads;
            int overload = slot_overload_total(loads, rb_budget);
            bool capacity_ok = overload == 0;
            int repair_threshold = n <= 1200 ? std::max(160, n) : std::max(80, n / 10);
            if (!capacity_ok && overload <= repair_threshold) {
                capacity_ok = repair_selected_slot_capacity(
                    options_by_user,
                    selected,
                    loads,
                    rb_budget,
                    n <= 1200 ? std::max(160, n * 2) : std::max(80, n / 2)
                );
            }

            if (capacity_ok) {
                std::vector<Answer> colored = color_selected_options(selected, rb_budget);
                if (!colored.empty()) {
                    colored = polish_classical_fixed_budget(
                        options_by_user,
                        std::move(colored),
                        rb_budget,
                        19000 + 101 * rb_budget + iter,
                        false
                    );
                    if (objective_value(colored) < best_obj) {
                        colored = polish_classical_fixed_budget(
                            options_by_user,
                            std::move(colored),
                            rb_budget,
                            21000 + 101 * rb_budget + iter,
                            true
                        );
                    }
                    consider(std::move(colored));
                }
            }

            for (int slot = 0; slot < L; ++slot) {
                double violation = static_cast<double>(dual_loads[slot] - rb_budget);
                prices[slot] = std::max(0.0, prices[slot] + step * violation);
            }
            step *= 0.970;
        }
    }

    return best;
}

std::vector<Answer> solve_flow320_upgrade_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool strong = true
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, ignored_upper] = rb_budget_window(options_by_user, strong ? 8 : 5, 0);
    (void)ignored_upper;
    int upper = std::min(R, lower + (strong ? 8 : 5));

    std::vector<Answer> best = solve_flow320_sweep_prepared(options_by_user, strong ? 8 : 5);
    int64_t best_obj = best.empty() ? std::numeric_limits<int64_t>::max() : objective_value(best);
    auto consider = [&](std::vector<Answer> candidate) {
        if (candidate.empty()) return;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> start = solve_flow320_budget_prepared(options_by_user, rb_budget);
        if (start.empty()) continue;
        consider(start);

        std::vector<Answer> current = polish_fixed_rb_quality_dynamic(
            options_by_user,
            start,
            rb_budget,
            n <= 1500 ? 6 : 3,
            1,
            23000 + rb_budget
        );
        consider(current);

        current = polish_ejection_chain_quality(
            options_by_user,
            current,
            rb_budget,
            n <= 1500 ? 5 : 2,
            n <= 1500 ? 4 : 2
        );
        consider(current);

        if (strong) {
            current = polish_tabu_fixed_rb_quality(
                options_by_user,
                current,
                rb_budget,
                n <= 1500 ? 70 : 18,
                9,
                n <= 1500 ? 22 : 14,
                n <= 1500 ? 650 : 300
            );
            consider(current);

            current = polish_fixed_pattern_flow(options_by_user, current);
            current = polish_period160_role_swaps(options_by_user, current, 8);
            current = polish_period160_double_role_swaps(options_by_user, current, 2, 40);
            current = polish_period160_role_swaps(options_by_user, current, 4);
            current = polish_fixed_period160_global_p320(options_by_user, current, rb_budget);
            current = polish_fixed_pattern_flow(options_by_user, current);
            current = polish_fixed_pattern_assignment(options_by_user, current);
            consider(current);
        }
    }

    return best;
}

std::vector<Answer> solve_single_cell_upgrade_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool strong = true
) {
    int n = static_cast<int>(options_by_user.size());
    auto [lower, ignored_upper] = rb_budget_window(options_by_user, strong ? 8 : 5, 0);
    (void)ignored_upper;
    int upper = std::min(R, lower + (strong ? 8 : 5));

    std::vector<Answer> best = solve_single_cell_flow_sweep_prepared(options_by_user, strong ? 8 : 5, 0);
    int64_t best_obj = best.empty() ? std::numeric_limits<int64_t>::max() : objective_value(best);
    auto consider = [&](std::vector<Answer> candidate) {
        if (candidate.empty()) return;
        if (compact_stats(candidate).unscheduled > 0) return;
        int64_t obj = objective_value(candidate);
        if (obj < best_obj) {
            best_obj = obj;
            best = std::move(candidate);
        }
    };

    for (int rb_budget = lower; rb_budget <= upper; ++rb_budget) {
        std::vector<Answer> start = solve_single_cell_flow_budget_prepared(options_by_user, rb_budget);
        if (compact_stats(start).unscheduled > 0) continue;
        consider(start);

        std::vector<Answer> current = polish_fixed_rb_quality_dynamic(
            options_by_user,
            start,
            rb_budget,
            n <= 1500 ? 6 : 3,
            1,
            33000 + rb_budget
        );
        consider(current);

        current = polish_ejection_chain_quality(
            options_by_user,
            current,
            rb_budget,
            n <= 1500 ? 5 : 2,
            n <= 1500 ? 4 : 2
        );
        consider(current);

        if (strong) {
            current = polish_tabu_fixed_rb_quality(
                options_by_user,
                current,
                rb_budget,
                n <= 1500 ? 70 : 18,
                9,
                n <= 1500 ? 22 : 14,
                n <= 1500 ? 650 : 300
            );
            consider(current);

            current = polish_fixed_pattern_flow(options_by_user, current);
            current = polish_period160_role_swaps(options_by_user, current, 8);
            current = polish_period160_double_role_swaps(options_by_user, current, 2, 40);
            current = polish_period160_role_swaps(options_by_user, current, 4);
            current = polish_fixed_period160_global_p320(options_by_user, current, rb_budget);
            current = polish_fixed_pattern_flow(options_by_user, current);
            current = polish_fixed_pattern_assignment(options_by_user, current);
            consider(current);
        }
    }

    return best;
}

bool better_solution(const std::vector<Answer>& candidate, const std::vector<Answer>& incumbent) {
    if (candidate.empty()) return false;
    if (incumbent.empty()) return true;
    SolutionStats candidate_stats = compact_stats(candidate);
    SolutionStats incumbent_stats = compact_stats(incumbent);
    if (candidate_stats.unscheduled != incumbent_stats.unscheduled) {
        return candidate_stats.unscheduled < incumbent_stats.unscheduled;
    }
    return candidate_stats.objective < incumbent_stats.objective;
}

std::vector<Answer> solve_price_relaxation_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
);

std::vector<Answer> solve_slot_capacity_regret_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user
);

std::vector<Answer> solve_priced_coloring_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    bool full_search
);

std::vector<Answer> solve_contextual_bandit_polish_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    const std::vector<Answer>& start_answers,
    int base_penalty,
    int polish_rounds,
    int seed
);

std::vector<Answer> solve_parallel_contextual_bandit_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    int rounds,
    int seed,
    int workers
);

std::vector<Answer> solve_universal_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty,
    int rounds,
    int seed,
    int workers,
    int candidate_cap
) {
    InstanceFeatures features = compute_instance_features(options_by_user);
    std::vector<Answer> best;

    auto consider = [&](std::vector<Answer> candidate) {
        if (better_solution(candidate, best)) best = std::move(candidate);
    };

    auto has_feasible_best = [&]() {
        return !best.empty() && compact_stats(best).unscheduled == 0;
    };

    auto fallback_feasibility = [&]() {
        consider(solve_sparse_capacity_prepared(options_by_user, true));
        if (!features.all_one_cell_feasible || features.n <= 1200) {
            consider(solve_sparse_capacity_prepared(options_by_user, false));
        }
        if (features.one_cell_ratio >= 0.80) {
            int extra = features.pressure > 0.80 ? 2 : 5;
            consider(solve_single_cell_flow_sweep_prepared(options_by_user, extra, 0));
        }
        if (!has_feasible_best()) {
            consider(solve_adaptive_prepared(options_by_user, base_penalty));
            consider(solve_saturation_prepared(options_by_user));
            if (features.one_cell_ratio >= 0.50) {
                consider(solve_assignment_sweep_prepared(options_by_user));
            }
        }
    };

    // If all users have one-cell options, start directly with the exact-start
    // upgrade appropriate for the size. Running a separate flow/pricing branch
    // first duplicates work and was the main universal-mode speed penalty.
    if (features.all_one_cell_feasible) {
        if (features.n > 2500 || features.pressure > 0.85) {
            consider(solve_assignment_upgrade_lean_prepared(options_by_user));
            if (!has_feasible_best()) fallback_feasibility();
            return best;
        }

        if (features.n <= 600) {
            consider(solve_sparse_two_cell_upgrade_prepared(options_by_user));
            if (has_feasible_best()) return best;
            if (candidate_cap > 0 && candidate_cap <= 32) {
                consider(solve_parallel_contextual_bandit_prepared(
                    options_by_user,
                    base_penalty,
                    std::max(8, rounds),
                    seed + 66000,
                    workers
                ));
                if (has_feasible_best()) return best;
            }
            consider(solve_single_cell_upgrade_prepared(options_by_user, true));
        } else {
            consider(solve_assignment_upgrade_sweep_prepared(options_by_user, false));
        }

        if (!has_feasible_best()) fallback_feasibility();
        if (has_feasible_best() && rounds > 0 && features.n <= 1200 && features.n > 600) {
            consider(solve_contextual_bandit_polish_prepared(
                options_by_user,
                best,
                base_penalty,
                std::min(rounds, 8),
                seed + 55000
            ));
        }
        return best;
    }

    // Feasibility-first incumbents for mixed/messier instances. These protect
    // against the unallocated-user pathology from the feedback letter.
    consider(solve_sparse_capacity_prepared(options_by_user, true));
    consider(solve_sparse_capacity_prepared(options_by_user, false));
    if (features.one_cell_ratio >= 0.80) {
        int extra = features.pressure > 0.80 ? 2 : 5;
        consider(solve_single_cell_flow_sweep_prepared(options_by_user, extra, 0));
    }

    SolutionStats best_stats = compact_stats(best);
    if (best.empty() || best_stats.unscheduled > 0) {
        consider(solve_adaptive_prepared(options_by_user, base_penalty));
        consider(solve_saturation_prepared(options_by_user));
        if (features.one_cell_ratio >= 0.50) {
            consider(solve_assignment_sweep_prepared(options_by_user));
        }
    }

    best_stats = compact_stats(best);
    bool has_all_allocated = !best.empty() && best_stats.unscheduled == 0;

    // If one-cell coverage is weak, fall back to fixed-K slot pricing and
    // DSATUR-style coloring rather than forcing the flow kernel.
    if (has_all_allocated) {
        if (features.n <= 1200) {
            consider(solve_price_relaxation_prepared(options_by_user));
            if (features.n <= 800 || features.one_cell_ratio < 0.65) {
                consider(solve_slot_capacity_regret_prepared(options_by_user));
            }
        } else {
            consider(solve_assignment_upgrade_sweep_prepared(options_by_user, false));
        }
    }

    best_stats = compact_stats(best);
    if (!best.empty() && best_stats.unscheduled == 0 && rounds > 0 && features.n <= 1500) {
        consider(solve_contextual_bandit_polish_prepared(
            options_by_user,
            best,
            base_penalty,
            std::min(rounds, features.n <= 800 ? 16 : 10),
            seed + 55000
        ));
    }

    return best;
}

std::vector<Answer> solve_adaptive_safe_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty
) {
    auto quality24 = limited_options_by_user(options_by_user, 24);
    std::vector<std::vector<Answer>> candidates;
    for (int penalty : {0, 15, 30, base_penalty}) {
        candidates.push_back(solve_adaptive_prepared(options_by_user, penalty));
        candidates.push_back(solve_adaptive_prepared(quality24, penalty));
    }
    candidates.push_back(solve_saturation_prepared(options_by_user));
    candidates.push_back(solve_sparse_capacity_prepared(options_by_user, false));
    candidates.push_back(solve_sparse_capacity_prepared(options_by_user, true));
    candidates.push_back(solve_assignment_sweep_prepared(options_by_user));
    candidates.push_back(solve_assignment_upgrade_sweep_prepared(options_by_user));
    if (static_cast<int>(options_by_user.size()) <= 1000) {
        candidates.push_back(solve_slot_capacity_regret_prepared(options_by_user));
    }

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

std::vector<Answer> solve_portfolio_prepared(
    const std::vector<std::vector<TimeOption>>& options_by_user,
    int base_penalty
) {
    std::vector<std::vector<Answer>> candidates;
    auto quality24 = limited_options_by_user(options_by_user, 24);
    auto quality64 = limited_options_by_user(options_by_user, 64);
    for (int penalty : {0, 15, 30, 40, 60, 80}) {
        candidates.push_back(solve_greedy_prepared(quality24, penalty, "adaptive"));
    }
    candidates.push_back(solve_greedy_prepared(quality24, base_penalty, "marginal"));
    candidates.push_back(solve_greedy_prepared(quality64, base_penalty, "marginal"));
    candidates.push_back(solve_sparse_capacity_prepared(options_by_user, false));
    candidates.push_back(solve_sparse_capacity_prepared(options_by_user, true));
    candidates.push_back(solve_assignment_sweep_prepared(options_by_user));
    candidates.push_back(solve_assignment_upgrade_sweep_prepared(options_by_user));
    if (static_cast<int>(options_by_user.size()) <= 1000) {
        candidates.push_back(solve_slot_capacity_regret_prepared(options_by_user));
    }
    candidates.push_back(solve_saturation_prepared(options_by_user));
    for (int penalty : {0, 15, 30, 40, 60, 80}) {
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
    int last_gain_type,
    int rb_capacity_slots
) {
    if (no_improve_rounds >= 5) return BanditContext::Plateau;
    if (last_gain_type == 1 && no_improve_rounds <= 2) return BanditContext::QualityPolish;

    auto loads = rb_loads_from_answers(answers);
    std::vector<int> active_loads;
    for (int load : loads) {
        if (load > 0) active_loads.push_back(load);
    }
    if (active_loads.empty()) return BanditContext::QualityPolish;

    int rb_capacity = std::max(1, rb_capacity_slots);
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
    int rb_capacity = seen_uplink_slot_count(options_by_user);
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
    int rb_capacity_slots = seen_uplink_slot_count(options_by_user);
    std::mt19937 rng(seed + 91000);
    int n = static_cast<int>(options_by_user.size());
    int base_destroy = std::max(8, std::min(30, n / 13));
    double temperature = std::max(50.0, double(current_obj) * 0.005);
    int no_improve_rounds = 0;
    int last_gain_type = 0;  // 0 none/objective, 1 rb, 2 quality

    for (int round = 0; round < polish_rounds; ++round) {
        if (round >= 8 && no_improve_rounds >= 7) break;

        BanditContext context = contextual_bandit_context_cpp(
            current,
            current_stats,
            no_improve_rounds,
            last_gain_type,
            rb_capacity_slots
        );
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
    int cap = 64;
    int base_penalty = 60;
    int rounds = 28;
    int workers = 2;
    int dl = 8;
    int ul = 2;
    std::string difficulty = "large";
    std::string input_jsonl;
    std::string dump_jsonl;
    std::string mode = "universal";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "--seeds" && i + 1 < argc) seeds = std::stoi(argv[++i]);
        else if (arg == "--cap" && i + 1 < argc) cap = std::stoi(argv[++i]);
        else if (arg == "--base-penalty" && i + 1 < argc) base_penalty = std::stoi(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc) rounds = std::stoi(argv[++i]);
        else if (arg == "--workers" && i + 1 < argc) workers = std::stoi(argv[++i]);
        else if (arg == "--dl" && i + 1 < argc) dl = std::stoi(argv[++i]);
        else if (arg == "--ul" && i + 1 < argc) ul = std::stoi(argv[++i]);
        else if (arg == "--difficulty" && i + 1 < argc) difficulty = argv[++i];
        else if (arg == "--input-jsonl" && i + 1 < argc) input_jsonl = argv[++i];
        else if (arg == "--dump-jsonl" && i + 1 < argc) dump_jsonl = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    }

    double total_ms = 0.0;
    int64_t total_obj = 0;
    int cases = 0;

    double total_prepare_ms = 0.0;
    double total_end_to_end_ms = 0.0;
    std::ofstream dump_out;
    if (!dump_jsonl.empty()) dump_out.open(dump_jsonl);

    auto run_case = [&](const std::vector<User>& users, int case_id) {
        auto prepare_start = std::chrono::steady_clock::now();
        bool online_mode = mode == "online" || mode == "sparse-online";
        auto options_by_user = online_mode
            ? prepare_sparse_online_options(users, dl, ul)
            : prepare_options(users, dl, ul, cap);
        auto prepare_end = std::chrono::steady_clock::now();
        auto start = std::chrono::steady_clock::now();
        std::vector<Answer> answers;
        if (mode == "priced-coloring" || mode == "classical" || mode == "classical-priced") {
            answers = solve_priced_coloring_prepared(options_by_user);
        } else if (mode == "priced-coloring-full" || mode == "classical-priced-full") {
            answers = solve_priced_coloring_prepared(options_by_user, true);
        } else if (mode == "annealed") {
            answers = solve_annealed_prepared(options_by_user, base_penalty, rounds, 7000 + case_id);
        } else if (mode == "bandit") {
            answers = solve_contextual_bandit_prepared(options_by_user, base_penalty, rounds, 7000 + case_id);
        } else if (mode == "adaptive-fast") {
            answers = solve_adaptive_prepared(options_by_user, base_penalty);
        } else if (mode == "online" || mode == "sparse-online") {
            answers = solve_sparse_capacity_prepared(options_by_user, true);
        } else if (mode == "single-cell-flow" || mode == "single_cell_flow") {
            answers = solve_single_cell_flow_sweep_prepared(options_by_user);
        } else if (mode == "single-cell-auction" || mode == "single_cell_auction") {
            answers = solve_single_cell_auction_sweep_prepared(options_by_user);
        } else if (mode == "single-cell-auction-upgrade" || mode == "single_cell_auction_upgrade") {
            answers = solve_single_cell_auction_upgrade_prepared(options_by_user, true);
        } else if (mode == "single-cell-auction-upgrade-fast" || mode == "single_cell_auction_upgrade_fast") {
            answers = solve_single_cell_auction_upgrade_prepared(options_by_user, false);
        } else if (mode == "sparse-two-cell-upgrade" || mode == "sparse_two_cell_upgrade") {
            answers = solve_sparse_two_cell_upgrade_prepared(options_by_user);
        } else if (mode == "dense-two-cell-exchange" || mode == "dense_two_cell_exchange") {
            answers = solve_dense_two_cell_exchange_prepared(options_by_user);
        } else if (mode == "single-cell-upgrade" || mode == "single_cell_upgrade") {
            answers = solve_single_cell_upgrade_prepared(options_by_user, true);
        } else if (mode == "single-cell-upgrade-fast" || mode == "single_cell_upgrade_fast") {
            answers = solve_single_cell_upgrade_prepared(options_by_user, false);
        } else if (mode == "flow320") {
            answers = solve_flow320_sweep_prepared(options_by_user);
        } else if (mode == "flow320-upgrade") {
            answers = solve_flow320_upgrade_prepared(options_by_user, true);
        } else if (mode == "flow320-upgrade-fast") {
            answers = solve_flow320_upgrade_prepared(options_by_user, false);
        } else if (mode == "flow320-lns") {
            std::vector<Answer> start_answers = solve_flow320_upgrade_prepared(options_by_user, true);
            if (start_answers.empty()) start_answers = solve_assignment_upgrade_sweep_prepared(options_by_user, true);
            answers = solve_contextual_bandit_polish_prepared(
                options_by_user,
                start_answers,
                base_penalty,
                std::max(8, rounds),
                17000 + case_id
            );
        } else if (mode == "assignment") {
            answers = solve_assignment_sweep_prepared(options_by_user);
        } else if (mode == "assignment-upgrade") {
            answers = solve_assignment_upgrade_sweep_prepared(options_by_user);
        } else if (mode == "assignment-upgrade-flow") {
            answers = solve_assignment_upgrade_sweep_prepared(options_by_user, true);
        } else if (mode == "assignment-upgrade-exchange" || mode == "assignment_upgrade_exchange") {
            answers = solve_assignment_upgrade_exchange_prepared(options_by_user, false);
        } else if (mode == "assignment-upgrade-lean" || mode == "assignment_upgrade_lean") {
            answers = solve_assignment_upgrade_lean_prepared(options_by_user);
        } else if (mode == "assignment-upgrade-diagnostics" || mode == "assignment_upgrade_diagnostics") {
            answers = solve_assignment_upgrade_diagnostics_prepared(options_by_user);
        } else if (mode == "price-relax") {
            answers = solve_price_relaxation_prepared(options_by_user);
        } else if (mode == "slot-regret") {
            answers = solve_slot_capacity_regret_prepared(options_by_user);
        } else if (mode == "sparse-quality") {
            answers = solve_sparse_capacity_prepared(options_by_user, false);
        } else if (mode == "sparse-balanced") {
            answers = solve_sparse_capacity_prepared(options_by_user, true);
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
        } else if (mode == "adaptive") {
            answers = solve_adaptive_safe_prepared(options_by_user, base_penalty);
        } else if (mode == "universal") {
            answers = solve_universal_prepared(options_by_user, base_penalty, rounds, 7000 + case_id, workers, cap);
        } else {
            answers = solve_priced_coloring_prepared(options_by_user);
        }
        auto end = std::chrono::steady_clock::now();
        auto [rb_used, quality_sum, unscheduled] = stats(answers);
        int64_t objective = objective_value(answers);
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double prepare_ms = std::chrono::duration<double, std::milli>(prepare_end - prepare_start).count();
        double end_to_end_ms = prepare_ms + ms;
        total_ms += ms;
        total_prepare_ms += prepare_ms;
        total_end_to_end_ms += end_to_end_ms;
        total_obj += objective;
        ++cases;
        std::cout << "case=" << case_id
                  << " obj=" << objective
                  << " rb=" << rb_used
                  << " q=" << quality_sum
                  << " uns=" << unscheduled
                  << " prepare_ms=" << prepare_ms
                  << " solve_ms=" << ms
                  << " total_ms=" << end_to_end_ms << "\n";
        if (dump_out) {
            dump_out << "{\"case\":" << case_id << ",\"users\":[";
            for (int i = 0; i < static_cast<int>(users.size()); ++i) {
                if (i) dump_out << ",";
                dump_out << "{\"csi_periods\":[";
                for (int j = 0; j < static_cast<int>(users[i].csi_periods.size()); ++j) {
                    if (j) dump_out << ",";
                    dump_out << users[i].csi_periods[j];
                }
                dump_out << "],\"srs_period\":" << users[i].srs_period << ",\"srs_offsets\":[";
                for (int j = 0; j < static_cast<int>(users[i].srs_offsets.size()); ++j) {
                    if (j) dump_out << ",";
                    dump_out << users[i].srs_offsets[j];
                }
                dump_out << "]}";
            }
            dump_out << "],\"answers\":[";
            for (int i = 0; i < static_cast<int>(answers.size()); ++i) {
                if (i) dump_out << ",";
                dump_out << "{\"user_id\":" << i
                         << ",\"rb\":" << answers[i].rb
                         << ",\"period\":" << answers[i].period
                         << ",\"offset\":" << answers[i].offset
                         << ",\"distance\":" << answers[i].distance
                         << ",\"quality\":" << answers[i].quality
                         << "}";
            }
            dump_out << "]}\n";
        }
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
                  << " avg_solve_ms=" << (total_ms / cases)
                  << " avg_total_ms=" << (total_end_to_end_ms / cases) << "\n";
    }
    return 0;
}
