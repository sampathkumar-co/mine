#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t kBudget = 250000;

struct BudgetExceeded : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct RemainingMask {
    std::uint64_t low{0};
    std::uint64_t high{0};

    bool operator==(const RemainingMask& other) const {
        return low == other.low && high == other.high;
    }

    bool test(int index) const {
        if (index < 64) return ((low >> index) & 1ULL) != 0;
        return ((high >> (index - 64)) & 1ULL) != 0;
    }

    void set(int index) {
        if (index < 64) low |= (1ULL << index);
        else high |= (1ULL << (index - 64));
    }

    RemainingMask without(int index) const {
        RemainingMask result = *this;
        if (index < 64) result.low &= ~(1ULL << index);
        else result.high &= ~(1ULL << (index - 64));
        return result;
    }

    int count() const {
        return std::popcount(low) + std::popcount(high);
    }
};

struct StateInput {
    std::string digest;
    int profile_seed{0};
    int hypothesis_count{0};
    int query_count{0};
    std::vector<int> labels;
    std::vector<std::int64_t> masses;
    std::vector<int> query_ids;
    std::vector<std::int64_t> costs;
    std::vector<std::vector<int>> matrix;
    std::vector<std::vector<std::uint32_t>> response_masks;

    void build_response_masks() {
        response_masks.assign(query_count, {});
        for (int query = 0; query < query_count; ++query) {
            std::map<int, std::uint32_t> buckets;
            for (int row = 0; row < hypothesis_count; ++row) {
                buckets[matrix[row][query]] |= (1U << row);
            }
            for (const auto& [_, mask] : buckets) {
                response_masks[query].push_back(mask);
            }
        }
    }
};

struct Plan {
    std::int64_t diagnosed_mass{0};
    std::int64_t expected_cost{0};
    std::int64_t worst_cost{0};
    int query_id{-1};
};

bool better(const Plan& left, const Plan& right) {
    if (left.diagnosed_mass != right.diagnosed_mass) {
        return left.diagnosed_mass > right.diagnosed_mass;
    }
    if (left.expected_cost != right.expected_cost) {
        return left.expected_cost < right.expected_cost;
    }
    if (left.worst_cost != right.worst_cost) {
        return left.worst_cost < right.worst_cost;
    }
    const int left_query = left.query_id < 0 ? 1000000000 : left.query_id;
    const int right_query = right.query_id < 0 ? 1000000000 : right.query_id;
    return left_query < right_query;
}

struct MemoKey {
    std::uint32_t allowed{0};
    RemainingMask remaining;

    bool operator==(const MemoKey& other) const {
        return allowed == other.allowed && remaining == other.remaining;
    }
};

struct MemoKeyHash {
    std::size_t operator()(const MemoKey& key) const {
        std::size_t value = static_cast<std::size_t>(key.allowed) * 0x9e3779b1U;
        value ^= static_cast<std::size_t>(key.remaining.low ^ (key.remaining.low >> 32));
        value ^= static_cast<std::size_t>(key.remaining.high ^ (key.remaining.high >> 32)) << 1;
        return value;
    }
};

struct Stats {
    std::uint64_t calls{0};
    std::uint64_t memo_entries{0};
    std::uint64_t query_expansions{0};
    std::uint64_t memo_hits{0};
    std::uint64_t raw_queries_considered{0};
    std::uint64_t representative_queries_considered{0};
};

class ExactSolver {
public:
    ExactSolver(const StateInput& input, bool local)
        : input_(input), local_(local) {
        memo_.reserve(300000);
    }

    Plan solve_root() {
        RemainingMask remaining;
        for (int query = 0; query < input_.query_count; ++query) {
            remaining.set(query);
        }
        const std::uint32_t allowed =
            input_.hypothesis_count == 32
                ? 0xffffffffU
                : ((1U << input_.hypothesis_count) - 1U);
        Plan result = solve(allowed, remaining);
        stats_.memo_entries = memo_.size();
        return result;
    }

    const Stats& stats() const { return stats_; }

private:
    const StateInput& input_;
    bool local_;
    Stats stats_;
    std::unordered_map<MemoKey, Plan, MemoKeyHash> memo_;

    std::vector<std::uint32_t> partition(
        std::uint32_t allowed,
        int query
    ) const {
        std::vector<std::uint32_t> children;
        for (std::uint32_t mask : input_.response_masks[query]) {
            const std::uint32_t child = allowed & mask;
            if (child != 0) children.push_back(child);
        }
        std::sort(children.begin(), children.end());
        return children;
    }

    RemainingMask canonical(
        std::uint32_t allowed,
        const RemainingMask& remaining
    ) {
        stats_.raw_queries_considered += remaining.count();
        std::map<std::vector<std::uint32_t>, int> representatives;
        for (int query = 0; query < input_.query_count; ++query) {
            if (!remaining.test(query)) continue;
            auto signature = partition(allowed, query);
            if (signature.size() <= 1) continue;
            auto found = representatives.find(signature);
            if (found == representatives.end()) {
                representatives.emplace(std::move(signature), query);
                continue;
            }
            const int previous = found->second;
            const auto current_key = std::pair{
                input_.costs[query], input_.query_ids[query]
            };
            const auto previous_key = std::pair{
                input_.costs[previous], input_.query_ids[previous]
            };
            if (current_key < previous_key) found->second = query;
        }
        RemainingMask result;
        for (const auto& [_, query] : representatives) result.set(query);
        stats_.representative_queries_considered += result.count();
        return result;
    }

    bool pure_label(std::uint32_t allowed) const {
        int first_label = -1;
        for (int index = 0; index < input_.hypothesis_count; ++index) {
            if (((allowed >> index) & 1U) == 0) continue;
            if (first_label < 0) first_label = input_.labels[index];
            else if (input_.labels[index] != first_label) return false;
        }
        return true;
    }

    std::int64_t mass(std::uint32_t allowed) const {
        std::int64_t total = 0;
        for (int index = 0; index < input_.hypothesis_count; ++index) {
            if (((allowed >> index) & 1U) != 0) total += input_.masses[index];
        }
        return total;
    }

    Plan solve(std::uint32_t allowed, RemainingMask remaining) {
        ++stats_.calls;
        if (local_) remaining = canonical(allowed, remaining);
        const MemoKey key{allowed, remaining};
        const auto cached = memo_.find(key);
        if (cached != memo_.end()) {
            ++stats_.memo_hits;
            return cached->second;
        }
        const std::int64_t state_mass = mass(allowed);
        if (pure_label(allowed)) {
            const Plan answer{state_mass, 0, 0, -1};
            memo_.emplace(key, answer);
            return answer;
        }

        bool found_candidate = false;
        Plan best;
        if (!local_) {
            stats_.raw_queries_considered += remaining.count();
            stats_.representative_queries_considered += remaining.count();
        }
        for (int query = 0; query < input_.query_count; ++query) {
            if (!remaining.test(query)) continue;
            const auto children = partition(allowed, query);
            if (children.size() <= 1) continue;
            ++stats_.query_expansions;
            if (stats_.query_expansions > kBudget) {
                throw BudgetExceeded("compiled exact-search budget exceeded");
            }
            const RemainingMask next = remaining.without(query);
            Plan candidate;
            candidate.query_id = input_.query_ids[query];
            candidate.expected_cost = input_.costs[query] * state_mass;
            candidate.worst_cost = input_.costs[query];
            std::int64_t child_worst = 0;
            for (std::uint32_t child : children) {
                const Plan child_plan = solve(child, next);
                candidate.diagnosed_mass += child_plan.diagnosed_mass;
                candidate.expected_cost += child_plan.expected_cost;
                child_worst = std::max(child_worst, child_plan.worst_cost);
            }
            candidate.worst_cost += child_worst;
            if (!found_candidate || better(candidate, best)) {
                best = candidate;
                found_candidate = true;
            }
        }
        const Plan answer = found_candidate ? best : Plan{};
        memo_.emplace(key, answer);
        return answer;
    }
};

std::vector<StateInput> read_states(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open input states");
    std::string token;
    int count = 0;
    input >> token >> count;
    if (token != "COUNT" || count < 0) {
        throw std::runtime_error("invalid state header");
    }
    std::vector<StateInput> states;
    states.reserve(count);
    for (int state_index = 0; state_index < count; ++state_index) {
        StateInput state;
        input >> token;
        if (token != "STATE") throw std::runtime_error("expected STATE");
        input >> state.digest >> state.profile_seed
              >> state.hypothesis_count >> state.query_count;
        if (state.hypothesis_count <= 0 || state.hypothesis_count > 31) {
            throw std::runtime_error("unsupported hypothesis count");
        }
        if (state.query_count <= 0 || state.query_count > 128) {
            throw std::runtime_error("unsupported query count");
        }
        state.labels.resize(state.hypothesis_count);
        state.masses.resize(state.hypothesis_count);
        state.query_ids.resize(state.query_count);
        state.costs.resize(state.query_count);
        state.matrix.assign(
            state.hypothesis_count,
            std::vector<int>(state.query_count)
        );
        input >> token;
        if (token != "LABELS") throw std::runtime_error("expected LABELS");
        for (int& value : state.labels) input >> value;
        input >> token;
        if (token != "MASSES") throw std::runtime_error("expected MASSES");
        for (std::int64_t& value : state.masses) input >> value;
        input >> token;
        if (token != "QUERY_IDS") throw std::runtime_error("expected QUERY_IDS");
        for (int& value : state.query_ids) input >> value;
        input >> token;
        if (token != "COSTS") throw std::runtime_error("expected COSTS");
        for (std::int64_t& value : state.costs) input >> value;
        for (int row = 0; row < state.hypothesis_count; ++row) {
            input >> token;
            if (token != "ROW") throw std::runtime_error("expected ROW");
            for (int& value : state.matrix[row]) input >> value;
        }
        input >> token;
        if (token != "END") throw std::runtime_error("expected END");
        state.build_response_masks();
        states.push_back(std::move(state));
    }
    return states;
}

struct RowResult {
    std::string digest;
    bool solved{false};
    Plan plan;
    Stats stats;
    double milliseconds{0.0};
};

long peak_rss_kb() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

void write_json(
    const std::string& path,
    const std::string& mode,
    const std::vector<RowResult>& rows,
    double total_ms
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open output JSON");
    int solved_count = 0;
    std::uint64_t total_expansions = 0;
    for (const auto& row : rows) {
        solved_count += row.solved ? 1 : 0;
        total_expansions += row.stats.query_expansions;
    }
    output << std::setprecision(17);
    output << "{\n";
    output << "  \"mode\": \"" << mode << "\",\n";
    output << "  \"state_count\": " << rows.size() << ",\n";
    output << "  \"solved_count\": " << solved_count << ",\n";
    output << "  \"total_query_expansions\": " << total_expansions << ",\n";
    output << "  \"total_milliseconds\": " << total_ms << ",\n";
    output << "  \"peak_rss_kb\": " << peak_rss_kb() << ",\n";
    output << "  \"rows\": [\n";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        output << "    {\"digest\": \"" << row.digest << "\", ";
        output << "\"solved\": " << (row.solved ? "true" : "false") << ", ";
        if (row.solved) {
            output << "\"plan\": ["
                   << row.plan.diagnosed_mass << ", "
                   << row.plan.expected_cost << ", "
                   << row.plan.worst_cost << "], ";
        } else {
            output << "\"plan\": null, ";
        }
        output << "\"query_expansions\": " << row.stats.query_expansions << ", ";
        output << "\"memo_entries\": " << row.stats.memo_entries << ", ";
        output << "\"calls\": " << row.stats.calls << ", ";
        output << "\"memo_hits\": " << row.stats.memo_hits << ", ";
        output << "\"raw_queries_considered\": "
               << row.stats.raw_queries_considered << ", ";
        output << "\"representative_queries_considered\": "
               << row.stats.representative_queries_considered << ", ";
        output << "\"milliseconds\": " << row.milliseconds << "}";
        if (index + 1 != rows.size()) output << ',';
        output << '\n';
    }
    output << "  ]\n}\n";
}

bool self_test() {
    StateInput state;
    state.digest = "self-test";
    state.profile_seed = 1;
    state.hypothesis_count = 4;
    state.query_count = 3;
    state.labels = {0, 1, 1, 0};
    state.masses = {1, 3, 5, 7};
    state.query_ids = {0, 1, 2};
    state.costs = {9, 2, 5};
    state.matrix = {
        {0, 0, 0},
        {0, 0, 1},
        {1, 1, 0},
        {1, 1, 1},
    };
    state.build_response_masks();
    ExactSolver local(state, true);
    ExactSolver plain(state, false);
    const Plan local_plan = local.solve_root();
    const Plan plain_plan = plain.solve_root();
    const bool same =
        local_plan.diagnosed_mass == plain_plan.diagnosed_mass &&
        local_plan.expected_cost == plain_plan.expected_cost &&
        local_plan.worst_cost == plain_plan.worst_cost;
    return same &&
           local.stats().query_expansions < plain.stats().query_expansions;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            if (!self_test()) {
                std::cerr << "self-test failed\n";
                return 1;
            }
            std::cout << "self-test passed\n";
            return 0;
        }
        if (argc != 7 || std::string(argv[1]) != "--input" ||
            std::string(argv[3]) != "--mode" ||
            std::string(argv[5]) != "--output") {
            std::cerr << "usage: weighted_quotient_v49 --input FILE "
                         "--mode local|plain --output FILE\n";
            return 2;
        }
        const std::string input_path = argv[2];
        const std::string mode = argv[4];
        const std::string output_path = argv[6];
        if (mode != "local" && mode != "plain") {
            throw std::runtime_error("mode must be local or plain");
        }
        const bool local_mode = mode == "local";
        const auto states = read_states(input_path);
        std::vector<RowResult> rows;
        rows.reserve(states.size());
        const auto total_start = std::chrono::steady_clock::now();
        for (const auto& state : states) {
            RowResult row;
            row.digest = state.digest;
            const auto start = std::chrono::steady_clock::now();
            ExactSolver solver(state, local_mode);
            try {
                row.plan = solver.solve_root();
                row.solved = true;
            } catch (const BudgetExceeded&) {
                row.solved = false;
            }
            const auto end = std::chrono::steady_clock::now();
            row.stats = solver.stats();
            row.milliseconds = std::chrono::duration<double, std::milli>(
                end - start
            ).count();
            rows.push_back(row);
        }
        const auto total_end = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(
            total_end - total_start
        ).count();
        write_json(output_path, mode, rows, total_ms);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
