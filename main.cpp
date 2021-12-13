#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <bitset>
#include "Timer.hpp"

#define NUM_THREADS 4
#define USE_CRITICAL
//#define USE_REDUCE


namespace BBSeq {
    using Matrix = std::vector<std::vector<float>>;

    void
    summon_solve (
        const Matrix& dist, uint8_t pos, uint16_t set, float path_length, /*in*/
        float& best /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
            if (length < best) {
                best = length;
            }
            return;
        }
        for (std::size_t next = 0; next < dist.size(); ++next) {
            if (set & (1u << next)) { /* if next position is usable */
                const auto extended_len = path_length + dist[pos][next];
                if (extended_len < best) { /* extend the path */
                    summon_solve(dist, next, set & ~(1u << next), extended_len, best);
                }
            }
        }
    }

    float solve (const Matrix& distances) {
        assert(!distances.empty());
        assert(distances.size() == distances[0].size());

        const auto mask = (1 << distances.size()) - 1;
        float best /*out*/ = INFINITY;
        summon_solve(distances, 0, mask & ~1, 0, best);
        return best;
    }
};

namespace BBPar {
    using Matrix = std::vector<std::vector<float>>;

    void
    summon_solve_par (
        const Matrix& dist, uint_fast8_t pos, uint_fast16_t set, float path_length, /*in*/
        float& best /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
#pragma omp critical
            if (length < best) {
                best = length;
            }
            return;
        }

        uint_fast8_t next_positions[16];
        uint_fast8_t next_pos_size = 0;

        for (uint_fast8_t next = 0; next < (uint_fast8_t) dist.size(); ++next) {
            if (set & (1u << next)) { /* if next position is usable */
                const auto extended_len = path_length + dist[pos][next];
                if (extended_len < best) { /* extend the path */
                    next_positions[next_pos_size++] = next;
                }
            }
        }

        uint_fast8_t it = 0;
        for (; it + 1 < next_pos_size; ++it) {
            auto next = next_positions[it];
            auto extended_len = path_length + dist[pos][next];
#pragma omp task default(none) shared(dist, next, set, best, extended_len)
            summon_solve_par(dist, next, set & ~(1u << next), extended_len, best);
        }
        if (next_pos_size) {
            auto next = next_positions[it];
            auto extended_len = path_length + dist[pos][next];
            summon_solve_par(dist, next, set & ~(1u << next), extended_len, best);
        }
    }

    float solve (Matrix& distances) {
        assert(!distances.empty());
        assert(distances.size() == distances[0].size());

        const auto mask = (1 << distances.size()) - 1;
        float best /*out*/ = INFINITY;
#pragma omp parallel default(none) shared(distances, mask, best) num_threads(8)
        {
#pragma omp single
            summon_solve_par(distances, 0, mask & ~1, 0, best);
        }
        return best;
    }
};

int main () {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    freopen("../in.txt", "r", stdin);
    freopen("nul", "w", stderr);

    int n, m;
    std::vector<std::vector<float>> distances;
    std::cin >> n >> m;
    distances.resize(n, std::vector<float>(n, INFINITY));
    for (int i = 0; i < n; ++i) {
        distances[i][i] = 0;
    }

    for (int i = 0; i < m; ++i) {
        int u, v;
        float w;
        std::cin >> u >> v >> w;
        distances[u][v] = distances[v][u] = w;
    }
    {
        Timer timer;
        std::cout << BBSeq::solve(distances) << '\n';
        for (int i = 1; i < 1000; ++i) {
            std::cerr << BBSeq::solve(distances) << '\n';
        }
    }
    {
        Timer timer;
        std::cout << BBPar::solve(distances) << '\n';
        for (int i = 1; i < 1000; ++i) {
            std::cerr << BBPar::solve(distances) << '\n';
        }
    }
    return 0;
}
