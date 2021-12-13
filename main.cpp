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
        const Matrix& dist, uint8_t pos, uint16_t set, float path_length, /*in*/
        float& best /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
#ifdef USE_CRITICAL
#pragma omp critical
#endif
            if (length < best) {
                best = length;
            }
        }

#if defined (USE_CRITICAL)
#pragma omp parallel for default(none) shared(dist, pos, path_length, set) shared(best)  num_threads(NUM_THREADS) schedule(dynamic)
#elif defined (USE_REDUCE)
#pragma omp parallel for default(none) shared(dist, pos, path_length, set) num_threads(NUM_THREADS) reduction(min: best) schedule(dynamic)
#else
#error No strategy used
#endif
        for (std::size_t next = 0; next < dist.size(); ++next) {
            if (set & (1u << next)) { /* if next position is usable */
                const auto extended_len = path_length + dist[pos][next];
                if (extended_len < best) { /* extend the path */
                    summon_solve_par(dist, next, set & ~(1u << next), extended_len, best);
                }
            }
        }
    }

    float solve (Matrix& distances) {
        assert(!distances.empty());
        assert(distances.size() == distances[0].size());

        const auto mask = (1 << distances.size()) - 1;
        float best /*out*/ = INFINITY;
        summon_solve_par(distances, 0, mask & ~1, 0, best);
        return best;
    }
};

int main () {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    freopen("../in.txt", "r", stdin);

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

    Timer timer;
    for (int i = 0; i < 1000; ++i)
    {
        std::cout << BBPar::solve(distances) << '\n';
    }
    return 0;
}
