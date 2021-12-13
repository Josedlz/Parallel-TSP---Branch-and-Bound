#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>
#include <omp.h>
#include <bitset>
#include "Timer.hpp"


template<class ... Args>
void crit_print (Args... args) {
#pragma omp critical
    ((std::cout << std::forward<Args>(args)), ...);
}

//#pragma omp declare reduction( min: float: omp_out=std::min( omp_out, omp_in ) )

namespace BBSeq {
    using Matrix = std::vector<std::vector<float>>;

    void
    summon_solve (
        const Matrix& dist, uint8_t pos, uint32_t set, float path_length, /*in*/
        float& best /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
            if (length < best) {
                best = length;
            }
        }
//#pragma omp parallel for default(none) shared(dist, pos, path_length, set) reduction(min: best) num_threads(8)
//#pragma omp parallel for default(none) schedule(dynamic) shared(dist, pos, path_length, set) shared(best) num_threads(4)
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
    summon_solve (
        const Matrix& dist, uint8_t pos, uint32_t set, float path_length, /*in*/
        float& best /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
#pragma omp critical
            if (length < best) {
                best = length;
            }
        }
        //#pragma omp parallel for default(none) shared(dist, pos, path_length, set) reduction(min: best) num_threads(8)
#pragma omp parallel for default(none) schedule(dynamic) shared(dist, pos, path_length, set) shared(best) num_threads(7)
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

int main () {
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
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << distances[i][j] << ' ';
        }
        std::cout << '\n';
    }
    {
        Timer timer;
        std::cout << BBPar::solve(distances) << '\n';
    }
    return 0;
}
