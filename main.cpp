#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <bitset>
#include <numeric>
#include <algorithm>
#include <random>
#include <stack>
#include <omp.h>
#include "Timer.hpp"

#ifndef THREADS
#define THREADS 1
#endif

constexpr uint8_t N = 10;
namespace BBSeq {
    void
    summon_solve (
        const float dist[N][N],
        const uint8_t pos, const uint16_t set, const float path_length, /*in*/
        float& best                                                     /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
            best = length < best ? length : best;
            return;
        }
        for (std::size_t next = 0; next < N; ++next) {
            if (set & (1u << next)) { /* if next position is usable */
                const auto extended_len = path_length + dist[pos][next];
                if (extended_len < best) { /* extend the path */
                    summon_solve(dist, next, set & ~(1u << next), extended_len, best);
                }
            }
        }
    }

    float solve (const float distances[N][N]) {
        //        assert(!distances.empty());
        //        assert(distances.size() == distances[0].size());

        const auto mask = (1 << N) - 1;
        float best /*out*/ = INFINITY;
        summon_solve(distances, 0, mask & ~1, 0, best);
        return best;
    }
};

namespace BBPar {
    void
    summon_solve_par (
        const float dist[N][N],
        const uint_fast8_t pos, const uint_fast16_t set, const float path_length, /*in*/
        float& best                                                               /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
#pragma omp critical
            best = length < best ? length : best;
            return;
        }

        /* basically the same as a vector to save which ones do we need to task out */
        std::vector<uint_fast8_t> tasks;
        for (uint_fast8_t next = 0; next < N; ++next) {
            if (set & (1u << next)) { /* if next position is usable */
                const auto extended_len = path_length + dist[pos][next];
                if (extended_len < best) { /* extend the path */
                    tasks.push_back(next);
                }
            }
        }

        if (tasks.empty()) {
            return;
        }

        /* summon tasks for everyone... except the last one... */

        for (std::size_t it = 0; it < tasks.size() - 1; ++it) {
            const auto next = tasks[it];
            const auto extended_len = path_length + dist[pos][next];
#pragma omp task default(none) shared(dist, best) shared(next, set, extended_len)
            {
                summon_solve_par(dist, next, set & ~(1u << next), extended_len, best);
            }
        }

        /* because we want to keep it running without creating a new thread */
        const auto next = tasks.back();
        const auto extended_len = path_length + dist[pos][next];
        summon_solve_par(dist, next, set & ~(1u << next), extended_len, best);
#pragma omp taskwait
    }

    float solve (const float distances[N][N], int numthreads) {
        const auto mask = (1 << N) - 1;
        float best /*out*/ = INFINITY;

        /* create the caller task */
#pragma omp parallel default(none) shared(distances, best) num_threads(numthreads) // shared(mask)
        {
#pragma omp single
            summon_solve_par(distances, 0, mask & ~1, 0, best);
        }
        return best;
    }

    template <class T>
    std::pair<bool, T> pop_if_not_empty(std::stack<T>& s) {
        bool empty = true;
        T ret;
#pragma omp critical
        {
            if (!s.empty()) {
                empty = false;
                ret = s.top();
                s.pop();
            }
        }
        return std::make_pair(empty, ret);
    }

    float solve_non_rec (const float distances[N][N], int numthreads) {
        float best = INFINITY;
        std::stack<std::tuple<uint_fast8_t, uint_fast16_t, float>> stack;
        stack.emplace(0, (1 << N) - 2, 0);

        uint_fast8_t pos;
        uint_fast16_t set;
        float path_length;

#pragma omp parallel default(none) shared(stack, distances, best, std::cout) num_threads(numthreads) private(pos, set, path_length)
        while (true) {
            auto [empty, value] = pop_if_not_empty(stack);
            if (empty) {
                break;
            }
            std::tie(pos, set, path_length) = value;

            if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
                auto length = path_length + distances[pos][0];
#pragma omp critical
                best = length < best ? length : best;
                continue;
            }

            for (uint_fast8_t next = 0; next < N; ++next) {
                if (set & (1u << next)) { /* if next position is usable */
                    const auto extended_len = path_length + distances[pos][next];
                    if (extended_len < best) { /* extend the path */
#pragma omp critical
                        stack.emplace(next, set & ~(1 << next), extended_len);
                    }
                }
            }
        }
        return best;
    }
};

int main () {
    freopen("../in.txt", "r", stdin);
    freopen("/dev/null", "w", stderr);

    std::random_device rd;
    std::mt19937 g(rd());
    int name[N];
    std::iota(name, name + N, 0);
    std::shuffle(name, name + N, g);

    int n, m;
    float distances[N][N];
    std::cin >> n >> m;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            distances[i][j] = i == j ? 0 : INFINITY;
        }
    }

    for (int i = 0; i < m; ++i) {
        int u, v;
        float w;
        std::cin >> u >> v >> w;
        distances[name[u]][name[v]] = distances[name[v]][name[u]] = w; /* Read shuffled to account for best and worst cases */
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << distances[i][j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Recursive\n";
    for (int nt = 1; nt < 16; ++nt) {
        Timer timer;
        std::cerr << BBPar::solve_non_rec(distances, nt) << '\n';
        for (int i = 1; i < 1000; ++i) {
            std::cerr << BBPar::solve_non_rec(distances, nt) << '\n';
        }
    }

    std::cout << "Non recursive\n";
    for (int nt = 1; nt < 16; ++nt) {
        Timer timer;
        std::cerr << BBPar::solve_non_rec(distances, nt) << '\n';
        for (int i = 1; i < 1000; ++i) {
            std::cerr << BBPar::solve_non_rec(distances, nt) << '\n';
        }
    }
    return 0;
}
