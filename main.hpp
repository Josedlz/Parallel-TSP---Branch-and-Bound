//
// Created by Esteban on 12/15/2021.
//

#ifndef PARALLEL_TSP___BRANCH_AND_BOUND_TSP_HPP
#define PARALLEL_TSP___BRANCH_AND_BOUND_TSP_HPP

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
using Set = std::bitset<32>;

Set create_mask (const int n) {
    Set mask;
    mask.set();
    mask >>= N - n;
    return mask;
}

Set with (Set s, const std::size_t n, bool value) {
    s.set(n, value);
    return s;
}

namespace BBSeq {
    void
    summon_solve (
        const float dist[N][N], const int n,
        const uint8_t pos, const Set set, const float path_length, /*in*/
        float& best /*out*/
        ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            auto length = path_length + dist[pos][0];
            best = length < best ? length : best;
            return;
        }
        for (std::size_t next = 0; next < n; ++next) {
            if (set[next]) { /* if next position is usable */
                const auto extended_len = path_length + dist[pos][next];
                if (extended_len < best) { /* extend the path */
                    summon_solve(dist, n, next, with(set, next, false), extended_len, best);
                }
            }
        }
    }

    float solve (const float distances[N][N], const int n) {
        const auto mask = with(create_mask(n), 0, false);
        float best /*out*/ = INFINITY;
        summon_solve(distances, n, 0, mask, 0, best);
        return best;
    }
};

namespace BBPar {
    void summon_solve_par (
        const float dist[N][N], const int n,
        const uint_fast8_t pos, const uint16_t set, const float path_length, /*in*/
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
        for (uint_fast8_t next = 0; next < n; ++next) {
            if (set & (1 << next)) {
                if (path_length + dist[pos][next] < best) {
#pragma omp task default(none) shared(dist, best, next) shared(tasks, set, path_length, pos, n)
                    summon_solve_par(dist, n, next, set & ~(1 << next), path_length + dist[pos][next], best);
                }
            }
        }
    }

    float solve (const float distances[N][N], const int n, int numthreads) {
        const auto mask = (1 << n) - 2;
        float best /*out*/ = INFINITY;

#pragma omp parallel default(none) shared(distances, best) num_threads(numthreads) shared(mask, n)
#pragma omp single
        { summon_solve_par(distances, n, 0, mask, 0, best); }
#pragma omp taskwait

        return best;
    }

    template<class T>
        std::pair<bool, T> pop_if_not_empty (std::stack<T>& s) {
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

    float solve_non_rec (const float distances[N][N], int n, int numthreads) {
        float best = INFINITY;
        std::stack<std::tuple<uint_fast8_t, Set, float>> stack;
        stack.emplace(0, with(create_mask(n), 0, false), 0);

#pragma omp parallel default(none) shared(stack, distances, best) num_threads(numthreads)
        while (true) {
            auto[empty, value] = pop_if_not_empty(stack);
            if (empty) {
                break;
            }
            auto[pos, set, path_length] = value;
            if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
                auto length = path_length + distances[pos][0];
#pragma omp critical
                best = length < best ? length : best;
                continue;
            }

            for (uint_fast8_t next = 0; next < N; ++next) {
                if (set[next]) { /* if next position is usable */
                    const auto extended_len = path_length + distances[pos][next];
                    if (extended_len < best) { /* extend the path */
#pragma omp critical
                        stack.emplace(next, with(set, next, false), extended_len);
                    }
                }
            }
        }
        return best;
    }
};

#endif //PARALLEL_TSP___BRANCH_AND_BOUND_TSP_HPP
