#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>


namespace BBSeq {
    using Matrix = std::vector<std::vector<float>>;

    void
    summon_solve (
        const Matrix& dist, uint8_t pos, uint8_t set, float path_length, /*in*/
        float& best /*out*/
    ) {
        if (set == 0) { /* nowhere else to go: return to 0 and call it a day */
            best = std::min(best, path_length + dist[pos][0]);
        }

        for (std::size_t next = 0; next < dist.size(); ++next) {
            if (set & (1u << next)) { /* if next position is usable */
                const auto extended_len = path_length + dist[pos][next];
                if (extended_len >= best) { /* no reason to keep on */
                    continue;
                }
                /* extend the path */
                const auto set_wno_next = set & ~(1u << next);
                summon_solve(dist, next, set_wno_next, extended_len, best);
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
    std::vector<std::vector<float>> distances = {
        { 0,  10, 15, 20 },
        { 10, 0,  35, 25 },
        { 15, 35, 0,  30 },
        { 20, 25, 30, 0 }
    };

    std::cout << BB::solve(distances);
    return 0;
}
