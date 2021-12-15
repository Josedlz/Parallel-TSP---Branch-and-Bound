#include "main.hpp"
#include <random>

const int n = 10;

int main () {
    freopen("../in.txt", "r", stdin);
    freopen("/dev/null", "w", stderr);

//    int n, m;
    float distances[N][N] = {
        { 0, 5.1, 9.3, 11.8, 3.7, 12.5, 17.9, 6.5, 6.4, 10.2 },
        { 5.0, 0, 3.2, 7.0, 7.7, 17.5, 13.7, 2.9, 5.2, 5.8 },
        { 9.0, 3.8, 0, 4.0, 11.7, 21.5, 13.9, 5.8, 6.8, 6.2 },
        { 12.3, 7.9, 4.2, 0, 15.0, 24.8, 16.7, 9.4, 11.5, 9.5 },
        { 4.6, 7.8, 11.9, 14.3, 0, 12.1, 18.7, 8.1, 9.9, 12.9 },
        { 13.7, 17.1, 21.4, 23.8, 12.3, 0, 29.8, 17.9, 17.1, 22.3 },
        { 17.5, 14.0, 14.7, 17.7, 19.2, 28.9, 0, 12.6, 17.3, 9.3 },
        { 6.0, 3.5, 6.2, 9.2, 8.0, 17.7, 13.0, 0, 7.0, 4.7 },
        { 6.5, 5.0, 6.0, 9.1, 10.2, 16.7, 17.2, 7.4, 0, 9.5 },
        { 9.6, 6.3, 5.5, 8.8, 12.6, 21.9, 9.2, 4.6, 9.6, 0 },
    };
//    std::cin >> n >> m;
//
//    std::random_device rd;
//    std::mt19937 g(rd());
//    int name[N];
//    std::iota(name, name + n, 0);
//    std::shuffle(name, name + n, g);
//
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            distances[i][j] = i == j ? 0 : INFINITY;
//        }
//    }
//
//    for (int i = 0; i < m; ++i) {
//        int u, v;
//        float w;
//        std::cin >> u >> v >> w;
//        distances[name[u]][name[v]] = distances[name[v]][name[u]] = w; /* Read shuffled to account for best and worst cases */
//    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << distances[i][j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Recursive\n";
    for (int nt = 1; nt <= 16; nt *= 2) {
        std::cout << nt << ": ";
        Timer timer;
        for (int i = 0; i < 100; ++i) {
            std::cerr << BBPar::solve(distances, n, nt) << '\n';
        }
    }

    return 0;
}
