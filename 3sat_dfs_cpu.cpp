#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

const int BLOCK_SIZE = 32;

void Check(const int e, const int n, const int m, const int startLevel,
         const int* exprVar, const int* exprNeg,
         int* set, int* flags) {
    for (int i = 0; i < startLevel; ++i) {
        if (((uint32_t)e) & (((uint32_t)1) << ((uint32_t)i))) {
            set[e * n + i] = 1;
        } else {
            set[e * n + i] = 0;
        }
    }

    flags[e] = 1;
    for (int i = 0; i < m; ++i) {
        int disjunctRes = 0;
        for (int j = 0; j < 3; ++j) {
            int index = exprVar[i * 3 + j];
            if (index >= startLevel) {
                disjunctRes = -1;
            } else {
                int elem = set[e * n + index];
                elem ^= exprNeg[i * 3 + j];
                if (elem == 1) {
                    disjunctRes = 1;
                    break;
                }
            }
        }
        if (disjunctRes == 0) {
            flags[e] = 0;
            break;
        }
        if (disjunctRes == -1) {
            flags[e] = -1;
        }
    }
}

void DFS(const int e, const int n, const int m, const int startLevel,
         const int* exprVar, const int* exprNeg,
         int* set, int* flags) {
    for (int k = startLevel; k >= startLevel;) {
        if (k == n) {
            --k;
        } else if (set[e * n + k] != 0) {
            set[e * n + k] = (set[e * n + k] == -1 ? 1 : 0);
            flags[e] = 1;
            for (int i = 0; i < m; ++i) {
                int disjunctRes = 0;
                for (int j = 0; j < 3; ++j) {
                    int index = exprVar[i * 3 + j];
                    if (index > k) {
                        disjunctRes = -1;
                    } else {
                        int elem = set[e * n + index];
                        elem ^= exprNeg[i * 3 + j];
                        if (elem == 1) {
                            disjunctRes = 1;
                            break;
                        }
                    }
                }
                if (disjunctRes == 0) {
                    flags[e] = 0;
                    break;
                }
                if (disjunctRes == -1) {
                    flags[e] = -1;
                }
            }
            if (flags[e] == 0) {
                continue;
            } else if (flags[e] == 1) {
                return;
            } else {
                ++k;
            }
        } else {
            set[e * n + k] = -1;
            --k;
        }
    }
}

int main(int argc, char* argv[]) {
    std::chrono::high_resolution_clock::time_point totalStart = std::chrono::high_resolution_clock::now();

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file precalc_depth" << std::endl;
        return 0;
    }

    std::ifstream fin(argv[1]);
    std::ofstream fout(argv[2]);

    int n, m;
    fin >> n >> m;

    int* exprVar = (int*)malloc(3 * m * sizeof(*exprVar));
    int* exprNeg = (int*)malloc(3 * m * sizeof(*exprNeg));
    for (int i = 0; i < m; ++i) {
        fin >> exprVar[3 * i]
            >> exprNeg[3 * i]
            >> exprVar[3 * i + 1]
            >> exprNeg[3 * i + 1]
            >> exprVar[3 * i + 2]
            >> exprNeg[3 * i + 2];
        --exprVar[3 * i];
        --exprVar[3 * i + 1];
        --exprVar[3 * i + 2];
    }

    int startLevel = std::min(n, atoi(argv[3]));
    int q = (1 << startLevel);

    int* set = (int*)malloc(q * n * sizeof(*set));
    for (int i = 0; i < q * n; ++i) {
        set[i] = -1;
    }
    int* flags = (int*)malloc(q * sizeof(*flags));
    bool isSolution = false;
    for (int e = 0; e < q; ++e) {
        Check(e, n, m, startLevel, exprVar, exprNeg, set, flags);
    }
    for (int e = 0; e < q; ++e) {
        std::cout << "Subtree " << e << std::endl;
        if (flags[e] == -1) {
            DFS(e, n, m, startLevel, exprVar, exprNeg, set, flags);
        }
        if (flags[e] == 1) {
            for (int i = 0; i < n; ++i) {
                fout << "x_" << i + 1 << " = " <<
                    (set[e * n + i] == 1 ? 1 : 0) << std::endl;
            }
            isSolution = true;
            break;
        }
    }

    if (!isSolution) {
        fout << "No solution" << std::endl;
    }

    free(exprVar);
    free(exprNeg);
    free(set);
    free(flags);

    std::chrono::high_resolution_clock::time_point totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::duration<double>>(totalEnd - totalStart).count();
    std::cout << "Total time: " << totalTime << std::endl;

    return 0;
}
