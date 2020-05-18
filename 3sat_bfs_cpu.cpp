#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

const int BLOCK_SIZE = 32;

void BranchCPU(const ssize_t e, uint32_t* set, const int blockCount, const int k) {
    set[e * blockCount + k / BLOCK_SIZE] |= (((uint32_t)1) << ((uint32_t)(k % BLOCK_SIZE)));
}

void BoundCPU(const ssize_t e, const uint32_t* set, int* flags,
              const int blockCount, const int k, const int n, const int m, const int* exprVar, const int* exprNeg) {
    for (int i = 0; i < m; ++i) {
        int disjunctRes = 0;
        for (int j = 0; j < 3; ++j) {
            int index = exprVar[i * 3 + j];
            if (index > k) {
                disjunctRes = -1;
            } else {
                int elem = (set[e * blockCount + index / BLOCK_SIZE] & (((uint32_t)1) << ((uint32_t)(index % BLOCK_SIZE)))) ? 1 : 0;
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
    }
}

int main(int argc, char* argv[]) {
    std::chrono::high_resolution_clock::time_point totalStart = std::chrono::high_resolution_clock::now();
 
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file" << std::endl;
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

    ssize_t q = 1;

    const int blockCount = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t* set = (uint32_t*)calloc(q * blockCount, sizeof(*set));
    int* flags = (int*)malloc(q * sizeof(*flags));
    flags[0] = 1;

    for (int k = 0; k < n; ++k) {
        std::cout << "Step " << k + 1 << ", q = " << q << std::endl;
        set = (uint32_t*)realloc(set, 2 * q * blockCount * sizeof(*set));
        flags = (int*)realloc(flags, 2 * q * sizeof(*flags));
        memcpy(set + q * blockCount, set, q * blockCount * sizeof(*set));
        memcpy(flags + q, flags, q * sizeof(*flags));

        for (ssize_t e = 0; e < q; ++e) {
            BranchCPU(e, set, blockCount, k);
        }
        for (ssize_t e = 0; e < 2 * q; ++e) {
            BoundCPU(e, set, flags, blockCount, k, n, m, exprVar, exprNeg);
        }

        for (ssize_t i = 0, j = 2 * q - 1;;) {
            while (i < 2 * q && flags[i] != 0) {
                ++i;
            }
            while (j >= 0 && flags[j] == 0) {
                --j;
            }
            if (i >= j) {
                q = i;
                break;
            }
            memcpy(set + i * blockCount, set + j * blockCount, blockCount * sizeof(*set));
            std::swap(flags[i], flags[j]);
        }

        if (q == 0) {
            break;
        }
    }
    
    if (q == 0) {
        fout << "No solution" << std::endl;
    } else {
        for (int i = 0; i < n; ++i) {
            fout << "x_" << i + 1 << " = " <<
                ((set[i / BLOCK_SIZE] & (((uint32_t)1) << ((uint32_t)(i % BLOCK_SIZE)))) ? 1 : 0) << std::endl;
        }
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
