#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

const int THREADS_PER_BLOCK = 512;

__global__ void Check(const int n, const int m, const int startLevel,
         const int* exprVar, const int* exprNeg,
         int* set, int* flags, const int q) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
        return;
    }

    for (int i = 0; i < startLevel; ++i) {
        if (((uint32_t)e) & (((uint32_t)1) << ((uint32_t)i))) {
            set[e * n + i] = 1;
        } else {
            set[e * n + i] = 0;
        }
    }

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

__global__ void DFS(const int n, const int m, const int startLevel,
         const int* exprVar, const int* exprNeg,
         int* set, int* flags, int* isFound, const int q) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
        return;
    }

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
            atomicMax(isFound, flags[e]);
            if (*isFound == 1) {
                return;
            }
            if (flags[e] == 0) {
                continue;
            }
            ++k;
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
    int* cudaExprVar = nullptr;
    int* cudaExprNeg = nullptr;
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

    int* set = (int*)calloc(q * n, sizeof(*set));
    int* cudaSet = nullptr;
    for (int i = 0; i < q * n; ++i) {
        set[i] = -1;
    }
    int* flags = (int*)calloc(q, sizeof(*flags));
    for (int i = 0; i < q; ++i) {
        flags[i] = 1;
    }
    int* cudaFlags = nullptr;
    bool isSolution = false;

    cudaMalloc(&cudaExprVar, 3 * m * sizeof(*exprVar));
    cudaMalloc(&cudaExprNeg, 3 * m * sizeof(*exprNeg));
    cudaMalloc(&cudaSet, q * n * sizeof(*set));
    cudaMalloc(&cudaFlags, q * sizeof(*flags));
    cudaMemcpy(cudaExprVar, exprVar, 3 * m * sizeof(*exprVar), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaExprNeg, exprNeg, 3 * m * sizeof(*exprNeg), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaSet, set, q * n * sizeof(*set), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaFlags, flags, q * sizeof(*flags), cudaMemcpyHostToDevice);

    int qBlock = (q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    Check<<<qBlock, THREADS_PER_BLOCK>>>(n, m, startLevel,
                                         cudaExprVar, cudaExprNeg, cudaSet, cudaFlags, q);

    cudaMemcpy(set, cudaSet, q * n * sizeof(*set), cudaMemcpyDeviceToHost);
    cudaMemcpy(flags, cudaFlags, q * sizeof(*flags), cudaMemcpyDeviceToHost);

    for (int i = 0, j = q - 1;;) {
        while (i < q && flags[i] != 0) {
            ++i;
        }
        while (j >= 0 && flags[j] == 0) {
            --j;
        }
        if (i >= j) {
            q = i;
            break;
        }
        memcpy(set + i * n, set + j * n, n * sizeof(*set));
        std::swap(flags[i], flags[j]);
    }

    int* isFound = nullptr;
    cudaMalloc(&isFound, sizeof(*isFound));
    cudaMemset(isFound, 0, sizeof(*isFound));

    if (q > 0) {
        cudaMemcpy(cudaSet, set, q * n * sizeof(*set), cudaMemcpyHostToDevice);
        cudaMemcpy(cudaFlags, flags, q * sizeof(*flags), cudaMemcpyHostToDevice);

        qBlock = (q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        DFS<<<qBlock, THREADS_PER_BLOCK>>>(n, m, startLevel,
                                       cudaExprVar, cudaExprNeg, cudaSet, cudaFlags, isFound, q);
        cudaMemcpy(set, cudaSet, q * n * sizeof(*set), cudaMemcpyDeviceToHost);
        cudaMemcpy(flags, cudaFlags, q * sizeof(*flags), cudaMemcpyDeviceToHost);
    }

    for (int e = 0; e < q; ++e) {
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

    cudaFree(cudaExprVar);
    cudaFree(cudaExprNeg);
    cudaFree(cudaSet);
    cudaFree(cudaFlags);
    cudaFree(isFound);

    std::chrono::high_resolution_clock::time_point totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::duration<double>>(totalEnd - totalStart).count();
    std::cout << "Total time: " << totalTime << std::endl;

    return 0;
}
