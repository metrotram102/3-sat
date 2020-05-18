#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

const int BLOCK_SIZE = 32;
const int THREADS_PER_BLOCK = 512;

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

__global__ void BranchGPU(uint32_t* set, const int blockCount, const int k, const ssize_t q) {
    ssize_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
    	return;
    }
    set[e * blockCount + k / BLOCK_SIZE] |= (((uint32_t)1) << ((uint32_t)(k % BLOCK_SIZE)));
}

__global__ void BoundGPU(const uint32_t* set, int* flags,
              const int blockCount, const int k, const int n, const int m, const int* exprVar, const int* exprNeg, const ssize_t q) {
    ssize_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
    	return;
    }

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

    ssize_t q = 1;

    const int blockCount = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t* set = (uint32_t*)calloc(q * blockCount, sizeof(*set));
    int* flags = (int*)malloc(q * sizeof(*flags));
    flags[0] = 1;

    for (int k = 0; k < n; ++k) {
        std::cout << "Step " << k + 1 << ", q = " << q << std::endl;
        if (q > 1000) {
            if (cudaExprVar == nullptr) {
                cudaMalloc(&cudaExprVar, 3 * m * sizeof(*exprVar));
                cudaMalloc(&cudaExprNeg, 3 * m * sizeof(*exprNeg));
                cudaMemcpy(cudaExprVar, exprVar, 3 * m * sizeof(*exprVar), cudaMemcpyHostToDevice);
                cudaMemcpy(cudaExprNeg, exprNeg, 3 * m * sizeof(*exprNeg), cudaMemcpyHostToDevice);
            }

            uint32_t* cudaSet;
            int* cudaFlags;
            cudaMalloc(&cudaSet, 2 * q * blockCount * sizeof(*set));
            cudaMalloc(&cudaFlags, 2 * q * sizeof(*flags));
            cudaMemcpy(cudaSet, set, q * blockCount * sizeof(*set), cudaMemcpyHostToDevice);
            cudaMemcpy(cudaSet + q * blockCount, set, q * blockCount * sizeof(*set), cudaMemcpyHostToDevice);
            cudaMemcpy(cudaFlags, flags, q * sizeof(*flags), cudaMemcpyHostToDevice);
            cudaMemcpy(cudaFlags + q, flags, q * sizeof(*flags), cudaMemcpyHostToDevice);

            ssize_t qBlock = (q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            BranchGPU<<<qBlock, THREADS_PER_BLOCK>>>(cudaSet, blockCount, k, q);
            cudaDeviceSynchronize();

            q *= 2;

            qBlock = (q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; 
            BoundGPU<<<qBlock, THREADS_PER_BLOCK>>>(cudaSet, cudaFlags, blockCount, k, n, m, cudaExprVar, cudaExprNeg, q);
            cudaDeviceSynchronize();

            set = (uint32_t*)realloc(set, q * blockCount * sizeof(*set));
            flags = (int*)realloc(flags, q * sizeof(*flags));
            cudaMemcpy(set, cudaSet, q * blockCount * sizeof(*set), cudaMemcpyDeviceToHost);
            cudaMemcpy(flags, cudaFlags, q * sizeof(*flags), cudaMemcpyDeviceToHost);
            cudaFree(cudaSet);
            cudaFree(cudaFlags);

        } else {

            set = (uint32_t*)realloc(set, 2 * q * blockCount * sizeof(*set));
            flags = (int*)realloc(flags, 2 * q * sizeof(*flags));
            memcpy(set + q * blockCount, set, q * blockCount * sizeof(*set));
            memcpy(flags + q, flags, q * sizeof(*flags));

            for (ssize_t e = 0; e < q; ++e) {
                BranchCPU(e, set, blockCount, k);
            }
            q *= 2;
            for (ssize_t e = 0; e < q; ++e) {
                BoundCPU(e, set, flags, blockCount, k, n, m, exprVar, exprNeg);
            }
        }

        for (ssize_t i = 0, j = q - 1;;) {
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
            memcpy(set + i * blockCount, set + j * blockCount, blockCount * sizeof(*set));
            std::swap(flags[i], flags[j]);
        }

        if (q == 0) {
            break;
        }
    }

    if (cudaExprVar != nullptr) {
        cudaFree(cudaExprVar);
        cudaFree(cudaExprNeg);
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
