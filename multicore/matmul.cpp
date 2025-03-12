#include<iostream>
#include<cmath>
#include<chrono>
#include<ctime>
#include<functional>

float **allocateMatrix(int n) {
    float **matrix = new float *[n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new float[n];
    }
    return matrix;
}

float **generateMatrix(int n) {
    float **matrix = allocateMatrix(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = static_cast<float>(rand());
        }
    }
    return matrix;
}

void freeMatrix(float **matrix, int n) {
  for (int i = 0; i < n; i++) {
    delete[] matrix[i];
  }
  delete[] matrix;
}

void printMatrix(float **matrix, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

bool matricesAreEqual(float **A, float **B, int n, float tolerance = 1e-1) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (fabs(A[i][j] - B[i][j]) > tolerance) {
        return false; // Elements are not equal within tolerance
      }
    }
  }
  return true;
}

std::chrono::duration<double> multiplyMatrices(float **A, float **B, int n, int loop_count,
                 const std::function<void(float **, float **, float **, int)>
                     &loopImplementation) {
  float **result = allocateMatrix(n);

  auto start = std::chrono::high_resolution_clock::now();
  while (loop_count--) {
    loopImplementation(A, B, result, n);
  }
  auto end = std::chrono::high_resolution_clock::now();

  freeMatrix(result, n);

  return end - start;
}

void baseLoop(float **A, float **B, float **result, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //start accumilating in features length k from here
            for (int k = 0; k < n; k++) {
                result[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

void changeLoopOrder(float **A, float **B, float **result, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void loopUnrolling(float **A, float **B, float **result, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = 0;  // Ensure initialization
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j <= n - 10; j += 10) {  // Unroll by factor of 10
                float a_ik = A[i][k]; // Avoid redundant memory access
                result[i][j] += a_ik * B[k][j];
                result[i][j+1] += a_ik * B[k][j+1];
                result[i][j+2] += a_ik * B[k][j+2];
                result[i][j+3] += a_ik * B[k][j+3];
                result[i][j+4] += a_ik * B[k][j+4];
                result[i][j+5] += a_ik * B[k][j+5];
                result[i][j+6] += a_ik * B[k][j+6];
                result[i][j+7] += a_ik * B[k][j+7];
                result[i][j+8] += a_ik * B[k][j+8];
                result[i][j+9] += a_ik * B[k][j+9];
            }
            // Handle remaining values if n is not a multiple of 10
            for (int j = (n / 10) * 10; j < n; j++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


#define BLOCK_SIZE 10

void tiling(float **A, float **B, float **result, int n) {
    // Initialize the result matrix (optional if already zeroed)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                // Multiply the tiles
                for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ii++) {
                    for (int kk = k; kk < k + BLOCK_SIZE && kk < n; kk++) {
                        float temp = A[ii][kk];  // Reduce redundant memory access
                        for (int jj = j; jj < j + BLOCK_SIZE && jj < n; jj++) {
                            result[ii][jj] += temp * B[kk][jj];  // More cache-friendly
                        }
                    }
                }
            }
        }
    }
}


int main() {
    int n = 300;

    float **A = generateMatrix(n);
    float **B = generateMatrix(n);
    float **referenceResult = allocateMatrix(n);
    float **testResultChangeLoopOrder = allocateMatrix(n);
    float **testResultLoopUnrolling = allocateMatrix(n);
    float **testResultTiling = allocateMatrix(n);

        // Validate each implementation
    baseLoop(A, B, referenceResult, n);
    changeLoopOrder(A, B, testResultChangeLoopOrder, n);
    if (!matricesAreEqual(referenceResult, testResultChangeLoopOrder, n)) {
        std::cerr << "Changed loop order implementation failed correctness check!"
            << std::endl;
        return 1;
    }

    loopUnrolling(A, B, testResultLoopUnrolling, n);
    if (!matricesAreEqual(referenceResult, testResultLoopUnrolling, n)) {
        std::cerr << "Loop unrolling implementation failed correctness check!" << std::endl;
        return 1;
    }

    tiling(A, B, testResultTiling, n);
    if (!matricesAreEqual(referenceResult, testResultTiling, n)) {
        std::cerr << "Tiling implementation failed correctness check!" << std::endl;
        return 1;
    }

    

    

    // Free allocated memory
    freeMatrix(referenceResult, n);
    freeMatrix(testResultChangeLoopOrder, n);
    freeMatrix(testResultLoopUnrolling, n);
    freeMatrix(testResultTiling, n);
    std::cout << "All implementations passed correctness checks!" << std::endl;


    int iterations = 1;
    auto elapsedBase = multiplyMatrices(A, B, n, iterations, baseLoop);
    auto elapsedChangeLoopOrder =
        multiplyMatrices(A, B, n, iterations, changeLoopOrder);
    auto elapsedLoopUnrolling =
        multiplyMatrices(A, B, n, iterations, loopUnrolling);
    auto elapsedTiling = multiplyMatrices(A, B, n, iterations, tiling);

    std::cout << "Elapsed time (base): " << elapsedBase.count() << " s" << std::endl;
    std::cout << "Elapsed time (change loop order): " << elapsedChangeLoopOrder.count()
        << " s" << std::endl;
    std::cout << "Elapsed time (loop unrolling): " << elapsedLoopUnrolling.count()
        << " s" << std::endl;
    std::cout << "Elapsed time (tiling): " << elapsedTiling.count() << " s" << std::endl;
}