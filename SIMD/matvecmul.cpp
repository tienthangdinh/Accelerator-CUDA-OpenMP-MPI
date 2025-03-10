#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>  // OpenMP header

constexpr int N = 2048; // Rows of matrix
constexpr int M = 2048; // Columns of matrix

// Function to initialize matrix and vector with random values
void initialize(std::vector<std::vector<float>>& matrix, std::vector<float>& vec) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            matrix[i][j] = static_cast<float>(rand() % 100);

    for (int j = 0; j < M; ++j)
        vec[j] = static_cast<float>(rand() % 100);
}

// **1. Matrix-Vector Multiplication Without OpenMP**
void mat_vec_mult_no_omp(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec, std::vector<float>& result) {
    for (int i = 0; i < N; ++i) {
        result[i] = 0;
        for (int j = 0; j < M; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

// **2. Matrix-Vector Multiplication With OpenMP SIMD**
void mat_vec_mult_omp_simd_parallel(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec, std::vector<float>& result) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float tmp = 0.0f;
        #pragma omp simd reduction(+:tmp)
        for (int j = 0; j < M; ++j) {
            tmp += matrix[i][j] * vec[j];
        }
        result[i] = tmp;
    }
}

void mat_vec_mult_omp_simd_sequential(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec, std::vector<float>& result) {
    for (int i = 0; i < N; ++i) {
        float tmp = 0.0f;
        #pragma omp simd reduction(+:tmp)
        for (int j = 0; j < M; ++j) {
            tmp += matrix[i][j] * vec[j];
        }
        result[i] = tmp;
    }
}

// **Function to Measure Execution Time**
void benchmark(void (*func)(const std::vector<std::vector<float>>&, const std::vector<float>&, std::vector<float>&), 
               const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec, std::vector<float>& result, 
               const std::string &label) {
    auto start = std::chrono::high_resolution_clock::now();
    func(matrix, vec, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << label << ": " << duration.count() << " seconds\n";
}

// **Main Function**
int main() {
    std::vector<std::vector<float>> matrix(N, std::vector<float>(M));
    std::vector<float> vec(M), result(N);

    std::cout << "Initializing matrix and vector...\n";
    initialize(matrix, vec);

    std::cout << "Running benchmarks...\n";
    benchmark(mat_vec_mult_no_omp, matrix, vec, result, "No OpenMP");
    benchmark(mat_vec_mult_omp_simd_sequential, matrix, vec, result, "With OpenMP SIMD Sequential");
    benchmark(mat_vec_mult_omp_simd_parallel, matrix, vec, result, "With OpenMP SIMD Parallel");

    return 0;
}
