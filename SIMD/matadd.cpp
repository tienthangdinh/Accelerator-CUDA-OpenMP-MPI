#include <iostream>
#include <chrono>
#include <immintrin.h>  // For SSE/AVX
#include <cstdlib>      // For rand()
#include <iomanip>      // For std::setw()

// Matrix size (large for better benchmarking)
constexpr int N = 2048;  // 1024x1024 matrix
constexpr int M = 2048;

// Function to initialize matrices with random values
void initialize_matrices(float A[N][M], float B[N][M]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = static_cast<float>(rand() % 100);
            B[i][j] = static_cast<float>(rand() % 100);
        }
    }
}

// **1. Matrix Addition WITHOUT SIMD (Regular Loops)**
void matrix_add_no_simd(float A[N][M], float B[N][M], float C[N][M]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// **2. Matrix Addition WITH SSE (128-bit SIMD)**
void matrix_add_sse(float A[N][M], float B[N][M], float C[N][M]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; j += 4) {  // Process 4 floats at a time
            __m128 a = _mm_loadu_ps(&A[i][j]);  // Load 4 floats from A
            __m128 b = _mm_loadu_ps(&B[i][j]);  // Load 4 floats from B
            __m128 c = _mm_add_ps(a, b);        // SIMD addition
            _mm_storeu_ps(&C[i][j], c);         // Store result in C
        }
    }
}

// **3. Matrix Addition WITH AVX (256-bit SIMD)**
void matrix_add_avx(float A[N][M], float B[N][M], float C[N][M]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; j += 8) {  // Process 8 floats at a time
            __m256 a = _mm256_loadu_ps(&A[i][j]);  // Load 8 floats from A
            __m256 b = _mm256_loadu_ps(&B[i][j]);  // Load 8 floats from B
            __m256 c = _mm256_add_ps(a, b);        // SIMD addition
            _mm256_storeu_ps(&C[i][j], c);         // Store result in C
        }
    }
}

// **Function to Measure Execution Time**
void benchmark(void (*func)(float[N][M], float[N][M], float[N][M]), float A[N][M], float B[N][M], float C[N][M], const std::string &label) {
    auto start = std::chrono::high_resolution_clock::now();
    func(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << std::setw(20) << label << ": " << duration.count() << " seconds" << std::endl;
}

// **Main Function**
int main() {
    static float A[N][M], B[N][M], C[N][M];

    std::cout << "Initializing matrices..." << std::endl;
    initialize_matrices(A, B);

    std::cout << "Running benchmarks...\n";

    benchmark(matrix_add_no_simd, A, B, C, "No SIMD");
    benchmark(matrix_add_sse, A, B, C, "SSE (128-bit)");
    benchmark(matrix_add_avx, A, B, C, "AVX (256-bit)");

    return 0;
}
