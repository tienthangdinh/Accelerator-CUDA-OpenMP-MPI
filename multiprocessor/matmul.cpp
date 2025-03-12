#include <cmath>
#include <cstdlib>
#include <cstring>
#include <c++/10/bits/algorithmfwd.h>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <ostream>
#include <chrono>

double **allocateMatrix(int n) {
    double **matrix = new double *[n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new double[n];
    }
    return matrix;
}
double **generateMatrix(int n) {
  double **matrix = allocateMatrix(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i][j] = static_cast<double>(rand()) / RAND_MAX * 10.0;
    }
  }
  return matrix;
}
void freeMatrix(double **matrix, int n) {
  for (int i = 0; i < n; i++) {
    delete[] matrix[i];
  }
  delete[] matrix;
}
bool matricesAreEqual(double **A, double **B, int n, double tolerance = 1e-2) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (fabs(A[i][j] - B[i][j]) > tolerance) {
        return false;
      }
    }
  }
  return true;
}
void flattenMatrix(double **matrix, double *flatMatrix, int n) {
  for (int i = 0; i < n; i++) {
    memcpy(flatMatrix + i * n, matrix[i], n * sizeof(double));
  }
}
void unflattenMatrix(double *flatMatrix, double **matrix, int n) {
  for (int i = 0; i < n; i++) {
    memcpy(matrix[i], flatMatrix + i * n, n * sizeof(double));
  }
}
void matrixMultiply(double **A, double **B, double **C, int n) {
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
void partialMatrixMultiplyOnFlatMatrix(double *A, double *B, double *C, int n,
                                       int rows) {
  // Perform matrix multiplication with loop unrolling and blocking for
  // optimization
  const int blockSize = 40; // Block size fits A, B, and C into L1 cache

  for (int bi = 0; bi < rows; bi += blockSize) { // Block row of A
    for (int bk = 0; bk < n; bk += blockSize) { // Block column of A and row of B
      for (int bj = 0; bj < n; bj += blockSize) { // Block column of B
        // Compute the product for one block
        for (int i = bi; i < std::min(bi + blockSize, rows); i++) {
          for (int k = bk; k < std::min(bk + blockSize, n); k++) {
            double valA = A[i * n + k];
            // Loop unrolling for the innermost loop
            // hopefully that gets optimized to SIMD instructions by the
            // compiler
            int j;
            for (j = bj; j <= std::min(bj + blockSize, n) - 8; j += 8) {
              C[i * n + j] += valA * B[k * n + j];
              C[i * n + j + 1] += valA * B[k * n + j + 1];
              C[i * n + j + 2] += valA * B[k * n + j + 2];
              C[i * n + j + 3] += valA * B[k * n + j + 3];
              C[i * n + j + 4] += valA * B[k * n + j + 4];
              C[i * n + j + 5] += valA * B[k * n + j + 5];
              C[i * n + j + 6] += valA * B[k * n + j + 6];
              C[i * n + j + 7] += valA * B[k * n + j + 7];
            }
            // Handle remaining columns
            for (; j < std::min(bj + blockSize, n); j++) {
              C[i * n + j] += valA * B[k * n + j];
            }
          }
        }
      }
    }
  }
}


using namespace std;

int main(int argc, char*argv[]) {
    int n = 400;
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double **result = nullptr;
    double *flatA = nullptr, *flatB = new double[n * n], *flatResult = nullptr;

    int rowsPerRank = n / size, rowsPerRankRemainder = n % size;
    // Allocate memory for local computations
    double *localA = new double[rowsPerRank * n],
            *localResult = new double[rowsPerRank * n];
    fill(localA, localA + rowsPerRank * n, 0.0);
    fill(localResult, localResult + rowsPerRank * n, 0.0);

    if (rank == 0) {
        double **A = generateMatrix(n);
        double **B = generateMatrix(n);
        result = allocateMatrix(n);
        flatA = new double[n * n];
        flatResult = new double[n * n];
        flattenMatrix(A, flatA, n);
        flattenMatrix(B, flatB, n);

        // perform a basic matrix multiplication for reference only till a matrix
        // size of 4096
        if (n <= 4096) {
        matrixMultiply(A, B, result, n);
        }

        freeMatrix(A, n);
        freeMatrix(B, n);
    }
    // ATTENTION: RUNS ON ALL PROCESSES
    // Allocate memory for local computations
    auto start_time = chrono::high_resolution_clock::now(); // Start time
    MPI_Request request;
    MPI_Ibcast(flatB, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);
    MPI_Scatter(flatA, rowsPerRank * n, MPI_DOUBLE, localA, rowsPerRank * n,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Wait(&request, MPI_STATUS_IGNORE); // Wait for broadcast to complete
    partialMatrixMultiplyOnFlatMatrix(localA, flatB, localResult, n, rowsPerRank);
    MPI_Gather(localResult, rowsPerRank * n, MPI_DOUBLE, flatResult,
             rowsPerRank * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end_time = chrono::high_resolution_clock::now();

    delete[] localA;
    delete[] localResult;
    delete[] flatB;

    MPI_Finalize();

    if (rank = 0) {
        double **flatResult2D = allocateMatrix(n);
        unflattenMatrix(flatResult, flatResult2D, n);
        bool correct = matricesAreEqual(result, flatResult2D, n);
        cout << "Result is:\t\t\t" << (correct ? "correct" : "incorrect") << endl;
        freeMatrix(result, n);
        delete[] flatResult;
        chrono::duration<double> duration = end_time - start_time;
        cout << "Time the calculation took:\t" << duration.count() << endl;
    }
    return 0;
}