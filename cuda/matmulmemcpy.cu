#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REPETITIONS 1
/*
 * Each SM has 4 processing blocks, and each processing block contains 8 FP64
 * cores capable of performing FMA (Fused Multiply-Add)
 */
#define THREADS_PER_BLOCK 8 * 4

__global__ void matrixMultiplicationKernel(const double *A, const double *B,
                                           double *C, int n) {
  // Calculate thread IDs
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

void matrixMultiplicationCUDA(const double *A, const double *B, double *C,
                              int n) {
  // Allocate memory on GPU
  double *d_A, *d_B, *d_C;
  size_t size = n * n * sizeof(double);

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy data from host (CPU) to device (GPU)
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  // Define grid and block configuration
  dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  dim3 numBlocks(n / threadsPerBlock.x, n / threadsPerBlock.y);

  // Execute kernel
  matrixMultiplicationKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

  // Synchronize to ensure kernel completion
  cudaDeviceSynchronize();

  // Copy results back from device to host
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int N = atoi(argv[1]);
  if (N <= 0) {
    fprintf(stderr, "Matrix size must be a positive integer.\n");
    return EXIT_FAILURE;
  }

  // Allocate memory for matrices on the host
  size_t size = N * N * sizeof(double);
  double *A = (double *)malloc(size);
  double *B = (double *)malloc(size);
  double *C = (double *)malloc(size);

  // data initializing
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = (double)(rand()) / RAND_MAX * 10;
      B[i * N + j] = (double)(rand()) / RAND_MAX * 10;
      C[i * N + j] = 0.0;
    }
  }

  // Time measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < REPETITIONS; i++) {
    matrixMultiplicationCUDA(A, B, C, N);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);


  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double elapsed_time = ((double)(milliseconds));

  printf("Time taken: %f milliseconds\n", elapsed_time);

  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;
}