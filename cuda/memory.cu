#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Number of threads

// Kernel that stores thread IDs and outputs in an array
__global__ void printThreadInfo(int *threadIds, int *outputs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        threadIds[idx] = idx;  // Store thread ID
        outputs[idx] = idx * 2;  // Store some output (e.g., idx * 2)
    }
}

int main() {
    int *h_threadIds = (int*)malloc(N * sizeof(int));  // Host array for thread IDs
    int *h_outputs = (int*)malloc(N * sizeof(int));   // Host array for outputs

    //GPU Malloc
    int *d_threadIds, *d_outputs;
    cudaMalloc(&d_threadIds, N * sizeof(int));
    cudaMalloc(&d_outputs, N * sizeof(int));

    // Launch kernel
    int blockSize = 4;  // 4 threads per block
    int gridSize = (N + blockSize - 1) / blockSize;  // Number of blocks
    printThreadInfo<<<gridSize, blockSize>>>(d_threadIds, d_outputs);

    // Copy results back to host
    cudaMemcpy(h_threadIds, d_threadIds, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputs, d_outputs, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results on the host
    for (int i = 0; i < N; i++) {
        std::cout << "Thread ID: " << h_threadIds[i] << ", Output: " << h_outputs[i] << std::endl;
    }

    // Free memory
    free(h_threadIds);
    free(h_outputs);
    cudaFree(d_threadIds);
    cudaFree(d_outputs);

    return 0;
}