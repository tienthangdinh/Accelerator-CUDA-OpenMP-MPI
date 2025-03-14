#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Size of the array

// Kernel that increments each element by 1
__global__ void incrementKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

int main() {
    int *data;
    cudaMallocManaged(&data, N * sizeof(int));  // Unified memory allocation

    // Initialize data on the host
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }

    // Prefetch data to the GPU
    cudaMemPrefetchAsync(data, N * sizeof(int), 0);  // 0 is the GPU device ID

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    incrementKernel<<<gridSize, blockSize>>>(data);

    // Prefetch data back to the CPU
    cudaMemPrefetchAsync(data, N * sizeof(int), cudaCpuDeviceId);

    // Synchronize to ensure prefetch and kernel completion
    cudaDeviceSynchronize(); //after every asynchronous operation, make sure GPU done so that CPU can continue

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (data[i] != i + 1) {
            success = false;
            printf("Error at index %d: expected %d, got %d\n", i, i + 1, data[i]);
        }
    }

    if (success) {
        std::cout << "Kernel executed successfully!" << std::endl;
    }

    // Free memory
    cudaFree(data);

    return 0;
}