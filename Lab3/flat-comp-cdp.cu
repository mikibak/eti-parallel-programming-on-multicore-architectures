#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000        // Total array size
#define BLOCK_SIZE 128
#define CHUNK_SIZE 30000  // Number of elements per child kernel

// Child kernel: counts odd/even numbers in a chunk
__global__ void childKernel(int *data, int *odd, int *even, int startIdx, int endIdx) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    if (idx < endIdx) {
        if (data[idx] % 2)
            atomicAdd(odd, 1);
        else
            atomicAdd(even, 1);
    }
}

// Parent kernel: launches child kernels
__global__ void parentKernel(int *data, int *odd, int *even, int size) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int numChunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    if (threadId < numChunks) {
        int start = threadId * CHUNK_SIZE;
        int end = min(start + CHUNK_SIZE, size);

        int threads = 128;
        int blocks = (end - start + threads - 1) / threads;

        // Launch child kernel asynchronously
        childKernel<<<blocks, threads>>>(data, odd, even, start, end);
    }
}

int main() {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    int *h_data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_data[i] = rand() % 1000;

    int *d_data, *d_even, *d_odd;
    int h_even = 0, h_odd = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_even, sizeof(int));
    cudaMalloc(&d_odd, sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_even, 0, sizeof(int));
    cudaMemset(d_odd, 0, sizeof(int));

    int parentThreads = 32;
    int parentBlocks = ( (N + CHUNK_SIZE - 1) / CHUNK_SIZE + parentThreads -1)/ parentThreads;

    // Launch parent kernel
    parentKernel<<<parentBlocks, parentThreads>>>(d_data, d_odd, d_even, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_even, d_even, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_odd, d_odd, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Odd numbers: %d\n", h_odd);
    printf("Even numbers: %d\n", h_even);
    printf("Total: %d\n", h_odd + h_even);

    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    cudaFree(d_data);
    cudaFree(d_even);
    cudaFree(d_odd);
    free(h_data);

    return 0;
}
