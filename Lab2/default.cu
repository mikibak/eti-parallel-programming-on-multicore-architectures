/*
CUDA - generation of array of N elements and preparation of histogram.
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

using idx_t = int;
using count_t = unsigned long long;    // large counters

#define DEBUG 0

__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

__host__
void generateRandomNumbers(int *arr, count_t N, int A, int B) {
    srand(time(NULL));
    for (count_t i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }
}

__global__
void histogramKernel(const int *data, count_t *hist, count_t N, int A) {
    idx_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((count_t)idx < N) {
        int value = data[idx];
        atomicAdd(&hist[value - A], 1LL);
    }
}

int main(int argc, char **argv) {

    count_t N;
    int A = 0;
    int B = 100;

    printf("Enter number of elements:\n");
    scanf("%lld", &N);

    int range = B - A + 1;

    // Allocate host memory
    int *hData = (int*)malloc(N * sizeof(int));
    count_t *hHist = (count_t*)calloc(range, sizeof(count_t));

    if (!hData || !hHist)
        errorexit("Host memory allocation failed");

    // Generate random numbers
    generateRandomNumbers(hData, N, A, B);

    if (DEBUG) {
        printf("Generated numbers:\n");
        for (count_t i = 0; i < N; i++) printf("%d ", hData[i]);
        printf("\n");
    }

    // Device pointers
    int *dData = NULL;
    count_t *dHist = NULL;

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate device memory
    if (cudaSuccess != cudaMalloc((void**)&dData, N * sizeof(int)))
        errorexit("Error allocating data array on GPU");

    if (cudaSuccess != cudaMalloc((void**)&dHist, range * sizeof(count_t)))
        errorexit("Error allocating histogram array on GPU");

    // Initialize histogram to zero
    if (cudaSuccess != cudaMemset(dHist, 0, range * sizeof(count_t)))
        errorexit("Error initializing histogram on GPU");

    // Copy data to device
    if (cudaSuccess != cudaMemcpy(dData, hData, N * sizeof(int), cudaMemcpyHostToDevice))
        errorexit("Error copying data to device");

    // Grid configuration
    idx_t threadsinblock = 1024;
    idx_t blocksingrid = 1 + (idx_t)((N - 1) / threadsinblock);

    // Run kernel
    histogramKernel<<<blocksingrid, threadsinblock>>>(dData, dHist, N, A);

    // Copy histogram back to host
    if (cudaSuccess != cudaMemcpy(hHist, dHist, range * sizeof(count_t), cudaMemcpyDeviceToHost))
        errorexit("Error copying histogram back to host");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print histogram
    printf("\nHistogram:\n");
    count_t sum = 0;
    for (int i = 0; i < range; i++) {
        printf("Value %d occurs %llu times\n", A + i, hHist[i]);
        sum += hHist[i];
    }

    printf("\nTotal numbers counted: %llu\n", sum);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Free memory
    free(hData);
    free(hHist);

    if (cudaSuccess != cudaFree(dData))
        errorexit("Error freeing dData");

    if (cudaSuccess != cudaFree(dHist))
        errorexit("Error freeing dHist");

    return 0;
}
