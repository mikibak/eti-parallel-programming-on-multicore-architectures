/*
CUDA - generation of array of N elements and preparation of histogram.
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define DEBUG 0
__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

__host__
void generateRandomNumbers(int *arr, int N, int A, int B) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }
}

__global__
void histogramKernel(int *data, int *hist, int N, int A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int value = data[idx];
        atomicAdd(&hist[value - A], 1);
    }
}

int main(int argc,char **argv) {

    int N;
    int A = 0;
    int B = 100;

    printf("Enter number of elements:\n");
    scanf("%d", &N);

    int range = B - A + 1;

    // Allocate host memory
    int *hData = (int*)malloc(N * sizeof(int));
    int *hHist = (int*)calloc(range, sizeof(int));

    if (!hData || !hHist)
        errorexit("Host memory allocation failed");

    // Generate random numbers
    generateRandomNumbers(hData, N, A, B);

    if (DEBUG) {
        printf("Generated numbers:\n");
        for (int i = 0; i < N; i++) printf("%d ", hData[i]);
        printf("\n");
    }

    // Device pointers
    int *dData = NULL;
    int *dHist = NULL;

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate device memory
    if (cudaSuccess != cudaMalloc((void**)&dData, N * sizeof(int)))
        errorexit("Error allocating data array on GPU");

    if (cudaSuccess != cudaMalloc((void**)&dHist, range * sizeof(int)))
        errorexit("Error allocating histogram array on GPU");

    // Initialize histogram to zero
    if (cudaSuccess != cudaMemset(dHist, 0, range * sizeof(int)))
        errorexit("Error initializing histogram on GPU");

    // Copy data to device
    if (cudaSuccess != cudaMemcpy(dData, hData, N * sizeof(int), cudaMemcpyHostToDevice))
        errorexit("Error copying data to device");

    int threadsinblock = 1024;
    int blocksingrid = 1 + (N - 1) / threadsinblock;

    // Run kernel
    histogramKernel<<<blocksingrid, threadsinblock>>>(dData, dHist, N, A);

    // Copy histogram back to host
    if (cudaSuccess != cudaMemcpy(hHist, dHist, range * sizeof(int), cudaMemcpyDeviceToHost))
        errorexit("Error copying histogram back to host");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print histogram
    printf("\nHistogram:\n");
    int sum = 0;
    for (int i = 0; i < range; i++) {
        printf("Value %d occurs %d times\n", A + i, hHist[i]);
        sum += hHist[i];
    }

    printf("\nTotal numbers counted: %d\n", sum);
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
