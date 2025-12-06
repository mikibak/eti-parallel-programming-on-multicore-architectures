/*
CUDA - generation of array of N elements and calculates histogram - with streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

using idx_t = int;
using count_t = unsigned long long;  // histogram counter type

#define DEBUG 0
__host__
void errorexit(const char *s) {
    printf("\n%s\n",s); 
    exit(EXIT_FAILURE);   
}

__host__
void generate(int *matrix, int matrixSize, int A, int B) {
    srand(time(NULL));
    for (int i = 0; i < matrixSize; i++) {
        matrix[i] = A + rand() % (B - A + 1);
    }
}

__global__
void histogramKernel(const int *matrix, count_t *hist, idx_t startIdx, idx_t endIdx, int A) {
    idx_t my_index = blockIdx.x * blockDim.x + threadIdx.x + startIdx;

    if (my_index < endIdx) {
        int value = matrix[my_index];
        atomicAdd(&hist[value - A], 1LL);
    }
}

int main(int argc, char **argv) {

    /// define number of streams
    int numberOfStreams = 4;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // histogram range
    int A = 0;
    int B = 100;
    int range = B - A + 1;

    int matrixSize;
    printf("Enter number of elements:\n");
    scanf("%d", &matrixSize);

    // allocate memory on host
    int *hMatrix = (int*)malloc(matrixSize * sizeof(int));
    count_t *hHist = (count_t*)calloc(range, sizeof(count_t));

    if (!hMatrix || !hHist)
        errorexit("Host memory allocation failed");

    // compute chunk size
    int chunkSize = matrixSize / numberOfStreams;
    int remainder = matrixSize % numberOfStreams;

    printf("Stream chunk is %d\n", chunkSize);

    int threadsinblock = 1024;

    // create streams
    cudaStream_t streams[numberOfStreams];
    for (int i = 0; i < numberOfStreams; i++) {
        if (cudaSuccess != cudaStreamCreate(&streams[i]))
            errorexit("Error creating stream");
    }

    int *dMatrix = NULL;
    count_t *dHist = NULL;

    // pinned host memory
    if (cudaSuccess != cudaMallocHost((void**)&hMatrix, matrixSize * sizeof(int)))
        errorexit("Error allocating pinned memory");

    // generate input data
    generate(hMatrix, matrixSize, A, B);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // allocate device memory
    if (cudaSuccess != cudaMalloc((void**)&dMatrix, matrixSize * sizeof(int)))
        errorexit("Error allocating dMatrix");

    if (cudaSuccess != cudaMalloc((void**)&dHist, range * sizeof(count_t)))
        errorexit("Error allocating dHist");

    if (cudaSuccess != cudaMemset(dHist, 0, range * sizeof(count_t)))
        errorexit("Error memset histogram");

    int offset = 0;
    // per-stream kernel launches
    for (int i = 0; i < numberOfStreams; i++) {

        int streamChunk = chunkSize + (i < remainder ? 1 : 0);
        int blocksingrid = (streamChunk + threadsinblock - 1) / threadsinblock;

        int startIdx = offset;
        int endIdx = startIdx + streamChunk;

        cudaMemcpyAsync(&dMatrix[offset], &hMatrix[offset],
                        streamChunk * sizeof(int), cudaMemcpyHostToDevice, streams[i]);

        histogramKernel<<<blocksingrid, threadsinblock, 0, streams[i]>>>(
            dMatrix, dHist, startIdx, endIdx, A
        );

        offset += streamChunk;
    }

    cudaDeviceSynchronize();

    // copy histogram back
    if (cudaSuccess != cudaMemcpy(hHist, dHist, range * sizeof(count_t), cudaMemcpyDeviceToHost))
        errorexit("Error copying histogram back");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // print histogram
    printf("\nHistogram:\n");
    count_t total = 0;
    for (int i = 0; i < range; i++) {
        printf("Value %d occurs %llu times\n", A + i, hHist[i]);
        total += hHist[i];
    }
    printf("\nTotal numbers counted: %llu\n", total);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // cleanup
    for (int i = 0; i < numberOfStreams; i++) {
        if (cudaSuccess != cudaStreamDestroy(streams[i]))
            errorexit("Error destroying stream");
    }

    free(hHist);

    if (cudaSuccess != cudaFreeHost(hMatrix)) errorexit("Error freeing pinned host memory");
    if (cudaSuccess != cudaFree(dHist)) errorexit("Error freeing dHist");
    if (cudaSuccess != cudaFree(dMatrix)) errorexit("Error freeing dMatrix");

    return 0;
}
