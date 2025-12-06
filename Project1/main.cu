#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>

#define BLOCK_SIZE 1024          // Threads per block
#define ELEMENTS_PER_BLOCK 2048  // Elements sorted per block

using idx_t = int;

// ================================================================
// CPU helper to check correctness
// ================================================================
bool checkSorted(int* arr, idx_t n)
{
    for (idx_t i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1])
            return false;
    return true;
}

// ================================================================
// Device merge function (ping-pong)
__device__ void mergeShared(int* src, int* dst, idx_t start, idx_t mid, idx_t end)
{
    idx_t i = start, j = mid + 1, k = start;
    while (i <= mid && j <= end) dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i <= mid) dst[k++] = src[i++];
    while (j <= end) dst[k++] = src[j++];
}

// ================================================================
// Kernel: in-block merge using shared memory ping-pong
// ================================================================
__global__ void mergeSortInBlock(int* array, idx_t n)
{
    __shared__ int ping[ELEMENTS_PER_BLOCK];
    __shared__ int pong[ELEMENTS_PER_BLOCK];

    idx_t blockStart = blockIdx.x * ELEMENTS_PER_BLOCK;
    if (blockStart >= n) return;

    idx_t tid = threadIdx.x;
    idx_t blockSize = min(ELEMENTS_PER_BLOCK, n - blockStart);

    // Load to shared memory
    for (idx_t i = tid; i < blockSize; i += blockDim.x)
        ping[i] = array[blockStart + i];
    __syncthreads();

    int* src = ping;
    int* dst = pong;

    // Iterative merge
    for (idx_t size = 2; size <= blockSize; size <<= 1)
    {
        idx_t numIntervals = (blockSize + size - 1) / size;
        idx_t intervalId = tid;
        if (intervalId < numIntervals)
        {
            idx_t start = intervalId * size;
            idx_t mid   = min(start + size / 2 - 1, blockSize - 1);
            idx_t end   = min(start + size - 1, blockSize - 1);
            mergeShared(src, dst, start, mid, end);
        }
        __syncthreads();
        int* tmp = src; src = dst; dst = tmp;
    }

    // Copy back
    for (idx_t i = tid; i < blockSize; i += blockDim.x)
        array[blockStart + i] = src[i];
}

// ================================================================
// Kernel: merge multiple block intervals in global memory
// ================================================================
__global__ void mergeBlockPairs(int* array, int* temp, idx_t n, idx_t intervalBlocks)
{
    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t blocksPerMerge = intervalBlocks * 2;
    idx_t numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    idx_t startMerge = tid * blocksPerMerge;
    if (startMerge >= numBlocks) return;

    idx_t lBlock = startMerge;
    idx_t rBlock = min(startMerge + intervalBlocks, numBlocks);

    idx_t l = lBlock * ELEMENTS_PER_BLOCK;
    idx_t mid = min(rBlock * ELEMENTS_PER_BLOCK, n) - 1;
    idx_t r = min((startMerge + blocksPerMerge) * ELEMENTS_PER_BLOCK, n) - 1;

    if (l > mid || mid >= r) return;

    idx_t i = l, j = mid + 1, k = l;
    while (i <= mid && j <= r) temp[k++] = (array[i] <= array[j]) ? array[i++] : array[j++];
    while (i <= mid) temp[k++] = array[i++];
    while (j <= r) temp[k++] = array[j++];
    for (idx_t t = l; t <= r; t++) array[t] = temp[t];
}

// ================================================================
// Host function: GPU merge sort
// ================================================================
void mergeSort_gpu(int* d_array, idx_t n)
{
    idx_t numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    size_t sharedMemSize = ELEMENTS_PER_BLOCK * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Phase 1: in-block merge
    mergeSortInBlock<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_array, n);
    cudaDeviceSynchronize();

    // Phase 2: merge blocks iteratively
    int* d_temp;
    cudaMalloc(&d_temp, n * sizeof(int));

    idx_t intervalBlocks = 1; // start merging 2 blocks
    while (intervalBlocks < numBlocks)
    {
        idx_t merges = (numBlocks + intervalBlocks * 2 - 1) / (intervalBlocks * 2); // ceil division
        idx_t threadsPerBlock = 1024;
        idx_t blocksPerGrid = (merges + threadsPerBlock - 1) / threadsPerBlock;

        mergeBlockPairs<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_temp, n, intervalBlocks);
        cudaDeviceSynchronize();

        intervalBlocks *= 2;
    }

    cudaFree(d_temp);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total GPU merge sort time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    idx_t N;
    printf("Enter number of elements: ");
    if (scanf("%d", &N) != 1 || N <= 0)
    {
        printf("Invalid input.\n");
        return 1;
    }

    int* h = (int*)malloc(N * sizeof(int));
    for (idx_t i = 0; i < N; i++) h[i] = rand();

    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice);

    printf("Sorting %d elements...\n", N);
    mergeSort_gpu(d, N);

    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (checkSorted(h, N))
        printf("OK: Array is sorted.\n");
    else
        printf("ERROR: Array is NOT sorted!\n");

    cudaFree(d);
    free(h);
    return 0;
}
