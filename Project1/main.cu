#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define BLOCK_SIZE 1024
#define ELEMENTS_PER_BLOCK 2048

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
// Device function: merge two sorted subarrays in shared memory
// ================================================================
__device__ void mergeShared(int* src, int* dst, idx_t start, idx_t mid, idx_t end)
{
    idx_t i = start;
    idx_t j = mid + 1;
    idx_t k = start;

    while (i <= mid && j <= end)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];

    while (i <= mid) dst[k++] = src[i++];
    while (j <= end) dst[k++] = src[j++];
}

// ================================================================
// Kernel: parallel in-block iterative merge with ping-pong
// ================================================================
__global__ void mergeSortInBlock(int* array, idx_t n)
{
    __shared__ int ping[ELEMENTS_PER_BLOCK];
    __shared__ int pong[ELEMENTS_PER_BLOCK];

    idx_t blockStart = blockIdx.x * ELEMENTS_PER_BLOCK;
    if (blockStart >= n) return;

    idx_t tid = threadIdx.x;

    // Copy block to shared memory (ping)
    for (idx_t i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
        ping[i] = (blockStart + i < n) ? array[blockStart + i] : INT_MAX;

    __syncthreads();

    int* src = ping;
    int* dst = pong;

    // Iterative merge
    for (idx_t size = 2; size <= ELEMENTS_PER_BLOCK; size <<= 1)
    {
        idx_t numIntervals = ELEMENTS_PER_BLOCK / size;
        idx_t intervalId = tid;
        if (intervalId < numIntervals)
        {
            idx_t start = intervalId * size;
            idx_t mid   = start + size / 2 - 1;
            idx_t end   = start + size - 1;
            mergeShared(src, dst, start, mid, end);
        }
        __syncthreads();

        // Swap buffers
        int* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Copy back to global memory
    for (idx_t i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
        if (blockStart + i < n)
            array[blockStart + i] = src[i];
}

// ================================================================
// Kernel: parallel merge sorted blocks in global memory
// ================================================================
__global__ void mergeBlocks(int* array, int* temp, idx_t n)
{
    idx_t numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    idx_t tid = threadIdx.x;

    // Each thread handles one pair of blocks per iteration
    for (idx_t size = 1; size < numBlocks; size <<= 1)
    {
        idx_t startBlock = tid * (size * 2);
        if (startBlock + size < numBlocks)
        {
            idx_t l   = startBlock * ELEMENTS_PER_BLOCK;
            idx_t mid = l + size * ELEMENTS_PER_BLOCK - 1;
            idx_t r   = l + 2 * size * ELEMENTS_PER_BLOCK - 1;
            if (r >= n) r = n - 1;

            idx_t i = l, j = mid + 1, k = l;
            while (i <= mid && j <= r)
                temp[k++] = (array[i] <= array[j]) ? array[i++] : array[j++];
            while (i <= mid) temp[k++] = array[i++];
            while (j <= r)   temp[k++] = array[j++];
            for (idx_t t = l; t <= r; t++)
                array[t] = temp[t];
        }
        __syncthreads();
    }
}

// ================================================================
// Host function: run GPU merge sort with total timing
// ================================================================
void mergeSort_gpu(int* d_array, idx_t n)
{
    idx_t numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    size_t sharedMemSize = ELEMENTS_PER_BLOCK * sizeof(int);

    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Phase 1: parallel in-block merge
    mergeSortInBlock<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_array, n);
    cudaDeviceSynchronize();

    // Phase 2: merge sorted blocks
    int* d_temp;
    cudaMalloc(&d_temp, n * sizeof(int));
    mergeBlocks<<<1, numBlocks>>>(d_array, d_temp, n);
    cudaDeviceSynchronize();
    cudaFree(d_temp);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
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
