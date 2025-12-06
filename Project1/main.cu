#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define BLOCK_SIZE 1024          // Threads per block
#define ELEMENTS_PER_BLOCK 2048  // Elements sorted per block

// ================================================================
// CPU helper to check correctness
// ================================================================
bool checkSorted(int* arr, int n)
{
    for (int i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1])
            return false;
    return true;
}

// ================================================================
// Device function: merge two sorted subarrays in shared memory
// ================================================================
__device__ void mergeShared(int* sdata, int start, int mid, int end, int* temp)
{
    int i = start;
    int j = mid + 1;
    int k = start;

    while (i <= mid && j <= end)
        temp[k++] = (sdata[i] <= sdata[j]) ? sdata[i++] : sdata[j++];

    while (i <= mid) temp[k++] = sdata[i++];
    while (j <= end) temp[k++] = sdata[j++];

    for (int t = start; t <= end; t++)
        sdata[t] = temp[t];
}

// ================================================================
// Kernel: parallel in-block iterative merge
// ================================================================
__global__ void mergeSortInBlock(int* array, int n)
{
    __shared__ int sdata[ELEMENTS_PER_BLOCK];
    __shared__ int temp[ELEMENTS_PER_BLOCK];

    int blockStart = blockIdx.x * ELEMENTS_PER_BLOCK;
    if (blockStart >= n) return;

    int tid = threadIdx.x;

    // Copy block to shared memory
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        if (blockStart + i < n)
            sdata[i] = array[blockStart + i];
        else
            sdata[i] = INT_MAX; // padding
    }
    __syncthreads();

    // Iterative merge: size = 2, 4, 8, ..., ELEMENTS_PER_BLOCK
    for (int size = 2; size <= ELEMENTS_PER_BLOCK; size <<= 1)
    {
        int numIntervals = ELEMENTS_PER_BLOCK / size;
        int intervalId = tid;
        if (intervalId < numIntervals)
        {
            int start = intervalId * size;
            int mid   = start + size / 2 - 1;
            int end   = start + size - 1;
            mergeShared(sdata, start, mid, end, temp);
        }
        __syncthreads();
    }

    // Copy back to global memory
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        if (blockStart + i < n)
            array[blockStart + i] = sdata[i];
    }
}

// ================================================================
// Kernel: merge sorted blocks in global memory
// ================================================================
__global__ void mergeBlocks(int* array, int* temp, int n)
{
    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    int tid = threadIdx.x;

    for (int size = 1; size < numBlocks; size <<= 1)
    {
        int startBlock = tid * (size * 2);
        if (startBlock + size < numBlocks)
        {
            int l   = startBlock * ELEMENTS_PER_BLOCK;
            int mid = l + size * ELEMENTS_PER_BLOCK - 1;
            int r   = l + 2 * size * ELEMENTS_PER_BLOCK - 1;
            if (r >= n) r = n - 1;

            int i = l, j = mid + 1, k = l;
            while (i <= mid && j <= r)
                temp[k++] = (array[i] <= array[j]) ? array[i++] : array[j++];
            while (i <= mid) temp[k++] = array[i++];
            while (j <= r)   temp[k++] = array[j++];
            for (int t = l; t <= r; t++)
                array[t] = temp[t];
        }
        __syncthreads();
    }
}

// ================================================================
// Host function: run GPU merge sort
// ================================================================
void mergeSort_gpu(int* d_array, int n)
{
    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    size_t sharedMemSize = ELEMENTS_PER_BLOCK * sizeof(int);

    // Phase 1: parallel in-block merge
    mergeSortInBlock<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_array, n);
    cudaDeviceSynchronize();

    // Phase 2: merge sorted blocks
    int* d_temp;
    cudaMalloc(&d_temp, n * sizeof(int));
    mergeBlocks<<<1, numBlocks>>>(d_array, d_temp, n);
    cudaDeviceSynchronize();
    cudaFree(d_temp);
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    const int N = 1 << 15; // 32768 elements

    int* h = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h[i] = rand();

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
