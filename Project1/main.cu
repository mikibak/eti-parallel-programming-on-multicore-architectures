#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>

#define BLOCK_SIZE 1024
#define ELEMENTS_PER_BLOCK 2048       // sorted per block
#define NSTREAMS 4                    // parallel CPU→GPU + GPU kernels

using idx_t = int;

// ================================================================
// CPU check
// ================================================================
bool checkSorted(int* arr, idx_t n)
{
    for (idx_t i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1]) return false;
    return true;
}

// ================================================================
// Local merge inside shared memory
// ================================================================
__device__ void mergeShared(int* src, int* dst, int start, int mid, int end)
{
    int i = start, j = mid + 1, k = start;
    while (i <= mid && j <= end) dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i <= mid) dst[k++] = src[i++];
    while (j <= end) dst[k++] = src[j++];
}

// ================================================================
// In-block merging (sort 2048 elements)
// ================================================================
__global__ void mergeSortInBlock(int* array, idx_t blockElems)
{
    __shared__ int ping[ELEMENTS_PER_BLOCK];
    __shared__ int pong[ELEMENTS_PER_BLOCK];

    int tid = threadIdx.x;

    for (int i = tid; i < blockElems; i += blockDim.x)
        ping[i] = array[i];
    __syncthreads();

    int* src = ping;
    int* dst = pong;

    for (int size = 1; size < blockElems; size <<= 1)
    {
        int interval = size * 2;
        for (int start = tid * interval; start < blockElems; start += blockDim.x * interval)
        {
            int mid = min(start + size - 1, blockElems - 1);
            int end = min(start + interval - 1, blockElems - 1);
            mergeShared(src, dst, start, mid, end);
        }
        __syncthreads();
        int* tmp = src; src = dst; dst = tmp;
    }

    for (int i = tid; i < blockElems; i += blockDim.x)
        array[i] = src[i];
}

// ================================================================
// Merge sorted blocks globally
// ================================================================
__global__ void mergeBlockPairs(int* arr, int* temp, int n, int intervalBlocks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int blocksPerMerge = intervalBlocks * 2;
    int startBlock     = tid * blocksPerMerge;

    int totalBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    if (startBlock >= totalBlocks) return;

    int LB = startBlock;
    int RB = startBlock + intervalBlocks;
    if (RB >= totalBlocks) return;

    int L = LB * ELEMENTS_PER_BLOCK;
    int M = min(RB * ELEMENTS_PER_BLOCK - 1, n - 1);
    int R = min((startBlock + blocksPerMerge) * ELEMENTS_PER_BLOCK - 1, n - 1);

    int i = L, j = M + 1, k = L;

    while (i <= M && j <= R) temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i <= M) temp[k++] = arr[i++];
    while (j <= R) temp[k++] = arr[j++];

    for (int t = L; t <= R; t++) arr[t] = temp[t];
}

// ================================================================
// GPU merge sort with stream overlap
// ================================================================
void mergeSort_gpu_streamed(int* h, idx_t n)
{
    // Create pinned memory for async transfers
    int* h_pinned;
    cudaMallocHost(&h_pinned, n * sizeof(int));
    memcpy(h_pinned, h, n * sizeof(int));

    int* d_full;
    cudaMalloc(&d_full, n * sizeof(int));

    // Create streams
    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++)
        cudaStreamCreate(&streams[i]);

    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    // Allocate per-block memory on GPU
    int* d_blocks[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++)
        cudaMalloc(&d_blocks[i], ELEMENTS_PER_BLOCK * sizeof(int));

    // ========================
    // Phase 1: streaming copies + per-block sorting
    // ========================
    for (int block = 0; block < numBlocks; block++)
    {
        int streamId = block % NSTREAMS;

        int offset = block * ELEMENTS_PER_BLOCK;
        int size   = min(ELEMENTS_PER_BLOCK, n - offset);

        cudaMemcpyAsync(
            d_blocks[streamId],
            h_pinned + offset,
            size * sizeof(int),
            cudaMemcpyHostToDevice,
            streams[streamId]
        );

        mergeSortInBlock<<<1, BLOCK_SIZE, 0, streams[streamId]>>>(
            d_blocks[streamId], size
        );

        // copy block back into final device array
        cudaMemcpyAsync(
            d_full + offset,
            d_blocks[streamId],
            size * sizeof(int),
            cudaMemcpyDeviceToDevice,
            streams[streamId]
        );
    }

    // Sync all streams
    cudaDeviceSynchronize();

    // ========================
    // Phase 2: global merging using streams
    // ========================
    int* d_temp;
    cudaMalloc(&d_temp, n * sizeof(int));

    int intervalBlocks = 1;

    while (intervalBlocks < numBlocks)
    {
        int merges = (numBlocks + 2 * intervalBlocks - 1) / (2 * intervalBlocks);
        int threads = 256;
        int blocks  = (merges + threads - 1) / threads;

        mergeBlockPairs<<<blocks, threads>>>(d_full, d_temp, n, intervalBlocks);
        cudaDeviceSynchronize();

        intervalBlocks <<= 1;
    }

    // Copy final back
    cudaMemcpy(h, d_full, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_temp);
    cudaFree(d_full);

    for (int i = 0; i < NSTREAMS; i++)
    {
        cudaFree(d_blocks[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_pinned);
}

// ================================================================
// MAIN
// ================================================================
int main()
{
    idx_t N;
    printf("Enter number of elements: ");
    scanf("%d", &N);

    int* arr = (int*)malloc(N * sizeof(int));
    for (idx_t i = 0; i < N; i++) arr[i] = rand();

    mergeSort_gpu_streamed(arr, N);

    if (checkSorted(arr, N))
        printf("OK: Array sorted.\n");
    else
        printf("ERROR: Array NOT sorted.\n");

    free(arr);
    return 0;
}
