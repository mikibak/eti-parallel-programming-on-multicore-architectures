#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>

#define BLOCK_SIZE 1024
#define ELEMENTS_PER_BLOCK 2048
#define NSTREAMS 4

using idx_t = int;

bool checkSorted(int* arr, idx_t n)
{
    for (idx_t i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1]) return false;
    return true;
}

__device__ 
void mergeShared(int* src, int* dst, int start, int mid, int end)
{
    int i = start, j = mid + 1, k = start;
    while (i <= mid && j <= end) dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i <= mid) dst[k++] = src[i++];
    while (j <= end) dst[k++] = src[j++];
}

__global__ 
void mergeSortInBlock(int* array, idx_t blockElems)
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

__global__ 
void mergeBlockPairs(int* arr, int* temp, int n, int intervalBlocks)
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

void mergeSort_gpu_streamed(int* h, idx_t n)
{
    int* h_pinned;
    cudaMallocHost(&h_pinned, n * sizeof(int));
    memcpy(h_pinned, h, n * sizeof(int));

    int* d_full;
    cudaMalloc(&d_full, n * sizeof(int));

    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++)
        cudaStreamCreate(&streams[i]);

    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    int* d_blocks[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++)
        cudaMalloc(&d_blocks[i], ELEMENTS_PER_BLOCK * sizeof(int));

    // phase 1: per-block sort
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

        cudaMemcpyAsync(
            d_full + offset,
            d_blocks[streamId],
            size * sizeof(int),
            cudaMemcpyDeviceToDevice,
            streams[streamId]
        );
    }

    cudaDeviceSynchronize();

    // phase 2: global merging
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

    cudaMemcpy(h, d_full, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_full);

    for (int i = 0; i < NSTREAMS; i++)
    {
        cudaFree(d_blocks[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_pinned);
}

int main()
{
    idx_t N;
    printf("Enter number of elements: ");
    scanf("%d", &N);

    int* arr = (int*)malloc(N * sizeof(int));
    int* backup = (int*)malloc(N * sizeof(int));

    float times[10];

    for (int r = 0; r < 10; r++)
    {
        for (idx_t i = 0; i < N; i++) backup[i] = rand();
        memcpy(arr, backup, N * sizeof(int));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        mergeSort_gpu_streamed(arr, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times[r] = ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (!checkSorted(arr, N))
        {
            printf("ERROR: Array NOT sorted!\n");
            return 0;
        }
        printf("Sorted array for run %d\n", r + 1);
    }

    float sum = 0, tmin = times[0], tmax = times[0];
    for (int i = 0; i < 10; i++)
    {
        sum += times[i];
        tmin = std::min(tmin, times[i]);
        tmax = std::max(tmax, times[i]);
    }

    float mean = sum / 10.0f;
    float uncertainty = (tmax - tmin) / 2.0f;

    printf("\nAverage time over 10 runs: %.3f ms\n", mean);
    printf("Uncertainty (Tmax-Tmin)/2: ± %.3f ms\n", uncertainty);

    free(arr);
    free(backup);
    return 0;
}
