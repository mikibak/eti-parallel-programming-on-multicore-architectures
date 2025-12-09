#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>

#define BLOCK_SIZE 256
#define NSTREAMS 4

using idx_t = long long int; // for large arrays

bool checkSorted(int* arr, idx_t n)
{
    for (idx_t i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1]) return false;
    return true;
}

__device__
void mergeGlobal(int* src, int* dst, idx_t start, idx_t mid, idx_t end)
{
    idx_t i = start, j = mid + 1, k = start;
    while (i <= mid && j <= end) dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i <= mid) dst[k++] = src[i++];
    while (j <= end) dst[k++] = src[j++];
}

__global__
void mergeKernel(int* arr, int* temp, idx_t n, idx_t runSize)
{
    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t totalThreads = gridDim.x * blockDim.x;

    idx_t totalPairs = (n + 2 * runSize - 1) / (2 * runSize);

    for (idx_t pair = tid; pair < totalPairs; pair += totalThreads)
    {
        idx_t start = pair * 2 * runSize;
        if (start >= n) continue;

        idx_t mid = min(start + runSize - 1, n - 1);
        idx_t end = min(start + 2 * runSize - 1, n - 1);

        if (mid >= end) continue; // only one run left, no merge needed

        mergeGlobal(arr, temp, start, mid, end);
    }
}

void mergeSort_gpu_streamed(int* h, idx_t n)
{
    int* h_pinned;
    cudaMallocHost(&h_pinned, n * sizeof(int));
    memcpy(h_pinned, h, n * sizeof(int));

    int* d_full;
    int* d_temp;
    cudaMalloc(&d_full, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));

    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++)
        cudaStreamCreate(&streams[i]);

    // copy input to GPU using streams
    idx_t chunkSize = 1 << 20; // 1M elements per stream chunk
    idx_t chunks = (n + chunkSize - 1) / chunkSize;
    for (idx_t i = 0; i < chunks; i++)
    {
        int streamId = i % NSTREAMS;
        idx_t offset = i * chunkSize;
        idx_t size = std::min(chunkSize, n - offset);
        cudaMemcpyAsync(d_full + offset, h_pinned + offset, size * sizeof(int),
                        cudaMemcpyHostToDevice, streams[streamId]);
    }
    cudaDeviceSynchronize();

    // Phase 1: iterative merge
    idx_t runSize = 1;
    while (runSize < n)
    {
        idx_t totalPairs = (n + 2 * runSize - 1) / (2 * runSize);
        int threads = BLOCK_SIZE;
        int blocks  = (totalPairs + threads - 1) / threads;

        mergeKernel<<<blocks, threads>>>(d_full, d_temp, n, runSize);
        cudaDeviceSynchronize();

        // swap buffers
        int* tmp = d_full;
        d_full = d_temp;
        d_temp = tmp;

        runSize <<= 1;
    }

    // copy back to host
    cudaMemcpy(h, d_full, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_full);
    for (int i = 0; i < NSTREAMS; i++)
        cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_pinned);
}

int main()
{
    idx_t N;
    printf("Enter number of elements: ");
    scanf("%lld", &N);

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
            printf("ERROR: Array NOT sorted on run %d!\n", r + 1);
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
