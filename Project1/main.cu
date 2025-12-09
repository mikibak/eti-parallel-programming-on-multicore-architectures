#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>

using idx_t = long long int;

bool checkSorted(int* arr, idx_t n)
{
    for (idx_t i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1]) return false;
    return true;
}

__global__
void mergeRunsKernel(const int* src, int* dst, idx_t n, idx_t runSize)
{
    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t pairSize = runSize * 2;
    idx_t start = tid * pairSize;
    if (start >= (idx_t)n) return;

    idx_t mid = start + runSize;
    if (mid > n) mid = n;
    idx_t end = start + pairSize;
    if (end > n) end = n;

    // If no right run, just copy
    if (mid >= end) {
        for (idx_t t = start; t < end; ++t) dst[t] = src[t];
        return;
    }

    idx_t i = start;
    idx_t j = mid;
    idx_t k = start;

    while (i < mid && j < end) dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < mid) dst[k++] = src[i++];
    while (j < end) dst[k++] = src[j++];
}

void mergeSort_gpu_anyN(int* h, idx_t n)
{
    if (n <= 0) return;

    int* d_buf1;
    int* d_buf2;
    cudaMalloc(&d_buf1, n * sizeof(int));
    cudaMalloc(&d_buf2, n * sizeof(int));
    cudaMemcpy(d_buf1, h, n * sizeof(int), cudaMemcpyHostToDevice);

    int* src = d_buf1;
    int* dst = d_buf2;

    idx_t runSize = 1;
    const int threadsPerBlock = 256;
    int passCount = 0;

    while (runSize < n)
    {
        idx_t pairs = (n + (runSize * 2) - 1) / (runSize * 2);
        int blocks = (int)((pairs + threadsPerBlock - 1) / threadsPerBlock);

        mergeRunsKernel<<<blocks, threadsPerBlock>>>(src, dst, n, runSize);
        cudaDeviceSynchronize();

        // swap
        int* tmp = src; src = dst; dst = tmp;

        runSize <<= 1;
        passCount++;
    }

    // Determine which buffer has the final sorted data
    int* d_final = (passCount % 2 == 0) ? d_buf1 : d_buf2;
    cudaMemcpy(h, d_final, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_buf1);
    cudaFree(d_buf2);
}

int main()
{
    idx_t N;
    printf("Enter number of elements: ");
    scanf("%d", &N);

    int* arr = (int*)malloc(N * sizeof(int));
    int* backup = (int*)malloc(N * sizeof(int));

    float times[10];
    srand(12345);

    for (int r = 0; r < 10; ++r)
    {
        for (idx_t i = 0; i < N; i++) backup[i] = rand();
        memcpy(arr, backup, N * sizeof(int));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        mergeSort_gpu_anyN(arr, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[r] = ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (!checkSorted(arr, N)) {
            printf("ERROR: Array NOT sorted on run %d!\n", r + 1);
            return 1;
        }
        printf("Run %d sorted, time = %.3f ms\n", r + 1, ms);
    }

    float sum = 0.0f;
    float tmin = times[0], tmax = times[0];
    for (int i = 0; i < 10; ++i) {
        sum += times[i];
        if (times[i] < tmin) tmin = times[i];
        if (times[i] > tmax) tmax = times[i];
    }
    float mean = sum / 10.0f;
    float uncertainty = (tmax - tmin) / 2.0f;

    printf("\nAverage time over 10 runs: %.3f ms\n", mean);
    printf("Uncertainty (Tmax-Tmin)/2: ± %.3f ms\n", uncertainty);

    free(arr);
    free(backup);
    return 0;
}
