#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_RECURSION_DEPTH 18
#define MIN_SIZE            32   // below this size do insertion sort
#define THREADS_PER_BLOCK   128

// Host side error checker
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ 
void insertionSort(int *data, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = data[i];
        int j = i - 1;
        while (j >= left && data[j] > key) {
            data[j + 1] = data[j];
            j--;
        }
        data[j + 1] = key;
    }
}

__device__
int partition(int *data, int left, int right) {
    int pivot = data[(left + right) / 2];
    int i = left;
    int j = right;

    while (i <= j) {
        while (data[i] < pivot) i++;
        while (data[j] > pivot) j--;
        if (i <= j) {
            int tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
            i++;
            j--;
        }
    }
    return i;
}

__global__
void quicksortKernel(int *data, int left, int right, int depth) {
    cudaStream_t s1, s2;

    if (depth >= MAX_RECURSION_DEPTH || right - left < MIN_SIZE) {
        insertionSort(data, left, right);
        return;
    }

    int index = partition(data, left, right);

    int leftEnd = index - 1;
    int rightStart = index;

    if (left < leftEnd) {
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        quicksortKernel<<<1, 1, 0, s1>>>(data, left, leftEnd, depth + 1);
    }

    if (rightStart < right) {
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        quicksortKernel<<<1, 1, 0, s2>>>(data, rightStart, right, depth + 1);
    }
}

int main(int argc, char **argv) {

    int N;

    printf("Enter number of elements: ");
    fflush(stdout);
    scanf("%d", &N);

    int *h_data = (int*)malloc(N * sizeof(int));
    int *d_data;

    if (!h_data) {
        printf("Host malloc failed");
        return -1;
    }

    for (int i = 0; i < N; i++)
        h_data[i] = rand();

    cudaCheckError(cudaMalloc(&d_data, N * sizeof(int)));

    cudaCheckError(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_RECURSION_DEPTH));

    printf("Sorting...\n");

    // start kernel
    quicksortKernel<<<1, 1>>>(d_data, 0, N - 1, 0);

    cudaCheckError(cudaGetLastError());

    cudaDeviceSynchronize();

    cudaCheckError(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // check correctness
    for (int i = 0; i < N - 1; i++) {
        if (h_data[i] > h_data[i + 1]) {
            printf("ERROR: array NOT sorted!\n");
            exit(0);
        }
    }

    printf("Array is sorted.\n");

    free(h_data);
    cudaFree(d_data);
    return 0;
}
