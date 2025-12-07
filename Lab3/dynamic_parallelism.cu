/*
CUDA - dynamic parallelism sample
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1024
#define MAX_DEPTH 24

// Error handling macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// The Recursive QuickSort Kernel
__global__ void cdp_quicksort(int *data, int left, int right, int depth) {
    // 1. Base Case: If the array segment is size 0 or 1, or depth limit reached
    if (left >= right || depth >= MAX_DEPTH) {
        return;
    }

    // 2. Partitioning Step (executed by thread 0 of this block)
    // We strictly use kernel recursion even for tiny arrays now.
    int i = left - 1;

    if (threadIdx.x == 0) {
        int pivot = data[right]; // Lomuto partition
        
        for (int j = left; j <= right - 1; j++) {
            if (data[j] <= pivot) {
                i++;
                swap(&data[i], &data[j]);
            }
        }
        swap(&data[i + 1], &data[right]);
        
        int partition_index = i + 1;

        // 3. Dynamic Parallelism: Launch Child Kernels
        // Launch Left Child
        if (left < partition_index - 1) {
            cdp_quicksort<<<1, 1>>>(data, left, partition_index - 1, depth + 1);
        }

        // Launch Right Child
        if (partition_index + 1 < right) {
            cdp_quicksort<<<1, 1>>>(data, partition_index + 1, right, depth + 1);
        }
    }
}

int main(int argc, char **argv) {
    printf("--- CUDA Dynamic Parallelism QuickSort (Pure Recursion) ---\n");

    // 1. Setup Host Data
    int *h_data = (int *)malloc(N * sizeof(int));
    int *d_data;
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1000;
    }

    // 2. Allocate and Copy to Device
    cudaCheckError(cudaMalloc((void **)&d_data, N * sizeof(int)));
    cudaCheckError(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    // 3. Set Device Limits
    // CRITICAL: Without selection_sort, we generate many more tiny kernels.
    // We ensure the pending launch count is high enough.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768); 

    // 4. Setup Timing Events
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // 5. Launch Parent Kernel with Timing
    printf("Launching kernel...\n");
    
    // Record start event
    cudaCheckError(cudaEventRecord(start, 0));

    cdp_quicksort<<<1, 1>>>(d_data, 0, N - 1, 0);
    
    // Record stop event
    cudaCheckError(cudaEventRecord(stop, 0));
    
    // Wait for the stop event to complete (this implies waiting for the kernel flow)
    cudaCheckError(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Execution Time: %.5f ms\n", milliseconds);

    // Check for kernel errors after sync
    cudaCheckError(cudaGetLastError());

    // 6. Copy back and Verify
    cudaCheckError(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify sort
    int correct = 1;
    for (int i = 0; i < N - 1; i++) {
        if (h_data[i] > h_data[i + 1]) {
            printf("Error at index %d: %d > %d\n", i, h_data[i], h_data[i+1]);
            correct = 0;
            break;
        }
    }

    if (correct) {
        printf("SUCCESS: Array is sorted.\n");
    } else {
        printf("FAILURE: Sorting failed.\n");
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_data);
    cudaFree(d_data);

    return 0;
}