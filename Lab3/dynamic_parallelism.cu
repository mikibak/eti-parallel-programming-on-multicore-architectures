#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_DEPTH 24

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

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void cdp_quicksort(int *data, int left, int right, int depth) {
    // 1. Depth Safety Check
    // If we hit this, the sort stops prematurely. We must warn the user.
    if (depth >= MAX_DEPTH) {
        if (threadIdx.x == 0) printf("ERROR: Max depth reached at index %d. Increase MAX_DEPTH.\n", left);
        return;
    }

    if (left >= right) {
        return;
    }

    int i = left - 1;

    // Only thread 0 manages the partition and children launches
    if (threadIdx.x == 0) {
        int pivot = data[right]; 
        
        for (int j = left; j <= right - 1; j++) {
            if (data[j] <= pivot) {
                i++;
                swap(&data[i], &data[j]);
            }
        }
        swap(&data[i + 1], &data[right]);
        
        int partition_index = i + 1;

        // 2. Dynamic Parallelism with Error Checking
        cudaError_t err;

        // Launch Left Child
        if (left < partition_index - 1) {
            cdp_quicksort<<<1, 1>>>(data, left, partition_index - 1, depth + 1);
            
            // Check if the launch actually worked
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Left Child Launch Failed (Depth %d): %d\n", depth, (int)err);
            }
        }

        // Launch Right Child
        if (partition_index + 1 < right) {
            cdp_quicksort<<<1, 1>>>(data, partition_index + 1, right, depth + 1);

            // Check if the launch actually worked
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Right Child Launch Failed (Depth %d): %d\n", depth, (int)err);
            }
        }
    }
}

int main(int argc, char **argv) {
    int N;
    printf("Enter number of elements: ");
    if(scanf("%d", &N) != 1) N = 1024; // Default if input fails

    int *h_data = (int *)malloc(N * sizeof(int));
    int *d_data;
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1000;
    }

    cudaCheckError(cudaMalloc((void **)&d_data, N * sizeof(int)));
    cudaCheckError(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    // --- CRITICAL CONFIGURATION START ---

    // 1. Increase Stack Size
    // Default is usually 1KB. Recursion requires significantly more stack per thread
    // to store return addresses and local variables for every nested call.
    // We set this to 8KB per thread.
    cudaCheckError(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    // 2. Pending Launch Count
    // If N is large, we might queue thousands of kernels before they execute.
    // The default is usually small (e.g., 2048).
    cudaCheckError(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768)); 

    // 3. Sync Depth
    // How deep the grid synchronization can go.
    cudaCheckError(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // --- CRITICAL CONFIGURATION END ---

    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    printf("Launching kernel with N=%d...\n", N);
    
    cudaCheckError(cudaEventRecord(start, 0));

    cdp_quicksort<<<1, 1>>>(d_data, 0, N - 1, 0);
    
    cudaCheckError(cudaEventRecord(stop, 0));
    cudaCheckError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Execution Time: %.5f ms\n", milliseconds);

    // This catches errors in the Parent kernel, but NOT the children
    cudaCheckError(cudaGetLastError());

    cudaCheckError(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    int correct = 1;
    for (int i = 0; i < N - 1; i++) {
        if (h_data[i] > h_data[i + 1]) {
            printf("Error at index %d: %d > %d\n", i, h_data[i], h_data[i+1]);
            correct = 0;
            break; // Stop at first error
        }
    }

    if (correct) {
        printf("SUCCESS: Array is sorted.\n");
    } else {
        printf("FAILURE: Sorting failed.\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_data);
    cudaFree(d_data);

    return 0;
}