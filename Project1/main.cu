#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define BLOCK_SIZE 1024          // Threads per block in phase 1
#define ELEMENTS_PER_BLOCK 2048  // Each block sorts 2048 elements

// ============================================================================
// CPU helper for debugging
// ============================================================================
bool checkSorted(int* arr, int n)
{
    for (int i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1])
            return false;
    return true;
}

// ============================================================================
// DEVICE FUNCTION: Merge two sorted subarrays
//
// Merges array[l..mid] and array[mid+1..r] into a temp buffer
// and copies result back into array.
//
// This is a simple sequential merge — but because each thread merges
// a different interval, the merges are parallel at the grid level.
// ============================================================================
__device__ void merge_gpu(int* array, int l, int mid, int r, int* temp)
{
    int i = l;
    int j = mid + 1;
    int k = l;

    while (i <= mid && j <= r)
    {
        if (array[i] <= array[j])
            temp[k++] = array[i++];
        else
            temp[k++] = array[j++];
    }

    while (i <= mid) temp[k++] = array[i++];
    while (j <= r)   temp[k++] = array[j++];

    // Copy merged result back to global memory
    for (int t = l; t <= r; t++)
        array[t] = temp[t];
}

// ============================================================================
// KERNEL 1: mergeSortInBlock_gpu
//
// Each block sorts exactly 2048 elements.
//
// Each block:
// 1. Loads its 2048 elements from global to shared memory
// 2. Performs iterative merging inside shared memory:
//      size = 2, 4, 8, 16, ... 2048
// 3. Writes the 2048 sorted elements back to global memory
//
// This kernel produces n/2048 sorted blocks.
// ============================================================================
__global__ void mergeSortInBlock_gpu(int* array, int n)
{
    extern __shared__ int sdata[];  // shared memory buffer (size = 2048 ints)

    int blockStart = blockIdx.x * ELEMENTS_PER_BLOCK;
    if (blockStart >= n) return;

    // Copy the block's elements to shared memory
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        if (blockStart + i < n)
            sdata[i] = array[blockStart + i];
        else
            sdata[i] = 2147483647;  // pad with large sentinel
    }
    __syncthreads();

    // Iterative merge: 2 → 4 → 8 → ... → 2048
    for (int size = 2; size <= ELEMENTS_PER_BLOCK; size <<= 1)
    {
        int half = size >> 1;

        int myStart = threadIdx.x * size;
        int l = myStart;
        int r = myStart + size - 1;

        if (r < ELEMENTS_PER_BLOCK)
        {
            int mid = l + half - 1;

            // Each thread merges two sorted halves inside shared memory
            int tempL = l;
            int tempR = r;

            __syncthreads();

            int i = l;
            int j = mid + 1;
            int k = l;

            __syncthreads();

            // Local merge into a temporary shared buffer region (reuse top half)
            while (i <= mid && j <= r)
                sdata[k++] = (sdata[i] <= sdata[j]) ? sdata[i++] : sdata[j++];
            while (i <= mid) sdata[k++] = sdata[i++];
            while (j <= r)   sdata[k++] = sdata[j++];
        }

        __syncthreads();
    }

    // Write sorted block to global memory
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        if (blockStart + i < n)
            array[blockStart + i] = sdata[i];
    }
}

// ============================================================================
// KERNEL 2: mergeSortInGrid_gpu
//
// After phase 1, the array consists of (n/2048) sorted blocks.
//
// Now the grid has 1 block and (n/2048) threads.
// Each thread merges two sorted blocks at a time.
//
// Iteration structure:
// size = 1 block, 2 blocks, 4 blocks, 8 blocks, ...
//
// Thread t merges:
//   [t * size * 2048 ... (t*size + size)*2048 - 1]
//
// ============================================================================
__global__ void mergeSortInGrid_gpu(int* array, int n, int threads_in_block)
{
    extern __shared__ int temp[];

    int numBlocks = n / ELEMENTS_PER_BLOCK;
    int tid = threadIdx.x;

    // Each iteration merges "size" blocks
    for (int size = 1; size < numBlocks; size <<= 1)
    {
        int startBlock = tid * (size * 2);
        if (startBlock + size < numBlocks)
        {
            int l = startBlock * ELEMENTS_PER_BLOCK;
            int mid = l + size * ELEMENTS_PER_BLOCK - 1;
            int r = l + 2 * size * ELEMENTS_PER_BLOCK - 1;

            if (r >= n) r = n - 1;

            merge_gpu(array, l, mid, r, temp);
        }

        __syncthreads();
    }
}

// ============================================================================
// HOST FUNCTION: mergeSort_gpu
// ============================================================================
void mergeSort_gpu(int* d_array, int n)
{
    int numBlocks = n / ELEMENTS_PER_BLOCK;

    size_t shared1 = ELEMENTS_PER_BLOCK * sizeof(int);
    mergeSortInBlock_gpu<<<numBlocks, BLOCK_SIZE, shared1>>>(d_array, n);
    cudaDeviceSynchronize();

    size_t shared2 = n * sizeof(int);
    mergeSortInGrid_gpu<<<1, numBlocks, shared2>>>(d_array, n, BLOCK_SIZE);
    cudaDeviceSynchronize();
}

// ============================================================================
// MAIN
// ============================================================================
int main()
{
    const int N = 1 << 5;

    int* h = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h[i] = rand();

    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice);

    printf("Sorting...\n");
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
