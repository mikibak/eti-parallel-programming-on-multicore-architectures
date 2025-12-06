#include <cstdio>
#include <cstdlib>
#include <algorithm>

#undef max
#undef min

typedef int idx_t;

// ------------------------------------------------------------
// Device: merge two sorted subarrays A and B into output C
// ------------------------------------------------------------
__device__ void gpu_merge(
    const int* A, idx_t a_len,
    const int* B, idx_t b_len,
    int* C)
{
    idx_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx_t total = a_len + b_len;
    if (idx >= total) return;

    // binary search partitioning:
    idx_t loA = max((idx_t)0, idx - b_len);
    idx_t hiA = min(a_len, idx);

    while (loA < hiA) {
        idx_t mid = (loA + hiA) >> 1;
        if (A[mid] <= B[idx - mid - 1])
            loA = mid + 1;
        else
            hiA = mid;
    }

    idx_t a_idx = loA;
    idx_t b_idx = idx - loA;

    int a_val = (a_idx < a_len) ? A[a_idx] : INT_MAX;
    int b_val = (b_idx < b_len) ? B[b_idx] : INT_MAX;

    C[idx] = (a_val <= b_val) ? a_val : b_val;
}

// ------------------------------------------------------------
// Kernel: merge many pairs of sorted subarrays
// ------------------------------------------------------------
__global__ void merge_pass_kernel(
    const int* input,
    int* output,
    idx_t N,
    idx_t L)
{
    idx_t pair_idx = blockIdx.y;  // pair index in this merge level
    idx_t a0 = pair_idx * (2 * L);
    idx_t b0 = a0 + L;

    if (a0 >= N) return;

    idx_t a_len = min(L, N - a0);
    idx_t b_len = (b0 < N) ? min(L, N - b0) : 0;

    const int* A = input + a0;
    const int* B = input + b0;
    int* C = output + a0;

    idx_t total = a_len + b_len;

    idx_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total) return;

    // binary search mapping
    idx_t loA = max((idx_t)0, idx - b_len);
    idx_t hiA = min(a_len, idx);

    while (loA < hiA) {
        idx_t mid = (loA + hiA) >> 1;
        if (A[mid] <= B[idx - mid - 1])
            loA = mid + 1;
        else
            hiA = mid;
    }

    idx_t a_idx = loA;
    idx_t b_idx = idx - loA;

    int a_val = (a_idx < a_len) ? A[a_idx] : INT_MAX;
    int b_val = (b_idx < b_len) ? B[b_idx] : INT_MAX;

    C[idx] = (a_val <= b_val) ? a_val : b_val;
}

// ------------------------------------------------------------
// Host: GPU merge sort (iterative, bottom-up)
// ------------------------------------------------------------
void gpu_merge_sort(int* data, idx_t N)
{
    int* d_in = nullptr;
    int* d_out = nullptr;

    cudaMalloc(&d_in,  N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, data, N * sizeof(int), cudaMemcpyHostToDevice);

    for (idx_t L = 1; L < N; L <<= 1) {
        idx_t num_pairs = (N + 2 * L - 1) / (2 * L);

        dim3 block(256);
        dim3 grid((2 * L + block.x - 1) / block.x, num_pairs);

        merge_pass_kernel<<<grid, block>>>(d_in, d_out, N, L);
        cudaDeviceSynchronize();

        // swap buffers
        std::swap(d_in, d_out);
    }

    cudaMemcpy(data, d_in, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

// ------------------------------------------------------------
// Main for testing
// ------------------------------------------------------------
int main()
{
    const int N = 1 << 20; // 1M elements
    int* arr = new int[N];

    // fill with random values
    for (int i = 0; i < N; ++i)
        arr[i] = rand();

    printf("Sorting...\n");

    gpu_merge_sort(arr, N);

    // verify
    for (int i = 1; i < N; ++i) {
        if (arr[i] < arr[i - 1]) {
            printf("ERROR: array not sorted at index %d\n", i);
            delete[] arr;
            return 1;
        }
    }

    printf("OK: sorted successfully!\n");

    delete[] arr;
    return 0;
}
