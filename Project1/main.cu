#include <cstdio>
#include <cstdlib>
#include <algorithm>

#undef max
#undef min

typedef int idx_t;

// -----------------------------------------------------------------------------
// Merge path partitioning function (Green's algorithm)
// -----------------------------------------------------------------------------
__device__ idx_t merge_path_search(
    idx_t diag,
    const int* A, idx_t a_len,
    const int* B, idx_t b_len)
{
    // search range
    idx_t lo = max((idx_t)0, diag - b_len);
    idx_t hi = min(diag, a_len);

    while (lo < hi) {
        idx_t mid = (lo + hi) >> 1;
        int a = A[mid];
        int b = B[diag - mid - 1];

        if (a < b)
            lo = mid + 1;
        else
            hi = mid;
    }

    return lo;
}

// -----------------------------------------------------------------------------
// Merge kernel using merge-path
// -----------------------------------------------------------------------------
__global__ void merge_kernel(
    const int* A, idx_t a_len,
    const int* B, idx_t b_len,
    int* C)
{
    idx_t total = a_len + b_len;

    idx_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    idx_t tcount = blockDim.x * gridDim.x;

    // Number of items per thread (parallel merge)
    for (idx_t diag = tid; diag < total; diag += tcount)
    {
        idx_t i = merge_path_search(diag, A, a_len, B, b_len);
        idx_t j = diag - i;

        int a_val = (i < a_len) ? A[i] : INT_MAX;
        int b_val = (j < b_len) ? B[j] : INT_MAX;

        C[diag] = (a_val <= b_val ? a_val : b_val);
    }
}

// -----------------------------------------------------------------------------
// Host function: merge sort driver (iterative, bottom-up)
// -----------------------------------------------------------------------------
void gpu_merge_sort(int* data, idx_t N)
{
    int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_in, data, N * sizeof(int), cudaMemcpyHostToDevice);

    const int BLOCK = 256;
    const int GRID  = 256;

    for (idx_t L = 1; L < N; L <<= 1)
    {
        idx_t num_pairs = (N + 2*L - 1) / (2*L);

        for (idx_t p = 0; p < num_pairs; ++p)
        {
            idx_t a0 = p * (2 * L);
            idx_t b0 = a0 + L;

            idx_t a_len = min(L, N - a0);
            idx_t b_len = (b0 < N) ? min(L, N - b0) : 0;

            if (b_len == 0) {
                cudaMemcpyAsync(
                    d_out + a0,
                    d_in + a0,
                    a_len * sizeof(int),
                    cudaMemcpyDeviceToDevice);
                continue;
            }

            merge_kernel<<<GRID, BLOCK>>>(
                d_in + a0, a_len,
                d_in + b0, b_len,
                d_out + a0
            );
        }

        cudaDeviceSynchronize();
        std::swap(d_in, d_out);
    }

    cudaMemcpy(data, d_in, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

// -----------------------------------------------------------------------------
// Main Test
// -----------------------------------------------------------------------------
int main()
{
    const int N = 1 << 20;
    int* arr = new int[N];

    for (int i = 0; i < N; ++i)
        arr[i] = rand();

    printf("Sorting...\n");
    gpu_merge_sort(arr, N);

    for (int i = 1; i < N; ++i) {
        if (arr[i] < arr[i-1]) {
            printf("ERROR: array not sorted at index %d\n", i);
            return 1;
        }
    }

    printf("OK: sorted successfully!\n");
    delete[] arr;
    return 0;
}
