// gpu_merge_sort.cu
// Compile: nvcc -O3 -arch=sm_60 gpu_merge_sort.cu -o gpu_merge_sort
// Example run: ./gpu_merge_sort 1000000
//
// Implements hybrid GPU merge sort:
//  - Stage 0: block-local bitonic sort of RUN-sized blocks (shared memory).
//  - Stage 1: iterative merge passes doubling run length each pass.
// Uses ping-pong buffers on device. Measures GPU time and verifies correctness.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <random>
#include <cassert>
#include <cstdint>
#include <limits>

using key_t = int32_t;             // element type
using idx_t = int64_t;             // for indexing large arrays safely

// Tunable parameters
constexpr int RUN = 1024;          // run size sorted inside a block (must be power of two)
constexpr int MERGE_TPB = 256;     // threads per block for merge_pass

inline void checkCuda(cudaError_t err, const char* msg = nullptr) {
    if (err != cudaSuccess) {
        if (msg) fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        else fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

// -------------------- Device helpers --------------------

__device__ __forceinline__
idx_t clamp_idx(idx_t x, idx_t lo, idx_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// lower_bound on device in [l, r) for val; returns smallest i such that arr[i] >= val
__device__ idx_t lower_bound_dev(const key_t* arr, idx_t l, idx_t r, key_t val) {
    idx_t lo = l, hi = r;
    while (lo < hi) {
        idx_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// -------------------- Stage 0: in-block bitonic sort --------------------
// Each block sorts RUN elements; threads per block == RUN (1 thread per element).
// Shared memory used to store a run. RUN must be a power of two.
__global__ void block_bitonic_sort_kernel(key_t* data, idx_t N) {
    extern __shared__ key_t s[]; // size RUN
    idx_t blockStart = (idx_t)blockIdx.x * RUN;
    int tid = threadIdx.x;
    idx_t gidx = blockStart + tid;

    // load into shared mem (pad with +INF if out-of-range)
    key_t val = (gidx < N) ? data[gidx] : std::numeric_limits<key_t>::max();
    s[tid] = val;
    __syncthreads();

    // bitonic sort (classic implementation)
    for (int k = 2; k <= RUN; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                key_t a = s[tid];
                key_t b = s[ixj];
                bool ascend = ((tid & k) == 0);
                if ((a > b) == ascend) {
                    // swap
                    s[tid] = b;
                    s[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // write back clipped to N
    if (gidx < N) data[gidx] = s[tid];
}

// -------------------- Stage 1: merge pass --------------------
// Each block merges two runs of length L: A=[a0..a0+a_len-1], B=[b0..b0+b_len-1].
// Each thread t computes an output segment [startOut,endOut) and finds corresponding splits in A/B
// using diagonal binary search; then merges that small chunk sequentially.
__global__ void merge_pass_kernel(const key_t* src, key_t* dst, idx_t N, idx_t L) {
    int pairId = blockIdx.x;                        // merging pair number
    idx_t a0 = (idx_t)pairId * (2 * L);
    if (a0 >= N) return;
    idx_t b0 = a0 + L;

    idx_t a_len = 0, b_len = 0;
    if (a0 < N) a_len = min(L, N - a0);
    if (b0 < N) b_len = min(L, max<idx_t>(0, N - b0));
    idx_t total = a_len + b_len;
    if (total == 0) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Compute thread's output range (balanced by element count)
    idx_t startOut = (total * (idx_t)tid) / nthreads;
    idx_t endOut   = (total * (idx_t)(tid + 1)) / nthreads;
    if (startOut >= endOut) return; // nothing to do

    // We need to find ia in A such that ia in [max(0,startOut-b_len), min(a_len, startOut)] and
    // ia + ib = startOut (ib = startOut - ia), and that arrA[ia-1] <= arrB[ib] and arrB[ib-1] <= arrA[ia]
    // We'll do binary search on ia in that range using comparisons of boundary values.

    auto find_split = [&](idx_t outPos) -> idx_t {
        idx_t loA = max<idx_t>(0, outPos - b_len);
        idx_t hiA = min<idx_t>(a_len, outPos);
        idx_t l = loA, r = hiA;
        while (l < r) {
            idx_t m = (l + r) >> 1;
            // m is candidate ia
            idx_t ib = outPos - m;
            // compare A[m] and B[ib-1] (be careful with bounds)
            key_t a_val = (m < a_len) ? src[a0 + m] : std::numeric_limits<key_t>::max();
            key_t b_val = (ib - 1 >= 0 && ib - 1 < b_len) ? src[b0 + (ib - 1)] : std::numeric_limits<key_t>::min();
            // if b_val <= a_val -> we can move left (keep fewer A)
            if (b_val <= a_val) r = m;
            else l = m + 1;
        }
        return l; // ia
    };

    idx_t ia = find_split(startOut);
    idx_t ib = startOut - ia;
    idx_t ia2 = find_split(endOut);
    idx_t ib2 = endOut - ia2;

    idx_t pa = a0 + ia;
    idx_t pb = b0 + ib;
    idx_t pa_end = a0 + ia2;
    idx_t pb_end = b0 + ib2;

    idx_t outPos = a0 + startOut;
    // merge sequentially the small chunk assigned to this thread
    while (pa < pa_end && pb < pb_end) {
        key_t aval = src[pa];
        key_t bval = src[pb];
        if (aval <= bval) {
            dst[outPos++] = aval;
            ++pa;
        } else {
            dst[outPos++] = bval;
            ++pb;
        }
    }
    while (pa < pa_end) { dst[outPos++] = src[pa++]; }
    while (pb < pb_end) { dst[outPos++] = src[pb++]; }
}

// -------------------- Host side: utilities --------------------

void gpu_merge_sort(key_t* d_src, key_t* d_dst, idx_t N) {
    // Stage 0: local block sorts
    int numBlocksLocal = (int)iDivUp((int)N, RUN);
    size_t shmemBytes = sizeof(key_t) * RUN;
    // Launch block bitonic sort: launch each block with RUN threads (1 thread per element)
    // Note: RUN should be <= maximum threads per block (typically 1024).
    if (RUN > 1024) {
        fprintf(stderr, "RUN too large for typical GPUs. Reduce RUN.\n");
        std::exit(EXIT_FAILURE);
    }
    dim3 blockDimLocal(RUN);
    dim3 gridDimLocal(numBlocksLocal);
    block_bitonic_sort_kernel<<<gridDimLocal, blockDimLocal, shmemBytes>>>(d_src, N);
    checkCuda(cudaGetLastError(), "block_bitonic_sort_kernel launch");

    // Iterative merge passes
    key_t* ping = d_src;
    key_t* pong = d_dst;
    idx_t L = RUN;
    cudaStream_t stream = 0;

    while (L < N) {
        idx_t pairs = iDivUp((int)N, (int)(2 * L));
        dim3 gdim((int)pairs);
        dim3 bdim(MERGE_TPB);

        merge_pass_kernel<<<gdim, bdim, 0, stream>>>(ping, pong, N, L);
        checkCuda(cudaGetLastError(), "merge_pass_kernel launch");

        // swap ping/pong
        std::swap(ping, pong);
        L *= 2;
    }

    // After the loop, sorted data is in 'ping' pointer. If that's not d_src (original target), we may need to copy or swap outside.
    // Caller should ensure which buffer holds the final result.
}

// -------------------- Main (host) --------------------

int main(int argc, char** argv) {
    idx_t N = 1 << 20; // default 1M elements
    if (argc > 1) {
        long long argN = atoll(argv[1]);
        if (argN > 0) N = (idx_t)argN;
    }

    printf("GPU hybrid merge sort: N = %lld, RUN = %d, MERGE_TPB = %d\n", (long long)N, RUN, MERGE_TPB);

    // Allocate pinned host memory for faster copies
    key_t* h_in = nullptr;
    checkCuda(cudaMallocHost((void**)&h_in, sizeof(key_t) * (size_t)N), "cudaMallocHost h_in");

    // Fill with random data
    std::mt19937_64 rng(123456789ULL);
    std::uniform_int_distribution<key_t> dist(std::numeric_limits<key_t>::min() / 4, std::numeric_limits<key_t>::max() / 4);
    for (idx_t i = 0; i < N; ++i) h_in[i] = dist(rng);

    // Allocate device buffers (ping/pong)
    key_t *d_buf0 = nullptr, *d_buf1 = nullptr;
    checkCuda(cudaMalloc((void**)&d_buf0, sizeof(key_t) * (size_t)N), "cudaMalloc d_buf0");
    checkCuda(cudaMalloc((void**)&d_buf1, sizeof(key_t) * (size_t)N), "cudaMalloc d_buf1");

    // Copy input to device (d_buf0)
    checkCuda(cudaMemcpy(d_buf0, h_in, sizeof(key_t) * (size_t)N, cudaMemcpyHostToDevice), "H2D copy");

    // Setup events for timing
    cudaEvent_t ev_start, ev_stop;
    checkCuda(cudaEventCreate(&ev_start), "create ev_start");
    checkCuda(cudaEventCreate(&ev_stop), "create ev_stop");
    checkCuda(cudaEventRecord(ev_start));

    // Run GPU merge sort: after returns, final sorted array is in ping pointer inside function's namespace.
    // Our function swaps ping/pong internally. To know where the final result is we can rerun the same logic here:
    // To keep it simple: call gpu_merge_sort with d_buf0 as src and d_buf1 as dst, then determine parity:
    // For convenience: we'll replicate the minimal parity logic here to know where result ended.
    // But easier: call gpu_merge_sort and then recompute where final data resides:
    gpu_merge_sort(d_buf0, d_buf1, N);

    // To determine final buffer: count number of merge passes: passes = ceil(log2(N/RUN))+1 maybe
    // Simpler: perform the same loop logic to figure final ping pointer:
    idx_t L = RUN;
    key_t* ping = d_buf0;
    key_t* pong = d_buf1;
    while (L < N) {
        std::swap(ping, pong);
        L *= 2;
    }
    // After the loop, 'ping' points to the buffer with sorted data.
    checkCuda(cudaEventRecord(ev_stop));
    checkCuda(cudaEventSynchronize(ev_stop));
    float elapsedMs = 0;
    checkCuda(cudaEventElapsedTime(&elapsedMs, ev_start, ev_stop));
    printf("GPU sort time (including device-side launches only): %.3f ms\n", elapsedMs);

    // Copy back to host for verification
    key_t* h_out = (key_t*)malloc(sizeof(key_t) * (size_t)N);
    assert(h_out);
    checkCuda(cudaMemcpy(h_out, ping, sizeof(key_t) * (size_t)N, cudaMemcpyDeviceToHost), "D2H copy");

    // Verify correctness vs std::sort on host (make a copy of input)
    std::vector<key_t> h_ref;
    h_ref.resize((size_t)N);
    for (idx_t i = 0; i < N; ++i) h_ref[i] = h_in[i];
    std::sort(h_ref.begin(), h_ref.end());

    bool ok = true;
    for (idx_t i = 0; i < N; ++i) {
        if (h_ref[(size_t)i] != h_out[(size_t)i]) {
            printf("Mismatch at %lld: expected %d, got %d\n", (long long)i, h_ref[(size_t)i], h_out[(size_t)i]);
            ok = false;
            break;
        }
    }
    printf("Verification: %s\n", ok ? "OK" : "FAILED");

    // Cleanup
    free(h_out);
    checkCuda(cudaFree(d_buf0));
    checkCuda(cudaFree(d_buf1));
    checkCuda(cudaFreeHost(h_in));
    checkCuda(cudaEventDestroy(ev_start));
    checkCuda(cudaEventDestroy(ev_stop));

    return ok ? 0 : 2;
}
