/*
CUDA - check if a number is prime using parallel divisibility test (no Unified Memory)
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef int idx_t;

// Host side error checker (from dynamic_parallelism.cu)
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


__global__
void check_prime(int n, int *is_prime) {
  idx_t idx = blockIdx.x * blockDim.x + threadIdx.x + 2; // start from 2
  if (idx > n/2) return;
  if (n % idx == 0) {
    *is_prime = 0;
  }
}


int main(int argc, char **argv) {
  int n = 0;
  printf("Enter a number for prime check:\n");
  scanf("%d", &n);

  int threadsinblock = 1024;
  int max_divisor = n / 2;
  int blocksingrid = (max_divisor + threadsinblock - 1) / threadsinblock;

  int h_is_prime = 1;
  int *d_is_prime = NULL;

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  cudaCheckError(cudaMalloc((void **)&d_is_prime, sizeof(int)));
  cudaCheckError(cudaMemcpy(d_is_prime, &h_is_prime, sizeof(int), cudaMemcpyHostToDevice));

  cudaCheckError(cudaEventRecord(start, 0));
  check_prime<<<blocksingrid, threadsinblock>>>(n, d_is_prime);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaEventRecord(stop, 0));
  cudaCheckError(cudaEventSynchronize(stop));
  cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

  cudaCheckError(cudaMemcpy(&h_is_prime, d_is_prime, sizeof(int), cudaMemcpyDeviceToHost));

  if (h_is_prime)
    printf("\n%d is a prime number.\n", n);
  else
    printf("\n%d is NOT a prime number.\n", n);

  printf("Kernel execution time: %.3f ms\n", milliseconds);

  cudaCheckError(cudaFree(d_is_prime));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
