/*
CUDA - generation and sum of arithmetic progression build of 10240000 elements a1=0 r=1 with Unified Memory
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef int idx_t;

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

    int *is_prime;
    cudaCheckError(cudaMallocManaged(&is_prime, sizeof(int)));
    *is_prime = 1;

    check_prime<<<blocksingrid, threadsinblock>>>(n, is_prime);
    cudaCheckError(cudaGetLastError());

    cudaDeviceSynchronize();

    if (*is_prime)
        printf("\n%d is a prime number.\n", n);
    else
        printf("\n%d is NOT a prime number.\n", n);

    cudaCheckError(cudaFree(is_prime));
}
