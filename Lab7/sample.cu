
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__host__
void errorexit(const char *s) {
  printf("\n%s",s); 
  exit(EXIT_FAILURE);   
}

// CUDA kernel: search for target in array, return index via result (first found in block)
__global__ void cuda_search_kernel(const int *array, int n, int target, int *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && array[idx] == target) {
    atomicMin(result, idx); // store the lowest index found
  }
}

// Host function callable from C/OMP: returns index in chunk or -1 if not found
extern "C" int cuda_search_in_chunk(const int *array, int chunk_size, int target) {
  int *d_array = NULL;
  int *d_result = NULL;
  int h_result;
  size_t bytes = chunk_size * sizeof(int);
  cudaError_t err;

  // Allocate device memory
  err = cudaMalloc((void**)&d_array, bytes);
  if (err != cudaSuccess) errorexit("cudaMalloc d_array failed");
  err = cudaMalloc((void**)&d_result, sizeof(int));
  if (err != cudaSuccess) errorexit("cudaMalloc d_result failed");

  // Copy data to device
  err = cudaMemcpy(d_array, array, bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) errorexit("cudaMemcpy to device failed");

  // Set result to max int
  int maxint = 0x7fffffff;
  err = cudaMemcpy(d_result, &maxint, sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) errorexit("cudaMemcpy set result failed");

  // Launch kernel
  int threads = 256;
  int blocks = (chunk_size + threads - 1) / threads;
  cuda_search_kernel<<<blocks, threads>>>(d_array, chunk_size, target, d_result);
  err = cudaGetLastError();
  if (err != cudaSuccess) errorexit("Kernel launch failed");

  // Copy result back
  err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) errorexit("cudaMemcpy result to host failed");

  cudaFree(d_array);
  cudaFree(d_result);

  if (h_result == 0x7fffffff) return -1;
  return h_result;
}

__global__ 
void calculate(int *result, long test) {
    __shared__ long sresults[1024];  
    int counter;
    long my_index=blockIdx.x*blockDim.x+threadIdx.x;
    if(my_index < test && my_index != 1){
      if(test%my_index == 0) {
        sresults[threadIdx.x] = 1;
      } else {
        sresults[threadIdx.x]=0;
      }
    } else {
      sresults[threadIdx.x]=0;
    }
    

    __syncthreads();
   for(counter=512;counter>0;counter/=2) {
      if (threadIdx.x<counter)
        sresults[threadIdx.x]=(sresults[threadIdx.x]+sresults[threadIdx.x+counter]);
      __syncthreads();      
    }

    if (threadIdx.x==0) {
      result[blockIdx.x]=sresults[0];
    }
}

//function called from OpenMP code
extern "C" int launchcomputekernel(long test) {

    int threadsinblock=1024;

    //calculate number of blocks
    long blocksingrid=test/threadsinblock;
    if(test%threadsinblock >0 ) {
      blocksingrid++;
    }

    //allocate memory on host
   int *hresults=(int*)malloc(blocksingrid*sizeof(int));
    if (!hresults) errorexit("Error allocating memory on the host");  

    //allocate memory on device
    int *dresults=NULL;
    if (cudaSuccess!=cudaMalloc((void **)&dresults,blocksingrid*sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    //begin GPU calculation - if number is prime
    calculate<<<blocksingrid,threadsinblock>>>(dresults, test);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //get partial results
    if (cudaSuccess!=cudaMemcpy(hresults,dresults,blocksingrid*sizeof(int),cudaMemcpyDeviceToHost))
       errorexit("Error copying results");

     //check overall results if number is prime
    int pierwsza = 1;
    for(int i=0;i<blocksingrid;i++) {
      if(hresults[i] != 0) {
        pierwsza = 0;
        break;
      }
    }
    //free momery on host
    free(hresults);

    //free memory on device
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");

    //return result - if 1 - prime number
    return pierwsza;
}