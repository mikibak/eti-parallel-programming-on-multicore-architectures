/*
Copyright 2017, Paweł Czarnul pawelczarnul@pawelczarnul.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_RECURSION_DEPTH 18
#define THREADS_PER_BLOCK 512
#define BLOCKS 256

__host__
void errorexit(const char *s) {
    printf("\n%s\n",s); 
    exit(EXIT_FAILURE);   
}

__device__ 
double f(double x) {
  return 1.0/(1.0+x);
}

__device__
double integratesimple(double start,double end,double step) {
    double result;
    long counter;
    long countermax;
    double arg;

    result=0;
    double length=(end-start)/THREADS_PER_BLOCK;
    countermax=((end-start)/step)/THREADS_PER_BLOCK;
    arg=start+length*threadIdx.x;
    for(counter=0;counter<countermax;counter++) {
      result+=step*(f(arg)+f(arg+step))/2;
      arg+=step;
    }  
    return result;
}

__global__ 
void integratedp(double *result,double starti,double endi,double step,int depth) {
    extern __shared__ double sresults[]; // use dynamically allocated shared memory
    long counter;
    cudaStream_t stream1,stream2;
    double width=(endi-starti)/(double)gridDim.x;
    double start=starti+blockIdx.x*width;
    double end=start+width;

    sresults[threadIdx.x]=integratesimple(start,end,step/2);    
    __syncthreads();
    for(counter=THREADS_PER_BLOCK/2;counter>0;counter/=2) {
      if (threadIdx.x<counter)
        sresults[threadIdx.x]+=sresults[threadIdx.x+counter];
      __syncthreads();      
    }
  
    if (threadIdx.x==0)
      *(result+blockIdx.x)=sresults[0];

    if (depth==MAX_RECURSION_DEPTH)
       return;

    // now repeat it with a smaller resolution
    sresults[threadIdx.x]=integratesimple(start,end,step);    
    __syncthreads();
    for(counter=THREADS_PER_BLOCK/2;counter>0;counter/=2) {
      if (threadIdx.x<counter)
        sresults[threadIdx.x]+=sresults[threadIdx.x+counter];
      __syncthreads();      
    }

    // now thread 0 checks whether to go deeper
    if (threadIdx.x==0) {
      if (fabs(*(result+blockIdx.x)-sresults[0])>0.0000001) { // need to invoke kernel recursively
    cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
          integratedp<<<1,THREADS_PER_BLOCK,THREADS_PER_BLOCK*sizeof(double),stream1>>>(result+blockIdx.x,start,(start+end)/2,step/2,depth+1);

    cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);
          integratedp<<<1,THREADS_PER_BLOCK,THREADS_PER_BLOCK*sizeof(double),stream2>>>(result+blockIdx.x+(long)BLOCKS*(1<<(MAX_RECURSION_DEPTH-depth-1)),(start+end)/2,end,step/2,depth+1);
      } else *(result+blockIdx.x+(long)BLOCKS*(1<<(MAX_RECURSION_DEPTH-depth-1)))=0;
    }  
    __syncthreads();    
    if (threadIdx.x==0) {
      cudaStreamDestroy(stream1); 
      cudaStreamDestroy(stream2);
    } 
    __syncthreads();
    if (threadIdx.x==0)
      *(result+blockIdx.x)+=*(result+blockIdx.x+(long)BLOCKS*(1<<(MAX_RECURSION_DEPTH-depth-1)));
}



int main(int argc,char **argv) {
    double step;
    double finalresult=0;
    double start,end; 
    long size=sizeof(double)*(long)(1<<MAX_RECURSION_DEPTH)*BLOCKS;
    double *hresults;
    double *dresults;
    cudaStream_t stream;
    long i;

    if (cudaSuccess!=cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth,MAX_RECURSION_DEPTH))
      errorexit("Error setting depth limit");

//cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

    hresults=(double *)malloc(size);
    if (!(hresults)) errorexit("Error allocating memory on the host");  

    if (cudaSuccess!=cudaMalloc((void **)&dresults,size))
      errorexit("Error allocating memory on the GPU");

    if (cudaSuccess!=cudaStreamCreate(&stream))
         errorexit("Error creating stream");

    // define input data
    start=1;
    end=1000000;
    step=(end-start)/((long)4000000*THREADS_PER_BLOCK);
    // invoke the parent kernel
    integratedp<<<BLOCKS,THREADS_PER_BLOCK,THREADS_PER_BLOCK*sizeof(double),stream>>>(dresults,start,end,step,0);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch in stream");

    // copy the result back to the host
    if (cudaSuccess!=cudaMemcpyAsync(hresults,dresults,BLOCKS*sizeof(double),cudaMemcpyDeviceToHost,stream))
      errorexit("Error copying results");
    cudaStreamSynchronize(stream);

    for(i=1;i<BLOCKS;i++)
      hresults[0]+=hresults[i];
    finalresult=hresults[0];
    printf("\nThe final result is %f\n",finalresult);

    // release resources
    free(hresults);
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");
}
