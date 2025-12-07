/*
CUDA - dynamic parallelism sample
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 0
#define N 20

__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

__global__ 
void kernel(int depth) {
	if(depth == N) {
		printf("Max depth = %d achieved \n",N);
		return;
	}
	for(int i=0;i<depth;i++) {
		printf("-");
	}
	printf("Kernel call depth %d \n", depth);

	kernel<<<1,1>>>(depth+1);
}

int main(int argc,char **argv) {
	//run kernel on GPU 
	printf("Dynamic parallelism example\n");
	kernel<<<1, 1>>>(0);

	if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");

  	cudaDeviceSynchronize();
}
