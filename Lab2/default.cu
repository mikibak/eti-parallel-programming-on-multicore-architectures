/*
CUDA - generation of array of N elements and calculates even and odd numbers occurence - no streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 0
__host__
void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

__host__ 
void generate(int *matrix, int matrixSize) {
	srand(time(NULL));
	for(int i=0; i<matrixSize; i++) {
		matrix[i] = rand()%1000;
	}
}

__global__ 
void calculation(int *matrix, int *even, int *odd, int matrixSize) {
		int my_index=blockIdx.x*blockDim.x+threadIdx.x;
		if(my_index < matrixSize) {
			if(matrix[my_index] % 2) {
				atomicAdd(odd, 1);
			} else {
				atomicAdd(even, 1);
			}
		} 
}

int main(int argc,char **argv) {

	//define array size and allocate memory on host
	int matrixSize=10000000;
	int *hMatrix=(int*)malloc(matrixSize*sizeof(int));
	cudaEvent_t start, stop;
    float milliseconds = 0;

	//generate random numbers
	generate(hMatrix, matrixSize);

	if(DEBUG) {
		printf("Generated numbers: \n");
		for(int i=0; i<matrixSize; i++) {
			printf("%d ", hMatrix[i]);
		}
		printf("\n");
	}

	//allocate memory for odd and even numbers counters - host
	int *hEven=(int*)malloc(sizeof(int));
	int *hOdd=(int*)malloc(sizeof(int));

	//allocate memory for odd and even numbers counters and array - device
	int *dEven=NULL;
	int *dOdd=NULL;
	int *dMatrix=NULL;

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	if (cudaSuccess!=cudaMalloc((void **)&dEven,sizeof(int)))
			errorexit("Error allocating memory on the GPU");

	if (cudaSuccess!=cudaMalloc((void **)&dOdd,sizeof(int)))
			errorexit("Error allocating memory on the GPU");
	
	if (cudaSuccess!=cudaMalloc((void **)&dMatrix,matrixSize*sizeof(int)))
			errorexit("Error allocating memory on the GPU");

	//initialize allocated counters with 0
	if (cudaSuccess!=cudaMemset(dEven,0, sizeof(int)))
			errorexit("Error initializing memory on the GPU");

	if(cudaSuccess!=cudaMemset(dOdd,0, sizeof(int)))
			errorexit("Error initializing memory on the GPU");

	//copy array to device
	if (cudaSuccess!=cudaMemcpy(dMatrix,hMatrix,matrixSize*sizeof(int),cudaMemcpyHostToDevice))
		 errorexit("Error copying input data to device");

	int threadsinblock=1024;
	int blocksingrid=1+((matrixSize-1)/threadsinblock); 

	//run kernel on GPU 
	calculation<<<blocksingrid, threadsinblock>>>(dMatrix, dEven, dOdd, matrixSize);

	//copy results from GPU
	if (cudaSuccess!=cudaMemcpy(hEven, dEven, sizeof(int),cudaMemcpyDeviceToHost))
		 errorexit("Error copying results");

	if (cudaSuccess!=cudaMemcpy(hOdd, dOdd, sizeof(int),cudaMemcpyDeviceToHost))
		 errorexit("Error copying results");

	cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Found %d even numbers \n", *hEven);
	printf("Found %d odd numbers \n", *hOdd);
	printf("Found %d total numbers \n", *hEven + *hOdd);
	printf("Kernel execution time: %.3f ms\n", milliseconds);
	//Free memory
	free(hOdd);
	free(hEven);
	free(hMatrix);
		
	if (cudaSuccess!=cudaFree(dEven))
		errorexit("Error when deallocating space on the GPU");
	if (cudaSuccess!=cudaFree(dOdd))
		errorexit("Error when deallocating space on the GPU");
	if (cudaSuccess!=cudaFree(dMatrix))
		errorexit("Error when deallocating space on the GPU");
}
