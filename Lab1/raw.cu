/*
CUDA - prepare the histogram of N numbers in range of <a;b> where a and b should be integers
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeHistogram(int *data, int *histogram, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
    
	srand(time(NULL));

    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A +1);
    }

}

int main(int argc,char **argv) {

    int threadsinblock=1024;
    int blocksingrid;

    int N;
    int A=0;
    int B=100;
    
 	cudaEvent_t start, stop;
    float milliseconds = 0;
    
    printf("Enter number of elements: \n");
    scanf("%d", &N);

	int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	generateRandomNumbers(randomNumbers, N,A,B);

	blocksingrid = ceil((double)N/threadsinblock);

	printf("The kernel will run with: %d blocks\n", blocksingrid);

	int *resultArrayHost, *resultArrayDevice, *randomNumbersDevice;

	resultArrayHost = (int *)calloc((B-A), sizeof(int));

	if (resultArrayHost == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&resultArrayDevice, (B-A+1) * sizeof(int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(resultArrayDevice, 0, (B-A+1) * sizeof(int));

    computeHistogram<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, resultArrayDevice, N);

    cudaMemcpy(resultArrayHost, resultArrayDevice, (B-A+1) * sizeof(int), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);


    printf("Histogram:\n");
    int assertion = 0;
    for (int i = 0; i <= B-A; i++) {
        printf("%d occures %d\n", i, resultArrayHost[i]);
        assertion += resultArrayHost[i];
    }
    printf("Total numbers=%d \n",assertion);


	printf("Kernel execution time: %.3f ms\n", milliseconds);

    free(randomNumbers);
    free(resultArrayHost);
    cudaFree(randomNumbersDevice);
    cudaFree(resultArrayDevice);

    return 0;

}
