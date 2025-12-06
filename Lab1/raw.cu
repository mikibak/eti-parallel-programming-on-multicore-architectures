#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeAverage(long long unsigned int *data, long long unsigned int *sum, long long unsigned int N) {
    long long unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(sum, data[idx]);
    }
}

void generateRandomNumbers(long long unsigned int *arr, long long unsigned int N, long long unsigned int A, long long unsigned int B) {
    
	srand(time(NULL));

    for (long long unsigned int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A +1);
    }

}

int main(int argc,char **argv) {

    long long unsigned int threadsinblock=1024;
    long long unsigned int blocksingrid;

    long long unsigned int N;
    long long unsigned int A=0;
    long long unsigned int B=100;
    
 	cudaEvent_t start, stop;
    float milliseconds = 0;
    
    printf("Enter number of elements: \n");
    scanf("%lld", &N);

	long long unsigned int *randomNumbers = (long long unsigned int *)malloc(N * sizeof(long long unsigned int));
    if (randomNumbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	generateRandomNumbers(randomNumbers, N,A,B);

	blocksingrid = ceil((double)N/threadsinblock);

	printf("The kernel will run with: %lld blocks\n", blocksingrid);

    long long unsigned int *sumHost, *sumDevice, *randomNumbersDevice;

    sumHost = (long long unsigned int *)calloc(1, sizeof(long long unsigned int));
    if (sumHost == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(long long unsigned int));
    cudaMalloc((void **)&sumDevice, sizeof(long long unsigned int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(long long unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(sumDevice, 0, sizeof(long long unsigned int));

    computeAverage<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, sumDevice, N);

    cudaMemcpy(sumHost, sumDevice, sizeof(long long unsigned int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    double average = (double) *sumHost / N;

    printf("Average value = %.3f\n", average);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    free(randomNumbers);
    free(sumHost);
    cudaFree(randomNumbersDevice);
    cudaFree(sumDevice);

    return 0;

}
