#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeAverage(int *data, int *sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(sum, data[idx]);
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

    int *sumHost, *sumDevice, *randomNumbersDevice;

    sumHost = (int *)calloc(1, sizeof(int));
    if (sumHost == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&sumDevice, sizeof(int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(sumDevice, 0, sizeof(int));

    computeAverage<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, sumDevice, N);

    cudaMemcpy(sumHost, sumDevice, sizeof(int), cudaMemcpyDeviceToHost);

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
