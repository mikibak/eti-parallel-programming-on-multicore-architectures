#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

__global__ void computeAverageSharedMemory(int *data, int *globalSum, int N) {
    extern __shared__ int sharedData[];

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

    if (globalId < N) {
        sharedData[threadId] = data[globalId];
    }
    __syncthreads();

    // Reduce inside block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadId < stride) {
            sharedData[threadId] += sharedData[threadId + stride];
        }
        __syncthreads();
    }

    // Thread 0 of each block updates global sum
    if (threadId == 0) {
        atomicAdd(globalSum, sharedData[0]);
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B) {
    
	srand(time(NULL));

    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }

}

int main(int argc,char **argv) {

    int threadsinblock=1024;
    int blocksingrid;

    int N;
    int A = 0;
    int B = 100;
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

    sumHost = (int *)malloc(sizeof(int));
    *sumHost = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&sumDevice, sizeof(int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(sumDevice, 0, sizeof(int));

    int sharedSize = threadsinblock * sizeof(int);

    computeAverageSharedMemory<<<blocksingrid, threadsinblock, sharedSize>>>(randomNumbersDevice, sumDevice, N);

    cudaMemcpy(sumHost, sumDevice, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    double average = (double)*sumHost / N;

    printf("Average value = %.3f\n", average);
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    free(randomNumbers);
    free(sumHost);
    cudaFree(randomNumbersDevice);
    cudaFree(sumDevice);

    return 0;

}
