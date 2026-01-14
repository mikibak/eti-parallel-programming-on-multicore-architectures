#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void errorexit(const char *s) {
    printf("\n%s\n",s);	
    exit(EXIT_FAILURE);	 	
}


// CUDA kernel launcher: returns index in chunk or -1 if not found
int cuda_search_in_chunk(const int *array, int chunk_size, int target);


int main(int argc, char **argv) {
    const int N = 10000000; // 10 million
    const int NUM_THREADS = 4; // or use omp_get_max_threads()
    int *array = (int*)malloc(N * sizeof(int));
    if (!array) errorexit("Failed to allocate array");

    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        array[i] = rand();
    }

    int target = array[rand() % N]; // Pick a value that is guaranteed to exist
    printf("Searching for value %d in array of size %d\n", target, N);

    int found_index = -1;

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        int chunk_size = N / NUM_THREADS;
        int start = tid * chunk_size;
        int end = (tid == NUM_THREADS - 1) ? N : start + chunk_size;
        int local_size = end - start;

        int local_index = cuda_search_in_chunk(array + start, local_size, target);
        if (local_index != -1) {
            int global_index = start + local_index;
            #pragma omp critical
            {
                if (found_index == -1 || global_index < found_index) {
                    found_index = global_index;
                }
            }
        }
    }

    if (found_index != -1) {
        printf("Found value %d at index %d\n", target, found_index);
    } else {
        printf("Value %d not found in array.\n", target);
    }

    free(array);
    return 0;
}