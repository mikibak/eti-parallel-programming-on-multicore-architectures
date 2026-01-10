#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#define N 1000000
#define NUM_BINS 10
#define MAX_VALUE 100

int main() {
    int *array = (int*)malloc(N * sizeof(int));
    int histogram[NUM_BINS] = {0};
    int i;

    // Generate random data
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        array[i] = rand() % MAX_VALUE;
    }

    double start_time = omp_get_wtime();

    // use atomic to prevent race conditions
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        int bin = array[i] * NUM_BINS / MAX_VALUE;
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;
        #pragma omp atomic
        histogram[bin]++;
    }

    double end_time = omp_get_wtime();
    printf("\nHistogram computation time: %f seconds\n", end_time - start_time);

    // Print histogram
    int total = 0;
    for (i = 0; i < NUM_BINS; i++) {
        printf("Bin %d: %d\n", i, histogram[i]);
        total += histogram[i];
    }

    // Assert that the sum of histogram bins equals N
    assert(total == N);
    printf("\nAssertion passed: total (%d) == N (%d)\n", total, N);

    free(array);
    return 0;
}
