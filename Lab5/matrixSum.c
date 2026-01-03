#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000
#define M 1000

int main() {
    int i, j;
    int threadnum = 4;
    omp_set_num_threads(threadnum);

    // Allocate matrices
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    for (i = 0; i < N; i++) {
        A[i] = (int *)malloc(M * sizeof(int));
        B[i] = (int *)malloc(M * sizeof(int));
        C[i] = (int *)malloc(M * sizeof(int));
    }

    // Initialize matrices A and B with random values
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    // Parallel element-wise sum using OpenMP
    #pragma omp parallel for private(j) shared(A, B, C)
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    // Print a small part of the result for verification
    printf("C[0][0] = %d\n", C[0][0]);
    printf("C[N-1][M-1] = %d\n", C[N-1][M-1]);

    // Free memory
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
