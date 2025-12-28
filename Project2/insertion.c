#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int is_sorted(int *arr, int n)
{
    for (int i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1])
            return 0;
    return 1;
}



#define MIN_PARALLEL_SIZE 10000

void merge(int* arr, int l, int m, int r, int* temp) {
    int i = l, j = m, k = l;
    while (i < m && j < r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i < m) temp[k++] = arr[i++];
    while (j < r) temp[k++] = arr[j++];
    for (i = l; i < r; i++) arr[i] = temp[i];
}

void mergesort_parallel(int* arr, int l, int r, int* temp) {
    if (r - l <= 32) { // insertion sort for small arrays
        for (int i = l + 1; i < r; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= l && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
        return;
    }
    int m = l + (r - l) / 2;
#pragma omp task shared(arr, temp) if(r-l > MIN_PARALLEL_SIZE)
    mergesort_parallel(arr, l, m, temp);
#pragma omp task shared(arr, temp) if(r-l > MIN_PARALLEL_SIZE)
    mergesort_parallel(arr, m, r, temp);
#pragma omp taskwait
    merge(arr, l, m, r, temp);
}

void sort(int* arr, int n) {
    int* temp = (int*)malloc(n * sizeof(int));
#pragma omp parallel
    {
#pragma omp single nowait
        mergesort_parallel(arr, 0, n, temp);
    }
    free(temp);
}


int main(int argc, char **argv)
{
    int n = 0;
    printf("Enter the array size:\n");
    int scan_res = scanf("%d", &n);
    int *arr = (int *)malloc(n * sizeof(int));
    int *backup = (int *)malloc(n * sizeof(int));

    double times[10];

    for (int r = 0; r < 10; r++)
    {
        for (int i = 0; i < n; i++) backup[i] = rand();
        for (int i = 0; i < n; i++) arr[i] = backup[i];

        double start = omp_get_wtime();

        sort(arr, n);

        double end = omp_get_wtime();
        times[r] = end - start;

        if (!is_sorted(arr, n))
        {
            printf("ERROR: Array NOT sorted!\n");
            free(arr);
            free(backup);
            return 1;
        }
        printf("Sorted array for run %d\n", r + 1);
    }

    double sum = 0, tmin = times[0], tmax = times[0];
    for (int i = 0; i < 10; i++)
    {
        sum += times[i];
        if (times[i] < tmin) tmin = times[i];
        if (times[i] > tmax) tmax = times[i];
    }


    double mean = sum / 10.0 * 1000.0; // ms
    double uncertainty = (tmax - tmin) / 2.0 * 1000.0; // ms

    printf("\nAverage time over 10 runs: %.3f ms\n", mean);
    printf("Uncertainty (Tmax-Tmin)/2: ± %.3f ms\n", uncertainty);

    free(arr);
    free(backup);
    return 0;
}
