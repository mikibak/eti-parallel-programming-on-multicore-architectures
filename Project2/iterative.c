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


void merge(int* arr, int left, int mid, int right, int* temp) {
    int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i < mid) temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];
}


void sort(int* arr, int n) {
    int* temp = (int*)malloc(n * sizeof(int));
    int* src = arr;
    int* dst = temp;
    int width;
    for (width = 1; width < n; width *= 2) {
        // merge one level
        int num_merges = (n + 2 * width - 1) / (2 * width);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_merges; i++) {
            int left = i * 2 * width;
            int mid = left + width < n ? left + width : n;
            int right = left + 2 * width < n ? left + 2 * width : n;
            merge(src, left, mid, right, dst);
        }
        // swap src and dst
        int* tmp = src;
        src = dst;
        dst = tmp;
    }
    // If the result is in temp, swap the pointers
    if (src != arr) {
        int* tmp = src;
        src = dst;
        dst = tmp;
    }
    free(dst);
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
