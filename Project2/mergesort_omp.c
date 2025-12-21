#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Merge two sorted subarrays:
// arr[left..mid] and arr[mid+1..right]
void merge(int *arr, int left, int mid, int right)
{
    int i = left;
    int j = mid + 1;
    int k = 0;

    int size = right - left + 1;
    int *temp = (int *)malloc(size * sizeof(int));

    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];

    while (j <= right)
        temp[k++] = arr[j++];

    for (i = 0; i < size; i++)
        arr[left + i] = temp[i];

    free(temp);
}


void merge_sort(int *arr, int left, int right)
{
    if (left >= right)
        return;

    int mid = (left + right) / 2;

    #pragma omp task
    merge_sort(arr, left, mid);

    #pragma omp task
    merge_sort(arr, mid + 1, right);

    #pragma omp taskwait
    merge(arr, left, mid, right);
}


int is_sorted(int *arr, int n)
{
    for (int i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1])
            return 0;
    return 1;
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

        #pragma omp parallel
        {
            #pragma omp single
            merge_sort(arr, 0, n - 1);
        }

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

    double mean = sum / 10.0;
    double uncertainty = (tmax - tmin) / 2.0;

    printf("\nAverage time over 10 runs: %.6f seconds\n", mean);
    printf("Uncertainty (Tmax-Tmin)/2: ± %.6f seconds\n", uncertainty);

    free(arr);
    free(backup);
    return 0;
}
