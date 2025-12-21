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
    int n = 1000000;
    int *arr = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
        arr[i] = rand();

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        merge_sort(arr, 0, n - 1);
    }

    double end = omp_get_wtime();

    printf("Time: %f seconds\n", end - start);
    printf("Sorted: %s\n", is_sorted(arr, n) ? "YES" : "NO");

    free(arr);
    return 0;
}
