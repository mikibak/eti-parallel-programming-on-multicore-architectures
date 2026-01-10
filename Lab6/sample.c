#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>


void quicksort_parallel(int *arr, int left, int right, int depth) {
  if (left < right) {
    int i = left, j = right;
    int pivot = arr[(left + right) / 2];
    while (i <= j) {
      while (arr[i] < pivot) i++;
      while (arr[j] > pivot) j--;
      if (i <= j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
        i++;
        j--;
      }
    }
    // Limit nesting depth to avoid oversubscription
    if (depth < 3) {
      #pragma omp parallel sections
      {
        #pragma omp section
        quicksort_parallel(arr, left, j, depth + 1);
        #pragma omp section
        quicksort_parallel(arr, i, right, depth + 1);
      }
    } else {
      quicksort_parallel(arr, left, j, depth + 1);
      quicksort_parallel(arr, i, right, depth + 1);
    }
  }
}

int main(int argc, char **argv) {
  omp_set_nested(1);
  omp_set_dynamic(0);

  int n = 20;
  int arr[20];

  #pragma omp parallel
  {
    #pragma omp single
    quicksort_parallel(arr, 0, n - 1, 0);
  }

  // Validate if array is sorted
  int sorted = 1;
  for (int i = 1; i < n; i++) {
    if (arr[i-1] > arr[i]) {
      sorted = 0;
      break;
    }
  }
  if (sorted) {
    printf("Array is sorted correctly.\n");
  } else {
    printf("Array is NOT sorted!\n");
  }
  return 0;
}
