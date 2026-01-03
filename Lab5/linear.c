/**


Generates a random matrix of size rows x columns.
 *


@param rows: The number of rows in the matrix.


@param columns: The number of columns in the matrix.


@return: A dynamically allocated 2D array representing the random matrix.
*/
int generateRandomMatrix(int rows, int columns) {
    // Allocating memory for the matrix.
    int matrix = (int) malloc(rows * sizeof(int));
    if (matrix == NULL) {  // Checking for unsuccessful memory allocation.
        printf(“Memory allocation failed.\n”);
        exit(EXIT_FAILURE);  // Exiting the program with a failure status.
    }
// Generating random numbers for each element in the matrix.
srand(time(NULL));  // Seeding the random number generator with the current time.
for (int i = 0; i < rows; i++) {
    matrix[i] = (int*) malloc(columns * sizeof(int));
    if (matrix[i] == NULL) {  // Checking for unsuccessful memory allocation.
        printf(“Memory allocation failed.\n”);
        exit(EXIT_FAILURE);  // Exiting the program with a failure status.
    }
    for (int j = 0; j < columns; j++) {
        matrix[i][j] = rand() % 100;  // Generating a random number between 0 and 99.
    }
}
return matrix;  // Returning the generated matrix.
}


/**


Sums two matrices element-wise and stores the result in a new matrix.
 *


@param matrix1: The first matrix.


@param matrix2: The second matrix.


@param rows: The number of rows in the matrices.


@param columns: The number of columns in the matrices.


@return: A dynamically allocated 2D array representing the sum of the matrices.
*/
int sumMatrices(int matrix1, int matrix2, int rows, int columns) {
    // Allocating memory for the result matrix.
    int result = (int) malloc(rows * sizeof(int));
    if (result == NULL) {  // Checking for unsuccessful memory allocation.
        printf(“Memory allocation failed.\n”);
        exit(EXIT_FAILURE);  // Exiting the program with a failure status.
    }
// Summing the elements of the matrices element-wise.
for (int i = 0; i < rows; i++) {
    result[i] = (int*) malloc(columns * sizeof(int));
    if (result[i] == NULL) {  // Checking for unsuccessful memory allocation.
        printf(“Memory allocation failed.\n”);
        exit(EXIT_FAILURE);  // Exiting the program with a failure status.
    }
    for (int j = 0; j < columns; j++) {
        result[i][j] = matrix1[i][j] + matrix2[i][j];
    }
}
return result;  // Returning the sum matrix.
}


// Usage example for generateRandomMatrix and sumMatrices
int main() {
    int rows = 3;
    int columns = 3;
// Generate two random matrices.
int** matrix1 = generateRandomMatrix(rows, columns);
int** matrix2 = generateRandomMatrix(rows, columns);

// Print the generated matrices.
printf("Matrix 1:\n");
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
        printf("%d ", matrix1[i][j]);
    }
    printf("\n");
}
printf("\n");

printf("Matrix 2:\n");
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
        printf("%d ", matrix2[i][j]);
    }
    printf("\n");
}
printf("\n");

// Sum the matrices.
int** sum = sumMatrices(matrix1, matrix2, rows, columns);

// Print the sum matrix.
printf("Sum Matrix:\n");
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
        printf("%d ", sum[i][j]);
    }
    printf("\n");
}
printf("\n");

// Free allocated memory to avoid memory leaks.
for (int i = 0; i < rows; i++) {
    free(matrix1[i]);
    free(matrix2[i]);
    free(sum[i]);
}
free(matrix1);
free(matrix2);
free(sum);

return 0;

}