#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int threadnum=4; 

int main(int argc,char **argv) {
    omp_set_num_threads(threadnum);

    int i, sum = 0;
    // Create a parallel region - each threads execute below instruction
    #pragma omp parallel shared(sum)
    {
	//distribute the iterations of the loop among the threads
        #pragma omp for
        for (i = 0; i < 1000; i++) {
	    //synchronization using critical section- critical section will execute only one thread at given time
            #pragma omp critical
            sum += i;
        }
    }
    printf("The sum of numbers from 0 to 999 is %d.\n", sum);


    // Create a parallel region - each threads execute below instruction
    #pragma omp parallel shared(sum)
    {
	//distribute the iterations of the loop among the threads
        #pragma omp for
        for (i = 0; i < 1000; i++) {
	    //synchronization through atomic operation
            #pragma omp atomic update
            sum += i;
        }
    }
    printf("The sum of numbers from 0 to 999 is %d.\n", sum);

    //synchronization using lock
    omp_lock_t lock;
    omp_init_lock(&lock);
    // Create a parallel region - each threads execute below instruction
    #pragma omp parallel shared(sum)
    {
	//distribute the iterations of the loop among the threads
        #pragma omp for
        for (i = 0; i < 1000; i++) {
	    //acquire the lock
            omp_set_lock(&lock);
            sum += i;
	    // release the lock
            omp_unset_lock(&lock);
        }
    }
    omp_destroy_lock(&lock);
    printf("The sum of numbers from 0 to 999 is %d.\n", sum);
    
    //synchronization using flush-  consistent view of the used memory 
    // Create a parallel region - each threads execute below instruction
    #pragma omp parallel shared(sum)
    {
	//distribute the iterations of the loop among the threads
        #pragma omp for
        for (i = 0; i < 1000; i++) {
            sum += i;
	    //ensures that all threads have a consistent view of sum before proceeding to the next statement
            #pragma omp flush(sum)
        }
    }
    printf("The sum of numbers from 0 to 999 is %d.\n", sum);
    
}