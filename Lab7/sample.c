#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void errorexit(const char *s) {
    printf("\n%s\n",s);	
    exit(EXIT_FAILURE);	 	
}

int launchcomputekernel(long number);

int main(int argc,char **argv) {

  printf("Begin on host application... \n");

  //Parallel generate number using thread id and current time
  #pragma omp parallel 
  {
    int threadid = omp_get_thread_num();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    
    long number = (tv.tv_sec) * threadid -  (tv.tv_usec);
    
    if(number<0) {
      number = number * (-1);
    
    }

    printf("Thread id %d found number %ld \n",threadid, number);
    //call CUDA with generated number as argument
    if(launchcomputekernel(number)) {
      printf("Threadid %d number is prime \n", threadid);
    } else {
      printf("Threadid %d number is not prime \n", threadid);
    }
  }

  printf("End application - host");
}