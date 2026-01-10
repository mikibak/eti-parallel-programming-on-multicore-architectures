#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define TOTAL_TH 39 //sum of power of three - 3^1 + 3^2 + 3^3


void dynamic_deeper(int depth, int father) {

  double max = 0;
  int threadid, j;

//how many branches are at this level - maximum active threads at this level
  for(j=1;j<=depth; j++) {
    max += pow((double)3, (double)j);
  }

//if current active thread number is higher than max value 3^1 + 3^2 + 3^3 = 39 - depth of nested calls is 3 
  if(max < TOTAL_TH) {
    int myDepth, myFather;
    //parallel - according to num_threads - 3 threads will work in parallel
    #pragma omp parallel private(myFather, myDepth, threadid) num_threads(3)
    {
      myDepth = depth;
      threadid = omp_get_thread_num();
      int i =0;

      //critical to ensure that output will be correctly printed
      #pragma omp critical 
      {
        for(i;i<myDepth;i++){
          printf("\t");
        }
        printf("Depth %d in child threadid %d with father threadid %d \n",myDepth, threadid, father);
      }

      myDepth++;
      myFather = threadid;
      dynamic_deeper(myDepth, father);
    }
  } 
}

int main(int argc,char **argv) {

  omp_set_nested(1); //enables nested parallelism
  omp_set_dynamic(0); //disable dynamic setting for number of threads
  
  int threadid;
//parallel - according to num_threads - 3 threads will work in parallel
#pragma omp parallel private(threadid) num_threads(3)
{
  threadid=omp_get_thread_num();
  printf("Level 0 - threadid %d \n",threadid);
  dynamic_deeper(1, threadid);
}

}
