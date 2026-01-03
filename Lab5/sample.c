#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int threadnum=4; 

int main(int argc,char **argv) {

  omp_set_num_threads(threadnum);
  omp_lock_t writelock;

  omp_init_lock(&writelock);
  
  int value=0;

printf("Value at the begining is %d \n",value);


//parallel - each threads execute below instruction
#pragma omp parallel shared(value) 
{
  //synchronization through atomic operation
  #pragma omp atomic update
  value++;
}

 printf("Value after parallel (4 thread) is %d \n",value);

//parallel - each threads execute below instruction - synchronization using lock - 
#pragma omp parallel shared(value) 
{
  omp_set_lock(&writelock);
  value++;
  omp_unset_lock(&writelock);
}

 printf("Value after parallel (4 thread) is %d \n",value);
//parallel i critical - each threads execute below instruction, but critical section will execute only one thread at given time
 #pragma omp parallel shared(value) 
{
  #pragma omp critical
  {
    value++;
  }
}

}
