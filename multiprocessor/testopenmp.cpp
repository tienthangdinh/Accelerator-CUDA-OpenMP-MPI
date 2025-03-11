#include <iostream>
#include <omp.h>

void function1() {
    int x = 0;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        if (thread_id == 0) {
            x = 100;  // Thread 0 updates x
        }

        #pragma omp barrier // Ensures all threads wait before reading x

        std::cout << "Thread " << thread_id << " reads x = " << x << std::endl;
    }
}

void function0() {
    int sum = 0;
    #pragma omp parallel  //have to explicitly manual workload distribute
    for (int i = 0; i < 100; i++) {
        #pragma omp critical //or try single
        sum = sum + 1;
    }
    std::cout << sum << std::endl;

    sum = 0;
    #pragma omp parallel for //automatic workload distribution (as long as can be executed independently)
    for (int i = 0; i < 100; i++) {
        sum = sum + 1;
    }
    std::cout << sum << std::endl;

    int x = 0;  // Shared variable
    #pragma omp parallel private(x) //or try share(x)
    for (int i = 0; i < 100; i++) {
        x=x+1;
    }
    std::cout << x << std::endl;
}

void testreduction() {
    int sum = 0;
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            sum+=1;
        }
    }
    std::cout << sum << std::endl;
}
//task and section are pragma that parallel and mapped to thread, but task is dynamic assigned to threads (good for unbalanced tasks)
int main() {
    
    testreduction();
    
    return 0;
}
