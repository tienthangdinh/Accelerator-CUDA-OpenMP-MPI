#include <iostream>
#include <omp.h>

void long_task(int id) {
    std::cout << "Start Task " << id << " in Thread " << omp_get_thread_num() << std::endl;
    #pragma omp taskyield  // Allows other tasks to execute
    std::cout << "End Task " << id << " in Thread " << omp_get_thread_num() << std::endl;
}

int main() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task untied
            long_task(1);

            #pragma omp task untied
            long_task(2);
        }
    }
    return 0;
}
