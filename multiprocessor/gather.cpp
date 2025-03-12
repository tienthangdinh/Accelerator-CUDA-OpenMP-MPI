#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_data = rank + 1;  // Each process sends its rank + 1
    int recv_data[4];  // Buffer to collect data at root

    MPI_Gather(&send_data, 1, MPI_INT, recv_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Root process gathered: ";
        for (int i = 0; i < 4; i++)
            std::cout << recv_data[i] << " ";
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}
