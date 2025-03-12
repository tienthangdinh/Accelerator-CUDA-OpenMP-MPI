#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int data = 100;
    if (rank == 0) {
        // Blocking Send: The function returns only after message is sent.
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process 0: Sent data " << data << " (Blocking)\n";
    } else if (rank == 1) {
        int received_data;
        MPI_Recv(&received_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process 1: Received data " << received_data << " (Blocking)\n";
    }

    MPI_Finalize();
    return 0;
}
