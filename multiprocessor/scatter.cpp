#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 8;
    int data[n] = {1, 2, 3, 4, 5, 6, 7, 8};  // Data at root (rank 0)
    int recv_buf[2];  // Buffer to receive scattered elements

    //data, element count, type of data
    MPI_Scatter(data, 2, MPI_INT, recv_buf, 2, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << "Process " << rank << " received: " << recv_buf[0] << ", " << recv_buf[1] << "\n";

    MPI_Finalize();
    return 0;
}
