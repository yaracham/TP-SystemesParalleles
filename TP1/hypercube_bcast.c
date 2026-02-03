#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int is_power_of_two(int x) { return x > 0 && (x & (x - 1)) == 0; }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int d = (argc > 1) ? atoi(argv[1]) : -1;
    if (d < 0) {
        if (rank == 0) fprintf(stderr, "Usage: %s <d>   with np = 2^d\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int expected = 1 << d;
    if (size != expected) {
        if (rank == 0) fprintf(stderr, "Error: need np=%d (2^d). Got np=%d\n", expected, size);
        MPI_Finalize();
        return 1;
    }

    int token;
    int has_token = 0;

    if (rank == 0) {
        token = 42;         
        has_token = 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int s = 0; s < d; s++) {
        int partner = rank ^ (1 << s);

        if (has_token) {
            MPI_Send(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            has_token = 1;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();
    printf("[rank %d] token=%d time=%.6fs\n", rank, token, t1 - t0);

    MPI_Finalize();
    return 0;
}
