#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline double urand(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = (argc > 1) ? atoll(argv[1]) : 10000000LL;

    long long base = N / size;
    long long rem  = N % size;
    long long localN = base + (rank < rem ? 1 : 0);

    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(rank * 987654321u);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    long long localHits = 0;
    for (long long i = 0; i < localN; i++) {
        double x = 2.0 * urand(&seed) - 1.0;
        double y = 2.0 * urand(&seed) - 1.0;
        if (x*x + y*y <= 1.0) localHits++;
    }

    long long totalHits = 0;

    if (rank == 0) {
        totalHits = localHits;
        for (int src = 1; src < size; src++) {
            long long h = 0;
            MPI_Recv(&h, 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            totalHits += h;
        }
    } else {
        MPI_Send(&localHits, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    double pi = 0.0;
    if (rank == 0) {
        pi = 4.0 * (double)totalHits / (double)N;
    }
    MPI_Bcast(&pi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("MPI: N=%lld totalHits=%lld pi≈%.10f time=%.3fs (np=%d)\n",
               N, totalHits, pi, t1 - t0, size);
    }

    MPI_Finalize();
    return 0;
}
