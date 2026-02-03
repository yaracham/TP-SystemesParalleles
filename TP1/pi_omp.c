#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static inline double urand(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

int main(int argc, char **argv)
{
    long long N = (argc > 1) ? atoll(argv[1]) : 10000000LL;
    long long hits_total = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(1234567 * omp_get_thread_num());
        long long hits = 0;

        #pragma omp for
        for (long long i = 0; i < N; i++) {
            double x = 2.0 * urand(&seed) - 1.0;
            double y = 2.0 * urand(&seed) - 1.0;
            if (x*x + y*y <= 1.0) hits++;
        }

        #pragma omp atomic
        hits_total += hits;
    }

    double t1 = omp_get_wtime();
    double pi = 4.0 * (double)hits_total / (double)N;

    printf("N=%lld hits=%lld pi≈%.10f time=%.3fs\n", N, hits_total, pi, t1 - t0);
    return 0;
}
