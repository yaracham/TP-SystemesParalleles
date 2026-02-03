#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double urand(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX; 
}

int main(int argc, char **argv)
{
    long long N = (argc > 1) ? atoll(argv[1]) : 10000000LL;

    unsigned int seed = (unsigned int)time(NULL);
    long long hits = 0;

    clock_t t0 = clock();

    for (long long i = 0; i < N; i++) {
        double x = 2.0 * urand(&seed) - 1.0;
        double y = 2.0 * urand(&seed) - 1.0;
        if (x*x + y*y <= 1.0) hits++;
    }

    clock_t t1 = clock();
    double pi = 4.0 * (double)hits / (double)N;
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("N=%lld hits=%lld pi≈%.10f time=%.3fs\n", N, hits, pi, secs);
    return 0;
}
