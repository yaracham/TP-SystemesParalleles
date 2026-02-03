from mpi4py import MPI
import numpy as np
from time import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 120
    assert N % size == 0, "N doit être divisible par le nombre de processus"

    Nloc = N // size
    i0 = rank * Nloc
    i1 = i0 + Nloc

    # u complet sur tous les processus
    u = np.arange(1, N + 1, dtype=np.float64)

    # Chaque processus calcule seulement v_local (taille Nloc)
    v_local = np.zeros(Nloc, dtype=np.float64)

    t0 = time()
    for local_i, i in enumerate(range(i0, i1)):
        s = 0.0
        for j in range(N):
            Aij = ((i + j) % N) + 1.0
            s += Aij * u[j]
        v_local[local_i] = s

    # Rassembler tous les morceaux pour reconstruire v complet partout
    v_parts = comm.allgather(v_local)   # liste de tableaux
    v = np.concatenate(v_parts)
    t1 = time()

    if rank == 0:
        print(f"[Lignes] N={N}, np={size}, Nloc={Nloc}, time={t1-t0:.6f}s")
        print("v =", v)

if __name__ == "__main__":
    main()
