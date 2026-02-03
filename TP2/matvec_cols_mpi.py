from mpi4py import MPI
import numpy as np
from time import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Dimension (modifiable)
    N = 120
    assert N % size == 0, "N doit être divisible par le nombre de processus"

    Nloc = N // size
    j0 = rank * Nloc
    j1 = j0 + Nloc

    # u complet sur tous les processus (on le construit pareil partout)
    u = np.arange(1, N + 1, dtype=np.float64)

    # Calcul local : contribution des colonnes j0..j1-1
    # v_part[i] = sum_{j=j0..j1-1} A[i,j]*u[j]
    v_part = np.zeros(N, dtype=np.float64)

    t0 = time()
    for i in range(N):
        # on calcule seulement les colonnes locales
        for j in range(j0, j1):
            Aij = ((i + j) % N) + 1.0
            v_part[i] += Aij * u[j]

    # Somme globale des contributions partielles -> v complet
    v = np.zeros(N, dtype=np.float64)
    comm.Allreduce(v_part, v, op=MPI.SUM)
    t1 = time()

    if rank == 0:
        print(f"[Colonnes] N={N}, np={size}, Nloc={Nloc}, time={t1-t0:.6f}s")
        print("v =", v)

if __name__ == "__main__":
    main()
