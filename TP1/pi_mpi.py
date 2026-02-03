from mpi4py import MPI
import random
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_00

base = N // size
rem = N % size
localN = base + (1 if rank < rem else 0)

random.seed(int(time.time()) + rank * 1234567)

comm.Barrier()
t0 = time.time()

local_hits = 0
for _ in range(localN):
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if x*x + y*y <= 1.0:
        local_hits += 1

total_hits = comm.reduce(local_hits, op=MPI.SUM, root=0)

pi = None
if rank == 0:
    pi = 4.0 * total_hits / N

pi = comm.bcast(pi, root=0)

t1 = time.time()
if rank == 0:
    print(f"mpi4py: N={N} hits={total_hits} pi≈{pi:.10f} time={t1-t0:.3f}s (np={size})")
