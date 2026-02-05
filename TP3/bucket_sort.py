import numpy as np
import mpi4py.MPI as MPI

globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank

def bucket_id(x, minimum, maximum, b):
    bid = int((x - minimum) / (maximum - minimum) * b)
    if bid == b:
        bid = b - 1
    return bid


def bucket_sort_seq(data, b):
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    buckets = [[] for _ in range(b)]
    for x in data:
        buckets[bucket_id(x, minimum, maximum, b)].append(float(x))
    for i in range(b):
        buckets[i].sort()
    res = []
    for i in range(b):
        res.extend(buckets[i])
    return res

def bucket_sort_par(data, b, comm):

    if rank >= b:
        return None
    
    if rank == 0:
        minimum = float(np.min(data))
        maximum = float(np.max(data))
        buckets = [[] for _ in range(b)]
        for x in data:
            buckets[bucket_id(x, minimum, maximum, b)].append(float(x))
        for i in range(1, b):
            comm.send(buckets[i], dest=i, tag=0)
    
        local = buckets[0]
        local.sort()

    else:
        local = comm.recv(source=0, tag=0)
        local.sort()
        comm.send(local, dest=0, tag=1)
    
    if rank == 0:
        result = []
        result.extend(local)
        for p in range(1, b):
            result.extend(comm.recv(source=p, tag=1))
        return result
    
def demo_print():
    if rank == 0:
        data = np.random.default_rng(0).random(20)
    else:
        data = None

    bucket_values = [b for b in [4, 8, 16, 32] if b <= nbp]

    seq_times = {}
    par_times = {}
    seq_results = {}
    par_results = {}

    for b in bucket_values:
        if rank == 0:
            t0 = MPI.Wtime()
            seq_results[b] = bucket_sort_seq(data, b=b)
            t1 = MPI.Wtime()
            seq_times[b] = t1 - t0

        globCom.barrier()
        t0 = MPI.Wtime()
        par_res = bucket_sort_par(data, b, globCom)
        globCom.barrier()
        t1 = MPI.Wtime()
        if rank == 0:
            par_results[b] = par_res
            par_times[b] = t1 - t0

    if rank == 0:
        print("Initial list:")
        print(np.round(data, 4))
        print()
        for b in bucket_values:
            print(f"Buckets: {b}")
            print(f"  Seq time (s): {seq_times[b]:.6f}")
            print(f"  Par time (s): {par_times[b]:.6f}")
            print("  Seq result:")
            print(np.round(seq_results[b], 4))
            print("  Par result:")
            print(np.round(par_results[b], 4))
            print()
        print()

if __name__ == "__main__":
    demo_print()