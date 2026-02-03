from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from math import log
import matplotlib.cm
import matplotlib.pyplot as plt
from time import time

TAG_WORK = 1
TAG_DONE = 2
TAG_STOP = 3

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (-0.75 < c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations

        z = 0
        for it in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return it + 1 - log(log(abs(z)))/log(2)
                return it
        return self.max_iterations

def compute_one_row(mset, width, height, y, xmin=-2.0, ymin=-1.125, xspan=3.0, yspan=2.25):
    scaleX = xspan / width
    scaleY = yspan / height
    row = np.empty((width,), dtype=np.float64)
    for x in range(width):
        c = complex(xmin + scaleX * x, ymin + scaleY * y)
        row[x] = mset.convergence(c, smooth=True)
    return row

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mset = MandelbrotSet(max_iterations=50, escape_radius=10.0)
    width, height = 1024, 1024

    comm.Barrier()
    t0 = time()

    if rank == 0:
        full = np.empty((height, width), dtype=np.float64)

        next_y = 0
        active = 0

        for worker in range(1, size):
            if next_y < height:
                comm.send(next_y, dest=worker, tag=TAG_WORK)
                next_y += 1
                active += 1
            else:
                comm.send(None, dest=worker, tag=TAG_STOP)

        while active > 0:
            status = MPI.Status()
            y, row = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DONE, status=status)
            full[y, :] = row

            worker = status.Get_source()

            if next_y < height:
                comm.send(next_y, dest=worker, tag=TAG_WORK)
                next_y += 1
            else:
                comm.send(None, dest=worker, tag=TAG_STOP)
                active -= 1

        comm.Barrier()
        t1 = time()
        print(f"Temps calcul (master-worker): {t1 - t0:.3f}s, np={size}")

        plt.imshow(matplotlib.cm.plasma(full))
        plt.axis("off")
        plt.show()

    else:
        while True:
            y = comm.recv(source=0, tag=MPI.ANY_TAG)
            status = MPI.Status()
            if y is None:
                break
            row = compute_one_row(mset, width, height, y)
            comm.send((y, row), dest=0, tag=TAG_DONE)

        comm.Barrier()

if __name__ == "__main__":
    main()
