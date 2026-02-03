from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from math import log
from time import time

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

def compute_rows(mset, width, height, y0, y1, xmin=-2.0, ymin=-1.125, xspan=3.0, yspan=2.25):
    scaleX = xspan / width
    scaleY = yspan / height
    block = np.empty((y1 - y0, width), dtype=np.float64) 
    for j, y in enumerate(range(y0, y1)):
        for x in range(width):
            c = complex(xmin + scaleX * x, ymin + scaleY * y)
            block[j, x] = mset.convergence(c, smooth=True)
    return block

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mset = MandelbrotSet(max_iterations=50, escape_radius=10.0)
    width, height = 1024, 1024

    base = height // size
    rem = height % size
    y0 = rank * base + min(rank, rem)
    local_h = base + (1 if rank < rem else 0)
    y1 = y0 + local_h

    comm.Barrier()
    t0 = time()

    local_block = compute_rows(mset, width, height, y0, y1)

    gathered = comm.gather(local_block, root=0)

    comm.Barrier()
    t1 = time()

    if rank == 0:
        import matplotlib
        matplotlib.use("TkAgg", force=True)
        import matplotlib.cm
        import matplotlib.pyplot as plt
        full = np.vstack(gathered)  

        print(f"Temps calcul+gather (bloc): {t1 - t0:.3f}s, np={size}")

        deb = time()
        plt.imshow(matplotlib.cm.plasma(full))
        plt.axis("off")
        plt.show()
        fin = time()
        print(f"Temps de constitution de l'image : {fin-deb}")

if __name__ == "__main__":
    main()
