from mpi4py import MPI
import numpy as np
import pygame as pg


def split_rows(n: int, size: int):
    base = n // size
    rem = n % size
    counts = [base + (1 if r < rem else 0) for r in range(size)] #on ajoute +1 aux premiers rem rangs pour bien equilibrer et ne pas tout mettre au dernier ou premier rang
    disp = [0] * size #on sauvegarde ici l'index de la premiere ligne de chaque rang
    acc = 0
    for r in range(size):
        disp[r] = acc
        acc += counts[r]
    return counts, disp


class LocalGrille:

    def __init__(self, global_shape, local_rows, init_local_cells,
                 color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.global_ny, self.nx = global_shape
        self.local_ny = local_rows
        self.cells = init_local_cells.astype(np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self, top_ghost: np.ndarray, bot_ghost: np.ndarray):

        assert top_ghost.shape == (self.nx,)
        assert bot_ghost.shape == (self.nx,)

        all = np.empty((self.local_ny + 2, self.nx), dtype=np.uint8)
        all[0, :] = top_ghost #ici on ajoute la ligne du voisin du haut avec les lignes du rang lui meme et ensuite al igne du voisin du bas
        all[1:-1, :] = self.cells
        all[-1, :] = bot_ghost

        next_cells = np.empty_like(self.cells, dtype=np.uint8)

        for li in range(self.local_ny):
            ei = li+1  
            i_above = ei-1
            i_below = ei+1

            for j in range(self.nx):
                j_left = (j-1 + self.nx) % self.nx
                j_right = (j+1) % self.nx

                voisins_i = [i_above, i_above, i_above, ei, ei, i_below, i_below, i_below]
                voisins_j = [j_left, j, j_right, j_left, j_right, j_left, j, j_right]

                voisines = all[voisins_i, voisins_j]
                nb_voisines_vivantes = int(np.sum(voisines))

                if all[ei, j] == 1:  # vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[li, j] = 0
                    else:
                        next_cells[li, j] = 1
                else:  # morte
                    if nb_voisines_vivantes == 3:
                        next_cells[li, j] = 1
                    else:
                        next_cells[li, j] = 0

        self.cells = next_cells


class App:
    #on affiche sur rang 0 uniquement, rang 1 lui envoie la grille globale

    def __init__(self, geometry, global_cells,
                 color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.col_life = color_life
        self.col_dead = color_dead
        self.global_cells = global_cells

        ny, nx = global_cells.shape

        # Taille d'une cellule
        self.size_x = geometry[1] // nx
        self.size_y = geometry[0] // ny

        # Couleur de grille 
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None

        # Ajuster taille fenêtre pour fitter pile
        self.width = nx * self.size_x
        self.height = ny * self.size_y

        # Créer fenêtre
        self.screen = pg.display.set_mode((self.width, self.height))

    def update_cells(self, new_global_cells: np.ndarray):
        self.global_cells = new_global_cells

    def compute_rectangle(self, i: int, j: int):
        return (self.size_x * j,
                self.height - self.size_y * (i + 1),
                self.size_x,
                self.size_y)

    def compute_color(self, i: int, j: int):
        if self.global_cells[i, j] == 0:
            return self.col_dead
        else:
            return self.col_life

    def draw(self):
        ny, nx = self.global_cells.shape

        [self.screen.fill(self.compute_color(i, j), self.compute_rectangle(i, j))
         for i in range(ny)
         for j in range(nx)]

        if self.draw_color is not None:
            [pg.draw.line(self.screen, self.draw_color,
                          (0, i * self.size_y),
                          (self.width, i * self.size_y))
             for i in range(ny)]
            [pg.draw.line(self.screen, self.draw_color,
                          (j * self.size_x, 0),
                          (j * self.size_x, self.height))
             for j in range(nx)]

        pg.display.update()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import sys
    import time

    # Patterns identiques
    dico_patterns = {
        'blinker': ((5, 5), [(2, 1), (2, 2), (2, 3)]),
        'toad': ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
        "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
        "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
        "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
        "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
        "glider_gun": ((400, 400), [
            (51, 76), (52, 74), (52, 76), (53, 64), (53, 65), (53, 72), (53, 73), (53, 86), (53, 87),
            (54, 63), (54, 67), (54, 72), (54, 73), (54, 86), (54, 87),
            (55, 52), (55, 53), (55, 62), (55, 68), (55, 72), (55, 73),
            (56, 52), (56, 53), (56, 62), (56, 66), (56, 68), (56, 69), (56, 74), (56, 76),
            (57, 62), (57, 68), (57, 76),
            (58, 63), (58, 67),
            (59, 64), (59, 65)
        ]),
        "space_ship": ((25, 25), [(11, 13), (11, 14), (12, 11), (12, 12), (12, 14), (12, 15),
                                  (13, 11), (13, 12), (13, 13), (13, 14), (14, 12), (14, 13)]),
        "die_hard": ((100, 100), [(51, 57), (52, 51), (52, 52), (53, 52), (53, 56), (53, 57), (53, 58)]),
        "pulsar": ((17, 17), [
            (2, 4), (2, 5), (2, 6), (7, 4), (7, 5), (7, 6), (9, 4), (9, 5), (9, 6), (14, 4), (14, 5), (14, 6),
            (2, 10), (2, 11), (2, 12), (7, 10), (7, 11), (7, 12), (9, 10), (9, 11), (9, 12), (14, 10), (14, 11),
            (14, 12),
            (4, 2), (5, 2), (6, 2), (4, 7), (5, 7), (6, 7), (4, 9), (5, 9), (6, 9), (4, 14), (5, 14), (6, 14),
            (10, 2), (11, 2), (12, 2), (10, 7), (11, 7), (12, 7), (10, 9), (11, 9), (12, 9), (10, 14), (11, 14),
            (12, 14)
        ]),
        "floraison": ((40, 40), [(19, 18), (19, 19), (19, 20), (20, 17), (20, 19), (20, 21), (21, 18), (21, 19), (21, 20)]),
        "block_switch_engine": ((400, 400), [(201, 202), (201, 203), (202, 202), (202, 203), (211, 203), (212, 204),
                                             (212, 202), (214, 204), (214, 201), (215, 201), (215, 202), (216, 201)]),
        "u": ((200, 200), [(101, 101), (102, 102), (103, 102), (103, 101), (104, 103), (105, 103), (105, 102),
                           (105, 101), (105, 105), (103, 105), (102, 105), (101, 105), (101, 104)]),
        "flat": ((200, 400), [(80, 200), (81, 200), (82, 200), (83, 200), (84, 200), (85, 200), (86, 200), (87, 200),
                              (89, 200), (90, 200), (91, 200), (92, 200), (93, 200), (97, 200), (98, 200), (99, 200),
                              (106, 200), (107, 200), (108, 200), (109, 200), (110, 200), (111, 200), (112, 200),
                              (114, 200), (115, 200), (116, 200), (117, 200), (118, 200)])
    }

    # Args
    choice = 'flat'
    resx, resy = 800, 800
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])

    # Init global sur rank 0
    if rank == 0:
        if choice not in dico_patterns:
            print("No such pattern. Available ones are:", list(dico_patterns.keys()))
            comm.Abort(1)
        (ny, nx), pattern = dico_patterns[choice]
        global_cells = np.zeros((ny, nx), dtype=np.uint8)
        for (i, j) in pattern:
            global_cells[i, j] = 1
        print(f"Pattern initial choisi : {choice}")
        print(f"resolution ecran : {(resx, resy)}")
    else:
        ny = nx = None
        global_cells = None

    # Broadcast dimensions
    ny = comm.bcast(ny if rank == 0 else None, root=0)
    nx = comm.bcast(nx if rank == 0 else None, root=0)

    # Décomposition
    counts_rows, disp_rows = split_rows(ny, size)
    local_ny = counts_rows[rank]

    sendcounts = [c * nx for c in counts_rows]
    displs_elems = [d * nx for d in disp_rows]

    # Scatter init
    local_buf = np.empty(local_ny * nx, dtype=np.uint8)
    if rank == 0:
        comm.Scatterv([global_cells.ravel(), sendcounts, displs_elems, MPI.UNSIGNED_CHAR], local_buf, root=0)
    else:
        comm.Scatterv([None, sendcounts, displs_elems, MPI.UNSIGNED_CHAR], local_buf, root=0)
    local_cells = local_buf.reshape((local_ny, nx))

    grid = LocalGrille(global_shape=(ny, nx), local_rows=local_ny, init_local_cells=local_cells)

    if rank == 0:
        pg.init()
        app = App((resx, resy), global_cells)

    up = (rank - 1) % size
    down = (rank + 1) % size

    mustContinue = True
    while mustContinue:
        t1 = time.time()

        # Exchange ghost rows
        top_send = grid.cells[0, :].copy()
        bot_send = grid.cells[-1, :].copy()
        top_ghost = np.empty(nx, dtype=np.uint8)
        bot_ghost = np.empty(nx, dtype=np.uint8)

        # Envoi/receive pour obtenir ghost bas depuis down
        comm.Sendrecv(sendbuf=top_send, dest=up, sendtag=11,  #c'est mieux de faire snedrecv directement car c'est plus rapide et ca empeche le blocage
                      recvbuf=bot_ghost, source=down, recvtag=11)

        # Envoi/receive pour obtenir ghost haut depuis up
        comm.Sendrecv(sendbuf=bot_send, dest=down, sendtag=22,
                      recvbuf=top_ghost, source=up, recvtag=22)

        # Calcul local
        grid.compute_next_iteration(top_ghost, bot_ghost)
        t2 = time.time()

        # tous -> hub, puis hub -> rank 0 (un seul message)
        hub = 1 if size > 1 else 0

        if rank == hub:
            recv_global = np.empty(ny * nx, dtype=np.uint8)
            comm.Gatherv(grid.cells.ravel(),
                         [recv_global, sendcounts, displs_elems, MPI.UNSIGNED_CHAR],
                         root=hub)
            if hub != 0:
                comm.Send([recv_global, MPI.UNSIGNED_CHAR], dest=0, tag=99)
            else:
                global_cells = recv_global.reshape((ny, nx))

        elif rank == 0:
            if hub != 0:
                recv_global = np.empty(ny * nx, dtype=np.uint8)
            else:
                recv_global = None

            comm.Gatherv(grid.cells.ravel(),
                         [None, sendcounts, displs_elems, MPI.UNSIGNED_CHAR],
                         root=hub)

            if hub != 0:
                comm.Recv([recv_global, MPI.UNSIGNED_CHAR], source=hub, tag=99)
                global_cells = recv_global.reshape((ny, nx))

        else:
            comm.Gatherv(grid.cells.ravel(),
                         [None, sendcounts, displs_elems, MPI.UNSIGNED_CHAR],
                         root=hub)

        t3 = time.time()

        if rank == 0:
            app.update_cells(global_cells)
            app.draw()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False

            print(
                f"Temps calcul prochaine generation : {t2 - t1:2.2e} s, "
                f"temps comm+affichage : {t3 - t2:2.2e} s\r",
                end=''
            )

        mustContinue = comm.bcast(mustContinue, root=0)

    if rank == 0:
        pg.quit()


if __name__ == "__main__":
    main()
