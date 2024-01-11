import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class HelmholtzMG:
    """
    Solve the 2D Helmholtz problem using multigrid method
    """

    def __init__(self, conn, X, num_mg_levels=2):
        """
        Args:
            conn, X: connectivity and nodal locations for the frame, i.e. the
                     coarsest mesh

        Note:
            - Numbering of the vertices for each conn entry should be
            counter-clockwise, i.e. conn[i, 0], conn[i, 1], conn[i, 2], conn[i,
            3] should be four counter-clockwise vertices
        """
        self.conn = conn
        self.X = X
        self.num_mg_levels = num_mg_levels

        self.nverts = len(X)
        self.ngroups = len(conn)

        self.conn_hierarchy = []
        self.X_hierarchy = []

        for level in range(num_mg_levels):
            _conn = []
            self.conn_hierarchy

        return

    def _init_finer(self, conn, X):
        """
        p4           p3          p4        pu       p3
         -------------               -------------
        |             |             |4e + 2|4e + 3|
        |      e      |      =>  pl |______|______| pr
        |             |             |4e  pc|4e + 1|
        |             |             |      |      |
         -------------           p1  -------------  p2
        p1           p2                    pd
        """
        pt = int(np.array(conn).max())

        def get_pt():
            nonlocal pt
            pt += 1
            return pt

        visited_edges = {}
        conn_finer = []
        X_finer = X.tolist()
        for e, (p1, p2, p3, p4) in enumerate(conn):
            e1 = 4 * e
            e2 = 4 * e + 1
            e3 = 4 * e + 2
            e4 = 4 * e + 3

            fl = (p1, p4) if p1 < p4 else (p4, p1)
            fr = (p2, p3) if p2 < p3 else (p3, p2)
            fu = (p4, p3) if p4 < p3 else (p3, p4)
            fd = (p1, p2) if p1 < p2 else (p2, p1)

            # Get mid points
            pc = get_pt()
            X_finer.append(0.25 * (X[p1] + X[p2] + X[p3] + X[p4]))

            if fl in visited_edges:
                pl = visited_edges[fl]
            else:
                pl = get_pt()
                X_finer.append(0.5 * (X[p1] + X[p4]))

            if fr in visited_edges:
                pr = visited_edges[fr]
            else:
                pr = get_pt()
                X_finer.append(0.5 * (X[p2] + X[p3]))

            if fu in visited_edges:
                pu = visited_edges[fu]
            else:
                pu = get_pt()
                X_finer.append(0.5 * (X[p3] + X[p4]))

            if fd in visited_edges:
                pd = visited_edges[fd]
            else:
                pd = get_pt()
                X_finer.append(0.5 * (X[p1] + X[p2]))

            conn_finer.append([p1, pd, pc, pl])
            conn_finer.append([pd, p2, pr, pc])
            conn_finer.append([pl, pc, pu, p4])
            conn_finer.append([pc, pr, p3, pu])

        X_finer = np.array(X_finer)

        return conn_finer, X_finer


def plot_mesh(conn, X, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    vertices = [X[elem] for elem in conn]
    polygons = []
    for verts in vertices:
        print(verts)
        polygons.append(Polygon(verts))

    p = PatchCollection(polygons, edgecolors="red", facecolors="none", lw=2.0)
    ax.add_collection(p)

    ax.set_aspect("equal")
    ax.set_xlim(left=X[:, 0].min(), right=X[:, 0].max())
    ax.set_ylim(bottom=X[:, 1].min(), top=X[:, 1].max())
    return ax


if __name__ == "__main__":
    conn = [[0, 1, 4, 3], [1, 2, 5, 4]]
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )

    helm = HelmholtzMG(conn, X)

    conn_finer, X_finer = helm._init_finer(conn, X)

    print(conn_finer)
    print(X_finer)

    plot_mesh(conn_finer, X_finer)
    plt.show()
