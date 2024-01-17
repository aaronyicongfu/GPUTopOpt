from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.tri as tri
from scipy import sparse
from scipy.sparse import linalg
from tqdm import tqdm


class QuadTree:
    """
    Quad tree for saving multi-grid hierarchy. Tree is assumed complete and
    implemented using array.
    """

    pass


class HelmholtzMG:
    """
    Solve the 2D Helmholtz problem using multigrid method
    """

    def __init__(self, conn_frame, X_frame, r0=0.1, nrefine=2):
        """
        Args:
            conn, X: connectivity and nodal locations for the frame, i.e. the
                     coarsest mesh
            r0: parameter of the Helmholtz PDE
            nrefine: number of multigrid refinements

        Note:
            - Numbering of the vertices for each conn entry should be
            counter-clockwise, i.e. conn[i, 0], conn[i, 1], conn[i, 2], conn[i,
            3] should be four counter-clockwise vertices
        """
        self.conn_frame = conn_frame
        self.X_frame = X_frame
        self.r0 = r0
        self.nrefine = nrefine

        self.conn = [conn_frame]
        self.X = [X_frame]

        for level in range(nrefine):
            conn_finer, X_finer = self.init_finer_mesh(self.conn[-1], self.X[-1])
            self.conn.append(conn_finer)
            self.X.append(X_finer)
            print(
                "refine(%d): nnodes: %d, nelems: %d"
                % (level, len(X_finer), len(conn_finer))
            )

        self.nelems = [len(conn) for conn in self.conn]
        self.nnodes = [len(X) for X in self.X]

        self.A = []
        self.B = []
        self.jacobi_smoothers = []
        self.gs_smoothers = []
        for level in range(nrefine + 1):
            A, B = self.assemble_jacobian(mg_level=level)
            self.A.append(A)
            self.B.append(B)
            self.jacobi_smoothers.append(JacobiSmoother(A))
            self.gs_smoothers.append(GSSmoother(A))

        return

    def vcycle(self, x0, b, level, smooth_method):
        if smooth_method == "jacobi":
            smoothers = self.jacobi_smoothers
        else:
            smoothers = self.gs_smoothers

        # Pre-smoothing
        x = smoothers[level].smooth(b, x0)

        # Compute residual
        res = b - self.A[level].dot(x)

        # Restrict residual
        res_coarse = self.restriction(res)

        # Recursion
        if level == 1:  # next level is coarsest
            # What's going on here? why even delta = 0.0 works?
            # delta = sparse.linalg.spsolve(self.A[level - 1], res_coarse)
            delta = smoothers[level - 1].smooth(res_coarse, np.zeros_like(res_coarse))
            # delta = np.zeros_like(res_coarse)
        else:
            delta = self.vcycle(
                np.zeros_like(res_coarse), res_coarse, level - 1, smooth_method
            )

        # Prolongation
        x += self.prolongation(delta)

        # Post-smoothing
        x = smoothers[level].smooth(b, x)

        return x

    def solve(self, b, niter=100, smooth_method="jacobi"):
        res = np.zeros(niter)
        x0 = np.zeros_like(b)
        res0 = np.linalg.norm(b - self.A[self.nrefine].dot(x0))

        for i in tqdm(range(niter)):
            x1 = self.vcycle(x0, b, self.nrefine, smooth_method)
            res[i] = np.linalg.norm(b - self.A[self.nrefine].dot(x0)) / res0
            x0 = x1

        return x1, res

    def init_finer_mesh(self, conn_coarse, X_coarse):
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
        conn = conn_coarse
        X = X_coarse
        pt = int(np.array(conn).max())

        def get_pt():
            nonlocal pt
            pt += 1
            return pt

        visited_edges = {}
        conn_finer = []
        X_finer = X.tolist()
        for p1, p2, p3, p4 in conn:
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

            # Order to append matters here
            conn_finer.append([p1, pd, pc, pl])
            conn_finer.append([pd, p2, pr, pc])
            conn_finer.append([pl, pc, pu, p4])
            conn_finer.append([pc, pr, p3, pu])

        X_finer = np.array(X_finer)

        return conn_finer, X_finer

    def assemble_jacobian(self, mg_level=0):
        conn = np.array(self.conn[mg_level])
        X = self.X[mg_level]
        nelems = len(conn)

        i = []
        j = []
        for index in range(nelems):
            for ii in conn[index, :]:
                for jj in conn[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        i_index = np.array(i, dtype=int)
        j_index = np.array(j, dtype=int)

        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 4 x 4 element stiffness matrices
        Ae = np.zeros((nelems, 4, 4))
        Ce = np.zeros((nelems, 4, 4))

        Be = np.zeros((nelems, 2, 4))
        He = np.zeros((nelems, 1, 4))
        J = np.zeros((nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = X[conn, 0]
        ye = X[conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N = 0.25 * np.array(
                    [
                        (1.0 - xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 + eta),
                        (1.0 - xi) * (1.0 + eta),
                    ]
                )
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                He[:, 0, :] = N
                Be[:, 0, :] = Nx
                Be[:, 1, :] = Ny

                Ce += np.einsum("n,nij,nil -> njl", detJ, He, He)
                Ae += np.einsum("n,nij,nil -> njl", detJ * self.r0**2, Be, Be)

        # Finish the computation of the Ae matrices
        Ae += Ce

        A = sparse.coo_matrix((Ae.flatten(), (i_index, j_index)))
        A = A.tocsr()

        B = sparse.coo_matrix((Ce.flatten(), (i_index, j_index)))
        B = B.tocsr()

        return A, B

    def restriction(self, vec):
        """
        Down-sampling the vector to a coarser mesh
        """
        if not len(vec) in self.nnodes:
            print("shape does not match")
            exit(-1)

        level = self.nnodes.index(len(vec))
        if level == 0:
            print("Can't restrict the coarsest mesh")
            return

        conn_fine = self.conn[level]
        conn_coarse = self.conn[level - 1]
        vec_r = np.zeros(self.nnodes[level - 1])

        for ec in range(self.nelems[level - 1]):
            p1c, p2c, p3c, p4c = conn_coarse[ec]
            p1f = conn_fine[4 * ec][0]
            p2f = conn_fine[4 * ec + 1][1]
            p3f = conn_fine[4 * ec + 3][2]
            p4f = conn_fine[4 * ec + 2][3]

            vec_r[p1c] = vec[p1f]
            vec_r[p2c] = vec[p2f]
            vec_r[p3c] = vec[p3f]
            vec_r[p4c] = vec[p4f]

        return vec_r

    def prolongation(self, vec):
        """
        Interpolating the vector to a finer mesh
        """
        if not len(vec) in self.nnodes:
            print("shape does not match")
            exit(-1)

        level = self.nnodes.index(len(vec))
        if level == self.nrefine:
            print("Can't interpolate the finest mesh")
            return

        conn_coarse = self.conn[level]
        conn_fine = self.conn[level + 1]
        vec_f = np.zeros(self.nnodes[level + 1])

        for ec in range(self.nelems[level]):
            p1c, p2c, p3c, p4c = conn_coarse[ec]
            p1f = conn_fine[4 * ec][0]
            p2f = conn_fine[4 * ec + 1][1]
            p3f = conn_fine[4 * ec + 3][2]
            p4f = conn_fine[4 * ec + 2][3]
            pl = conn_fine[4 * ec][3]
            pr = conn_fine[4 * ec + 1][2]
            pu = conn_fine[4 * ec + 2][2]
            pd = conn_fine[4 * ec][1]
            pc = conn_fine[4 * ec][2]

            vec_f[p1f] = vec[p1c]
            vec_f[p2f] = vec[p2c]
            vec_f[p3f] = vec[p3c]
            vec_f[p4f] = vec[p4c]
            vec_f[pl] = 0.5 * (vec[p1c] + vec[p4c])
            vec_f[pr] = 0.5 * (vec[p2c] + vec[p3c])
            vec_f[pu] = 0.5 * (vec[p3c] + vec[p4c])
            vec_f[pd] = 0.5 * (vec[p1c] + vec[p2c])
            vec_f[pc] = 0.25 * (vec[p1c] + vec[p2c] + vec[p3c] + vec[p4c])

        return vec_f


def plot_mesh(conn, X, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    vertices = [X[elem] for elem in conn]
    polygons = []
    for verts in vertices:
        polygons.append(Polygon(verts))

    p = PatchCollection(polygons, edgecolors="red", facecolors="none", lw=2.0)
    ax.add_collection(p)

    ax.set_aspect("equal")
    ax.set_xlim(left=X[:, 0].min(), right=X[:, 0].max())
    ax.set_ylim(bottom=X[:, 1].min(), top=X[:, 1].max())
    return ax


def plot_field(conn, X, vals, ax=None):
    """
    Plot a scalar field
    """
    if ax is None:
        fig, ax = plt.subplots()

    conn = np.array(conn)

    # Create the triangles
    nelems = len(conn)
    triangles = np.zeros((2 * nelems, 3), dtype=int)
    triangles[:nelems, 0] = conn[:, 0]
    triangles[:nelems, 1] = conn[:, 1]
    triangles[:nelems, 2] = conn[:, 2]

    triangles[nelems:, 0] = conn[:, 0]
    triangles[nelems:, 1] = conn[:, 2]
    triangles[nelems:, 2] = conn[:, 3]

    # Create the triangulation object
    tri_obj = tri.Triangulation(X[:, 0], X[:, 1], triangles)

    # Set the aspect ratio equal
    ax.set_aspect("equal")
    ax.set_xlim(left=X[:, 0].min(), right=X[:, 0].max())
    ax.set_ylim(bottom=X[:, 1].min(), top=X[:, 1].max())

    # Create the contour plot
    ax.tricontourf(tri_obj, vals, cmap="bwr")

    return


def to_vtk(vtk_path, conn, X, nodal_sols={}, cell_sols={}, nodal_vecs={}, cell_vecs={}):
    """
    Generate a vtk given conn, X, and optionally list of nodal solutions

    Args:
        nodal_sols: dictionary of arrays of length nnodes
        cell_sols: dictionary of arrays of length nelems
        nodal_vecs: dictionary of list of components [vx, vy], each has length nnodes
        cell_vecs: dictionary of list of components [vx, vy], each has length nelems
    """
    # vtk requires a 3-dimensional data point
    X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)

    conn = np.array(conn)
    nnodes = X.shape[0]
    nelems = conn.shape[0]

    # Create a empty vtk file and write headers
    with open(vtk_path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("my example\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")

        # Write nodal points
        fh.write("POINTS {:d} double\n".format(nnodes))
        for x in X:
            row = f"{x}"[1:-1]  # Remove square brackets in the string
            fh.write(f"{row}\n")

        # Write connectivity
        size = 5 * nelems

        fh.write(f"CELLS {nelems} {size}\n")
        for c in conn:
            node_idx = f"{c}"[1:-1]  # remove square bracket [ and ]
            npts = 4
            fh.write(f"{npts} {node_idx}\n")

        # Write cell type
        fh.write(f"CELL_TYPES {nelems}\n")
        for c in conn:
            vtk_type = 9
            fh.write(f"{vtk_type}\n")

        # Write solution
        if nodal_sols or nodal_vecs:
            fh.write(f"POINT_DATA {nnodes}\n")

        if nodal_sols:
            for name, data in nodal_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if nodal_vecs:
            for name, data in nodal_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

        if cell_sols or cell_vecs:
            fh.write(f"CELL_DATA {nelems}\n")

        if cell_sols:
            for name, data in cell_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if cell_vecs:
            for name, data in cell_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

    return


class JacobiSmoother:
    def __init__(self, A):
        self.diag = A.diagonal()
        self.lu = A.copy()
        self.lu.setdiag(0.0)
        self.lu.eliminate_zeros()
        return

    def smooth(self, b, x0):
        return (b - self.lu.dot(x0)) / self.diag


class GSSmoother:
    def __init__(self, A):
        self.ld = sparse.tril(A, format="csr")
        self.u = sparse.triu(A, k=1, format="csr")
        return

    def smooth(self, b, x0):
        return linalg.spsolve_triangular(self.ld, b - self.u.dot(x0), lower=True)


def solve(A, b, method="jacobi", niter=100):
    if method == "jacobi":
        smoother = JacobiSmoother(A)
    else:
        smoother = GSSmoother(A)
    res = np.zeros(niter)
    x0 = np.zeros_like(b)
    res0 = np.linalg.norm(b - A.dot(x0))

    for i in tqdm(range(niter)):
        x1 = smoother.smooth(b, x0)
        res[i] = np.linalg.norm(b - A.dot(x0)) / res0
        x0 = x1

    return x1, res


if __name__ == "__main__":
    np.random.seed(0)

    conn = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    X = np.array(
        [
            [0.0 + 0.1 * np.random.rand(), 0.0 + 0.1 * np.random.rand()],
            [1.0 + 0.1 * np.random.rand(), 0.0 + 0.1 * np.random.rand()],
            [2.0 + 0.1 * np.random.rand(), 0.0 + 0.1 * np.random.rand()],
            [0.0 + 0.1 * np.random.rand(), 1.0 + 0.1 * np.random.rand()],
            [1.0 + 0.1 * np.random.rand(), 1.0 + 0.1 * np.random.rand()],
            [2.0 + 0.1 * np.random.rand(), 1.0 + 0.1 * np.random.rand()],
            [0.0 + 0.1 * np.random.rand(), 2.0 + 0.1 * np.random.rand()],
            [1.0 + 0.1 * np.random.rand(), 2.0 + 0.1 * np.random.rand()],
            [2.0 + 0.1 * np.random.rand(), 2.0 + 0.1 * np.random.rand()],
        ]
    )

    x = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

    nrefine = 5

    helm = HelmholtzMG(conn, X, nrefine=nrefine)

    # plot_mesh(helm.conn[nrefine], helm.X[nrefine])
    # plt.show()

    # b = B.dot(np.random.rand(B.shape[0]))

    for i in range(nrefine):
        x = helm.prolongation(x)
    b = helm.B[nrefine].dot(x)

    sol_j, res_j = solve(helm.A[nrefine], b, method="jacobi")
    sol_gs, res_gs = solve(helm.A[nrefine], b, method="gs")
    sol_mg_j, res_mg_j = helm.solve(b, smooth_method="jacobi")
    sol_mg_gs, res_mg_gs = helm.solve(b, smooth_method="gs")

    plt.semilogy(res_j, label="Jacobi", color="red")
    plt.semilogy(res_gs, label="Gauss-Seidel", color="blue")
    plt.semilogy(res_mg_j, "--", label="MG Jacobi", color="red")
    plt.semilogy(res_mg_gs, "--", label="MG Gauss-Seidel", color="blue")
    plt.legend()

    fig, axs = plt.subplots(figsize=(9.6, 4.8), nrows=1, ncols=2)
    plot_field(helm.conn[nrefine], helm.X[nrefine], x, ax=axs[0])
    plot_field(helm.conn[nrefine], helm.X[nrefine], sol_mg_gs, ax=axs[1])

    plt.show()

    to_vtk(
        "sol.vtk",
        conn=helm.conn[nrefine],
        X=helm.X[nrefine],
        nodal_sols={
            "input": x,
            "jacobi": sol_j,
            "gs": sol_gs,
            "mg_jacobi": sol_mg_j,
            "mg_gs": sol_mg_gs,
        },
    )
