import numpy as np
import matplotlib.pylab as plt


class CGD:
    def __init__(self, nx=10, ny=10, lx=1.0, ly=1.0, p=3):
        self.p = p
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.nnodes = (nx + 1) * (ny + 1)

        self.nodes = np.arange(0, self.nnodes, dtype=int).reshape((nx + 1, ny + 1))

        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        x = np.linspace(0, lx, self.nx + 1)
        y = np.linspace(0, ly, self.ny + 1)

        self.xpts = np.zeros(self.nnodes)
        self.ypts = np.zeros(self.nnodes)
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                self.xpts[self.nodes[i, j]] = x[i]
                self.ypts[self.nodes[i, j]] = y[j]

    def eval_bernstein(self, u, p):
        """
        Evaluate the Bernstein polynomial basis functions at the given parametric locations
        Parameters
        ----------
        u : np.ndarray
            Parametric locations for the basis functions
        p : int
            Degree of the polynomial
        """
        u1 = 1.0 - u
        u2 = 1.0 * u

        N = np.zeros((len(u), p + 1))
        N[:, 0] = 1.0

        s = np.zeros(len(u))
        t = np.zeros(len(u))
        for j in range(1, p + 1):
            s[:] = 0.0
            t[:] = 0.0
            for k in range(j):
                t[:] = N[:, k]
                N[:, k] = s + u1 * t
                s = u2 * t
            N[:, j] = s

        return N

    def eval_poly_basis(self, u, v, p):
        """Evalaute the polynomial basis functions at all of the specified
        locations in the scaled coordinate system"""

        Nu = self.eval_bernstein(u, p)
        Nv = self.eval_bernstein(v, p)

        N = np.zeros((len(u), (self.p + 1) ** 2))
        for j in range(self.p + 1):
            for i in range(self.p + 1):
                N[:, i + j * (self.p + 1)] = Nu[:, i] * Nv[:, j]

        return N

    def get_stencil_1d(self, i, p, n):
        """Get the regular stencil along a given dimension"""
        m = (p - 1) // 2
        if i - m < 0:
            start = 0
            end = p + 1
        elif i + 1 + m >= n + 1:
            start = n - p
            end = n + 1
        else:
            start = i - m
            end = i + 2 + m

        return start, end

    def get_stencil(self, elem, p):
        """Get the stencil for the given element index"""

        i = elem % self.nx
        j = elem // self.nx

        istart, iend = self.get_stencil_1d(i, p, self.nx)
        jstart, jend = self.get_stencil_1d(j, p, self.ny)

        Sk = []
        for jx in range(jstart, jend):
            for ix in range(istart, iend):
                Sk.append(self.nodes[ix, jx])

        return Sk

    def apply_element_scaling(self, elem, x, y):
        # Apply the shift to the local element coordinates
        n = self.nodes[elem % self.nx, elem // self.nx]
        x0 = self.xpts[n]
        y0 = self.ypts[n]
        u = (x - x0) / self.dx
        v = (y - y0) / self.dy

        return u, v

    def get_basis_coef(self, elem):
        # Get the dof indicies that belong in the stencil
        sk = self.get_stencil(elem, self.p)

        # Get the coordinates for
        xk = self.xpts[sk]
        yk = self.ypts[sk]
        u, v = self.apply_element_scaling(elem, xk, yk)

        # Evaluate the polynomial basis
        Vk = self.eval_poly_basis(u, v, self.p)

        # Compute the basis coefficients
        return np.linalg.inv(Vk), sk

    def eval_element_basis(self, elem, x, y):
        # Get the coefficients for the basis functions
        Ck, sk = self.get_basis_coef(elem)

        u, v = self.apply_element_scaling(elem, x, y)

        # Evaluate the basis functions
        V = self.eval_poly_basis(u, v, self.p)

        return np.dot(V, Ck), sk

    def eval_element_interp(self, elem, x, y, values):
        """Perform an interpolation over the given element with the specified values"""

        N, sk = self.eval_element_basis(elem, x, y)

        uk = np.zeros(len(x))
        for i, dof in enumerate(sk):
            uk += N[:, i] * values[dof]

        return uk


p = 9
nx = 25
ny = 25
lx = 1.0
ly = 1.0
cgd = CGD(nx=nx, ny=ny, lx=lx, ly=ly, p=p)

fig, ax = plt.subplots(1, 1)

xint = np.linspace(0, lx, nx + 1)
yint = np.linspace(0, ly, ny + 1)

levels = np.linspace(-1, 1)

values = np.cos(2.0 * np.pi * cgd.xpts) * np.sin(3.0 * np.pi * cgd.ypts)

# values[:] = 0.0
# values[nx // 2 * (1 + (nx + 1))] = 1.0


for j in range(ny):
    for i in range(nx):
        elem = i + j * nx
        xv = np.linspace(xint[i], xint[i + 1], 8)
        yv = np.linspace(yint[j], yint[j + 1], 8)

        X, Y = np.meshgrid(xv, yv)
        x = X.flatten()
        y = Y.flatten()

        u = cgd.eval_element_interp(elem, x, y, values)

        ax.contour(X, Y, u.reshape(X.shape), levels=levels)

plt.show()
