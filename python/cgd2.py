import numpy as np
import matplotlib.pylab as plt


class CGD:
    def __init__(self, nx=10, ny=10, lx=1.0, ly=1.0, p=3, blank=None):
        self.p = p
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.nnodes = (nx + 1) * (ny + 1)

        # Assign an initial set of node numbers
        self.nodes = np.ones((nx + 1, ny + 1), dtype=int)

        if blank is not None:
            for j in range(ny):
                for i in range(nx):
                    if not blank[i, j]:
                        self.nodes[i, j] *= 0
                        self.nodes[i, j + 1] *= 0
                        self.nodes[i + 1, j] *= 0
                        self.nodes[i + 1, j + 1] *= 0

            self.blank = blank
        else:
            self.blank = None

        # Constrain nodes that have been blanked
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                if self.nodes[i, j] > 0:
                    self.nodes[i, j] = -1

        # Set the degrees of freedom
        dof_index = 0
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                if self.nodes[i, j] >= 0:
                    self.nodes[i, j] = dof_index
                    dof_index += 1

        self.ndof = dof_index

        # Set the dimensions of the problem
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = np.linspace(0, lx, self.nx + 1)
        self.y = np.linspace(0, ly, self.ny + 1)

        self.xpts = np.zeros(self.ndof)
        self.ypts = np.zeros(self.ndof)
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                if self.nodes[i, j] >= 0:
                    self.xpts[self.nodes[i, j]] = self.x[i]
                    self.ypts[self.nodes[i, j]] = self.y[j]

        return

    def _get_xedge_dof(self, i, j, p):
        # Set the degrees of freedom along the x-edge
        start, end = self.get_stencil_1d(i, p, self.nx)

        dof = []
        for ix in range(start, end):
            if self.nodes[ix, j] >= 0:
                break

        ii = 0
        while ix < self.nx + 1 and self.nodes[ix, j] >= 0 and ii < p + 1:
            dof.append(self.nodes[ix, j])
            ix += 1
            ii += 1

        return dof

    def _get_yedge_dof(self, i, j, p):
        # Set the degrees of freedom along the x-edge
        start, end = self.get_stencil_1d(j, p, self.ny)

        dof = []
        for jx in range(start, end):
            if self.nodes[i, jx] >= 0:
                break

        jj = 0
        while jx < self.ny + 1 and self.nodes[i, jx] >= 0 and jj < p + 1:
            dof.append(self.nodes[i, jx])
            jx += 1
            jj += 1

        return dof

    def eval_poly(self, u, p):
        N = np.zeros((len(u), p + 1))
        N[:, 0] = 1.0
        for k in range(1, p + 1):
            N[:, k] = (1.0 / k) * u**k

        return N

    def eval_poly_basis(self, u, v, px, py):
        """Evalaute the polynomial basis functions at all of the specified
        locations in the scaled coordinate system"""

        Nu = self.eval_poly(u, px)
        Nv = self.eval_poly(v, py)

        N = np.zeros((len(u), (px + 1) * (py + 1)))

        index = 0
        for k in range(px + py + 1):
            for i in range(k + 1):
                j = k - i
                if i <= px and j <= py:
                    N[:, index] = Nu[:, i] * Nv[:, j]
                    index += 1

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

    def get_xstencil(self, elem, p):
        i = elem % self.nx
        j = elem // self.nx

        px = p
        py = p

        istart, iend = self.get_stencil_1d(i, px, self.nx)
        for ix in range(istart, iend):
            dof = self._get_yedge_dof(ix, j, py)
            if len(dof) == 0:
                px = 1
            if len(dof) < py + 1:
                py = py - 1

        Sk = []
        istart, iend = self.get_stencil_1d(i, px, self.nx)
        for ix in range(istart, iend):
            dof = self._get_yedge_dof(ix, j, py)
            Sk.extend(dof)

        return Sk, px, py

    def get_ystencil(self, elem, p):
        i = elem % self.nx
        j = elem // self.nx

        px = p
        py = p

        jstart, jend = self.get_stencil_1d(j, py, self.ny)
        for jx in range(jstart, jend):
            dof = self._get_xedge_dof(i, jx, px)
            if len(dof) == 0:
                py = 1
            if len(dof) < px + 1:
                px = px - 1

        Sk = []
        jstart, jend = self.get_stencil_1d(j, py, self.ny)
        for jx in range(jstart, jend):
            dof = self._get_xedge_dof(i, jx, px)
            Sk.extend(dof)

        return Sk, px, py

    def get_stencil(self, elem, p):
        """Get the stencil for the given element index, ignoring any blanking that might exist"""

        Sk1, px1, py1 = self.get_xstencil(elem, p)
        Sk2, px2, py2 = self.get_ystencil(elem, p)

        if len(Sk1) > len(Sk2):
            return Sk1, px1, py1

        return Sk2, px2, py2

    def apply_element_scaling(self, elem, x, y):
        # Apply the shift to the local element coordinates
        x0 = self.x[elem % self.nx]
        y0 = self.y[elem // self.nx]

        m = (self.p - 1) // 2
        u = (x - x0) / (self.dx * (1 + m))
        v = (y - y0) / (self.dy * (1 + m))

        return u, v

    def get_basis_coef(self, elem):
        # Get the dof indicies that belong in the stencil
        sk, px, py = self.get_stencil(elem, self.p)

        # Get the coordinates
        xk = self.xpts[sk]
        yk = self.ypts[sk]
        u, v = self.apply_element_scaling(elem, xk, yk)

        # Evaluate the polynomial basis
        # Vk is dimension len(sk) x (p + 1)^2
        Vk = self.eval_poly_basis(u, v, px, py)

        print(
            "Element %3d |S|: %2d px: %2d py: %2d con. num. %10.4e"
            % (elem, len(sk), px, py, np.linalg.cond(Vk))
        )

        Ck = np.linalg.inv(Vk)

        # Compute the basis coefficients
        return Ck, sk, px, py

    def eval_element_basis(self, elem, x, y):
        # Get the coefficients for the basis functions
        Ck, sk, px, py = self.get_basis_coef(elem)

        # Apply the element scaling
        u, v = self.apply_element_scaling(elem, x, y)

        # Evaluate the basis functions
        V = self.eval_poly_basis(u, v, px, py)

        return np.dot(V, Ck), sk

    def eval_element_interp(self, elem, x, y, values):
        """Perform an interpolation over the given element with the specified values"""

        uk = np.zeros(len(x))
        if not self.blank[elem % self.nx, elem // self.nx]:
            N, sk = self.eval_element_basis(elem, x, y)

            for i, dof in enumerate(sk):
                uk += N[:, i] * values[dof]

        return uk


p = 5
nx = 15
ny = 15
lx = 1.0
ly = 1.0

blank = np.zeros((nx, ny))
blank[0, 0] = 1
blank[0, 1] = 1
blank[0, 2] = 1
blank[0, 3] = 1

blank[1, 0] = 1
blank[1, 2] = 1

blank[2, 0] = 1
blank[2, 1] = 1

blank[3, 0] = 1
blank[3, 1] = 1
blank[3, 2] = 1
blank[4, 2] = 1


cgd = CGD(nx=nx, ny=ny, lx=lx, ly=ly, p=p, blank=blank)

Sk_elem0, _, _ = cgd.get_stencil(elem=0, p=p)

xint = np.linspace(0, lx, nx + 1)
yint = np.linspace(0, ly, ny + 1)

levels = np.linspace(-1, 1, 100)

values = np.cos(2.0 * np.pi * cgd.xpts) * np.sin(3.0 * np.pi * cgd.ypts)

fig, ax = plt.subplots(1, 1)
for j in range(8):
    for i in range(8):
        elem = i + j * nx
        xv = np.linspace(xint[i], xint[i + 1], 25)
        yv = np.linspace(yint[j], yint[j + 1], 25)

        xcorners = [xint[i], xint[i + 1], xint[i + 1], xint[i], xint[i]]
        ycorners = [yint[j], yint[j], yint[j + 1], yint[j + 1], yint[j]]

        X, Y = np.meshgrid(xv, yv)
        x = X.flatten()
        y = Y.flatten()

        u = cgd.eval_element_interp(elem, x, y, values)

        ax.contour(X, Y, u.reshape(X.shape), levels=levels, linewidths=0.5)

        for dof in Sk_elem0:
            ax.plot(cgd.xpts[dof], cgd.ypts[dof], "o", color="red")

        if blank[i, j]:
            ax.fill(xcorners, ycorners, linewidth=0.5, color="red", alpha=0.2)
        else:
            ax.plot(xcorners, ycorners, linewidth=0.25, color="gray")


fig, ax = plt.subplots(1, 1)
values = np.zeros(cgd.ypts.shape)
values[cgd.nodes[3, 3]] = 1.0

for j in range(8):
    for i in range(8):
        elem = i + j * nx
        xv = np.linspace(xint[i], xint[i + 1], 25)
        yv = np.linspace(yint[j], yint[j + 1], 25)

        xcorners = [xint[i], xint[i + 1], xint[i + 1], xint[i], xint[i]]
        ycorners = [yint[j], yint[j], yint[j + 1], yint[j + 1], yint[j]]

        X, Y = np.meshgrid(xv, yv)
        x = X.flatten()
        y = Y.flatten()

        u = cgd.eval_element_interp(elem, x, y, values)

        ax.contour(X, Y, u.reshape(X.shape), levels=levels, linewidths=0.5)

        if blank[i, j]:
            ax.fill(xcorners, ycorners, linewidth=0.5, color="red", alpha=0.2)
        else:
            ax.plot(xcorners, ycorners, linewidth=0.25, color="gray")


plt.show()
