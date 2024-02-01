import numpy as np
import matplotlib.pyplot as plt


class CGDInterp:
    def __init__(self, xpts, p):
        self.xpts = xpts  # 1d mesh grids
        self.npts = len(xpts)
        self.nelems = self.npts - 1
        self.p = p  # polynomial degree, must be odd
        self.m = (p - 1) // 2
        self.dof_augmented = np.zeros(
            self.npts + 2 * self.m
        )  # regular dof plus halo nodes
        return

    def lagrange_polynomials(self, xs):
        """
        Args:
            xs: array, nodes that defines the Lagrange bases

        Return:
            funcs: list of callables, funcs[i](x) evaluates the Lagrange basis l_i(x)
        """
        funcs = []
        for j in range(len(xs)):

            def lj(x, j=j):
                ljx = 1.0
                for m in range(len(xs)):
                    if m != j:
                        ljx *= (x - xs[m]) / (xs[j] - xs[m])
                return ljx

            funcs.append(lj)
        return funcs

    # def get_stencils(self, elem):
    #     """
    #     Give the element index, return dof indices
    #     """
    #     m = (p - 1) // 2
    #     if elem < m:
    #         return list(range(0, p))
    #     elif elem >= self.nelems - m:
    #         return list(range(self.nelems - p, self.nelems))
    #     else:
    #         return list(range(elem - m, elem + m + 1))

    def get_stencils_including_halo(self, elem):
        return list(range(elem - self.m, elem + self.m + 1))


p = 3
nnodes = 20
h = 0.5
xpts = np.linspace(0, h * (nnodes - 1), nnodes)
wts = np.cos(xpts)

N = 101  # For visualization
xpts_ = np.linspace(0, h * (nnodes - 1), N)


plt.plot(xpts, wts)
plt.show()
