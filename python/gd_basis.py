import numpy as np
import matplotlib.pylab as plt

def get_stencil(k, p, inter):
    n = len(inter) - 1

    # Set the initial stencil
    Sk = [k, k + 1]

    # Expand the stencil to meet the criteria that len(Sk) == p + 1
    while len(Sk) < p + 1:
        if Sk[0] <= 0:
            Sk = Sk + [Sk[-1] + 1, Sk[-1] + 2]
        elif Sk[-1] >= n:
            Sk = [Sk[0] - 2, Sk[0] - 1] + Sk
        else:
            Sk = [Sk[0] - 1] + Sk + [Sk[-1] + 1]

    return Sk


def make_basis_plot(ax, p, inter):
    for k in range(n):
        Sk = get_stencil(k, p, inter)

        # Points for plotting
        x = np.linspace(inter[k], inter[k + 1])

        # Evaluate the Lagrange polynomials passing through the entries of the stencil
        for i in range(p + 1):
            poly = np.ones(len(x))

            for j in range(p + 1):
                if i != j:
                    poly *= (x - inter[Sk[j]]) / (inter[Sk[i]] - inter[Sk[j]])

            ax.plot(x, poly)


def make_interp_plot(ax, p, inter, data):
    for k in range(n):
        Sk = get_stencil(k, p, inter)

        # Points for plotting
        x = np.linspace(inter[k], inter[k + 1])
        f = np.zeros(len(x))

        # Evaluate the Lagrange polynomials passing through the entries of the stencil
        for i in range(p + 1):
            poly = np.ones(len(x))

            for j in range(p + 1):
                if i != j:
                    poly *= (x - inter[Sk[j]]) / (inter[Sk[i]] - inter[Sk[j]])

            f += poly * data[Sk[i]]

        ax.plot(x, f)


# Number of elements
n = 10

# Intervals
inter = np.linspace(0, 10.0, n + 1)

# Create the basis plot
fig, ax = plt.subplots(4, 1)

# polynomial degree
make_basis_plot(ax[0], 1, inter)
make_basis_plot(ax[1], 3, inter)
make_basis_plot(ax[2], 5, inter)
make_basis_plot(ax[3], 7, inter)

# Create interpolation plot
fig, ax = plt.subplots(4, 1)

data = np.sin(4 * np.pi * inter / 10.0)

make_interp_plot(ax[0], 1, inter, data)
make_interp_plot(ax[1], 3, inter, data)
make_interp_plot(ax[2], 5, inter, data)
make_interp_plot(ax[3], 7, inter, data)

plt.show()
