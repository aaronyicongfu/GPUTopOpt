import numpy as np
import matplotlib.pyplot as plt


def lagrange_polynomials_1d(nodes):
    funcs = []
    for j in range(len(nodes)):

        def lj(x, j=j):
            ljx = 1.0
            for m in range(len(nodes)):
                if m != j:
                    ljx *= (x - nodes[m]) / (nodes[j] - nodes[m])
            return ljx

        funcs.append(lj)
    return funcs


def test_lagrange_polynomial(nnodes=2):
    fig, ax = plt.subplots()
    nodes = np.linspace(0, 1, nnodes)
    x = np.linspace(-0.0, 1.0, 201)
    bases = lagrange_polynomials_1d(nodes)
    for i, basis in enumerate(bases):
        ax.plot(x, basis(x), "--", label="basis %d" % i)

    np.random.seed(0)
    coeffs = np.random.rand(len(nodes))
    y = np.zeros_like(x)
    for i, basis in enumerate(bases):
        y += coeffs[i] * basis(x)

    # ax.plot(x, y, "-", label="function")

    plt.legend()
    plt.grid()
    plt.show()
    return


if __name__ == "__main__":
    test_lagrange_polynomial(nnodes=11)
