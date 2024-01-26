import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lagrange_polynomials_1d(xs):
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


def lagrange_polynomials_2d(xs1, xs2):
    """
    2D Lagrange basis functions via tensor product

    Args:
        xs: array, nodes that defines the Lagrange bases

    Return:
        funcs: list of callables, funcs[i][j](x, y) evaluates l_i(x) * l_j(y),
               where l_i and l_j are Lagrange bases
    """
    li = lagrange_polynomials_1d(xs1)
    lj = lagrange_polynomials_1d(xs2)

    funcs = []
    for i in range(len(li)):
        funcs.append([])
        for j in range(len(lj)):
            funcs[i].append(lambda x, y, i=i, j=j: li[i](x) * lj[j](y))
    return funcs


def get_gd_shape_fun_1d(p=3):
    """
    Return:
        shape_fun: callable, shape_fun(x) is the shape function value,
                   where x is scalar or 1d array, shape_fun(x) != 0 for
                   -(p + 1) / 2 <= x <= (p + 1) / 2
    """
    assert p % 2 == 1
    pts = np.linspace(-(p + 1) // 2, (p + 1) // 2, p + 2)

    bases = []
    for i in range(p + 1):
        nodes = np.linspace(-p, 0, p + 1) + i
        bases.append(lagrange_polynomials_1d(nodes))

    def shape_fun(x):
        ret = np.zeros_like(x)
        for i, (start, end) in enumerate(zip(pts[:-1], pts[1:])):
            ret += bases[i][p - i](x) * (x >= start) * (x < end)

        return ret

    return shape_fun


def get_gd_shape_fun_2d(p=3):
    """
    Return:
        shape_fun: callable, shape_fun(x, y) is the shape function value,
                   where x, y are scalar or 1d arrays, shape_fun(x, y) != 0 for
                   -(p + 1) / 2 <= x <= (p + 1) / 2
    """
    assert p % 2 == 1
    pts = np.linspace(-(p + 1) // 2, (p + 1) // 2, p + 2)

    bases = []
    for i in range(p + 1):
        bases.append([])
        for j in range(p + 1):
            nodes1 = np.linspace(-p, 0, p + 1) + i
            nodes2 = np.linspace(-p, 0, p + 1) + j
            bases[i].append(lagrange_polynomials_2d(nodes1, nodes2))

    def shape_fun(x, y):
        ret = np.zeros_like(x)
        for i, (start1, end1) in enumerate(zip(pts[:-1], pts[1:])):
            for j, (start2, end2) in enumerate(zip(pts[:-1], pts[1:])):
                ret += (
                    bases[i][j][p - i][p - j](x, y)
                    * (x >= start1)
                    * (x < end1)
                    * (y >= start2)
                    * (y < end2)
                )

        return ret

    return shape_fun


def demo_plot_1d_lagrange_polys(p=3):
    pts = list(range(p + 1))
    ls = lagrange_polynomials_1d(pts)
    x = np.linspace(0, p, 201)

    fig, ax = plt.subplots()
    for i, l in enumerate(ls):
        ax.plot(x, l(x), label="basis %d" % i)

    plt.legend()
    plt.grid()
    plt.show()
    return


def demo_plot_2d_lagrange_polys(p=3, i=0, j=0):
    pts = list(range(p + 1))

    funcs = lagrange_polynomials_2d(pts, pts)

    h = 201
    x = np.linspace(0, p, h)
    y = np.linspace(0, p, h)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    vals = funcs[i][j](x, y)
    ax.plot_surface(x, y, vals)
    ax.set_title("p=%d, basis (%d, %d)" % (p, i, j))

    plt.show()

    return


def demo_plot_high_order_1d_gd_shape_function(p=3):
    gd_shape_fun = get_gd_shape_fun_1d(p)

    x = np.linspace(-(p + 1) / 2, (p + 1) / 2, 201)
    y = gd_shape_fun(x)

    plt.plot(x, y)
    plt.title("p=%d" % p)
    plt.grid()
    plt.show()
    return


def demo_plot_high_order_2d_gd_shape_function(p=3):
    shape = get_gd_shape_fun_2d(p)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    h = 401
    x = np.linspace(-(p + 1) / 2, (p + 1) / 2, h)
    x, y = np.meshgrid(x, x)
    z = shape(x, y)
    ax.contour3D(x, y, z, levels=np.linspace(-0.2, 1.0, 121))
    ax.set_title("p=%d" % p)
    # ax.plot_surface(x, y, z)
    plt.show()
    return


if __name__ == "__main__":
    # demo_plot_1d_lagrange_polys(p=5)
    # demo_plot_2d_lagrange_polys(p=5, i=3, j=3)
    # demo_plot_high_order_1d_gd_shape_function(p=7)
    demo_plot_high_order_2d_gd_shape_function(p=3)
    pass
