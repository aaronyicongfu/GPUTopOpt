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


def polynomials_fit_1d(p, pts):
    """
    Args:
        p: polynomial degree
        pts: list of x-coordinates
    """
    Nk = len(pts)
    Np = 1 + p
    if Nk != Np:
        print("Nk != Np (%d, %d), can't invert Vk" % (Nk, Np))

    Vk = np.zeros((Nk, Np))
    for j in range(Np):
        Vk[:, j] = pts**j

    Ck = np.linalg.inv(Vk)
    print("condition number of Vk:", np.linalg.cond(Vk))

    funcs = []
    for i in range(Nk):

        def phi(x, i=i):
            ret = 0.0
            for j in range(Np):
                ret += Ck[j, i] * x**j
            return ret

        funcs.append(phi)
    return funcs


def polynomials_fit_2d(p, pts):
    """
    Args:
        p: polynomial order along one dimension
        pts: list of pts
    """

    Nk = len(pts)
    Np_1d = 1 + p
    Np = Np_1d**2
    if Nk != Np:
        print("Nk != Np (%d, %d), can't invert Vk" % (Nk, Np))

    Vk = np.zeros((Nk, Np))
    for i, xy in enumerate(pts):
        x = xy[0]
        y = xy[1]
        xpows = [x**j for j in range(Np_1d)]
        ypows = [y**j for j in range(Np_1d)]
        for j in range(Np_1d):
            for k in range(Np_1d):
                idx = j * Np_1d + k
                Vk[i, idx] = xpows[j] * ypows[k]

    Ck = np.linalg.inv(Vk)
    print("condition number of Vk:", np.linalg.cond(Vk))

    funcs = []
    for i in range(Nk):

        def phi(x, y, i=i):
            xpows = [x**j for j in range(Np_1d)]
            ypows = [y**j for j in range(Np_1d)]
            ret = 0.0
            for j in range(Np_1d):
                for k in range(Np_1d):
                    idx = j * Np_1d + k
                    ret += Ck[idx, i] * xpows[j] * ypows[k]
            return ret

        funcs.append(phi)

    return funcs, Vk


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


def demo_poly_fit_1d(p=3):
    xs = np.linspace(0, p, p + 1)
    funcs_lag = lagrange_polynomials_1d(xs)
    funcs_fit = polynomials_fit_1d(p, xs)

    x = np.linspace(0, p, 201)

    fig, ax = plt.subplots(ncols=2, figsize=(9.6, 4.8))
    for i, fun in enumerate(funcs_lag):
        ax[0].plot(x, fun(x), label="basis %d" % i)

    for i, fun in enumerate(funcs_fit):
        ax[1].plot(x, fun(x), label="basis %d" % i)

    ax[0].set_title("Lagrange polynomials")
    ax[1].set_title("Polynomials by fit")

    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    plt.show()
    return


def demo_poly_fit_2d(p=3, i=0, j=0):
    pts = list(range(p + 1))
    funcs_lag = lagrange_polynomials_2d(pts, pts)

    pts2 = [(i, j) for i in range(p + 1) for j in range(p + 1)]
    # pts2[-1] = (pts2[-1][0] + 1, pts2[-1][1] + 1)
    funcs_fit, Vk = polynomials_fit_2d(p, pts2)

    x = np.linspace(0, p, 201)
    y = np.linspace(0, p, 201)
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(ncols=2, figsize=(9.6, 4.8))
    ax[0].contour(x, y, funcs_lag[i][j](x, y), levels=100)
    ax[1].contour(x, y, funcs_fit[(p + 1) * i + j](x, y), levels=100)

    ax[0].set_title("Lagrange polynomials (%d, %d)" % (i, j))
    ax[1].set_title("Polynomials by fit")

    plt.show()

    return


def demo_poly_fit_2d_test_kronecker(p=3):
    pts = [(i, j) for i in range(p + 1) for j in range(p + 1)]
    funcs_fit, Vk = polynomials_fit_2d(p, pts)

    vals = np.zeros((len(funcs_fit), len(pts)))
    for i in range(len(funcs_fit)):
        for j in range(len(pts)):
            vals[i, j] = funcs_fit[i](pts[j][0], pts[j][1])

    plt.matshow(vals)
    plt.show()

    return


def demo_poly_fit_2d_test_edge_interpolation():
    p = 5
    pts = np.array([(i, j) for i in range(p + 1) for j in range(p + 1)])
    # pts[4] = (-1, 4)
    # pts[5] = (0, 6)
    # pts[6] = (1, -2)
    # pts[30] = (7, -2)

    pts[4] = (-1, 0)

    funcs_fit, Vk = polynomials_fit_2d(p, pts)

    vals = np.zeros((len(funcs_fit), len(pts)))
    for i in range(len(funcs_fit)):
        for j in range(len(pts)):
            vals[i, j] = funcs_fit[i](pts[j][0], pts[j][1])

    fig, ax = plt.subplots(ncols=4, figsize=(16.0, 5.0))
    ax[0].set_title(
        "basis functions evaluated at dof nodes\ncondition number of Vk: %.4e"
        % np.linalg.cond(Vk)
    )
    ax[0].set_xlabel("basis function")
    ax[0].set_ylabel("node")
    ax[0].matshow(vals)

    for pt in pts:
        ax[1].plot(pt[0], pt[1], "ro")
    ax[1].set_title("Degrees of freedom (p=%d)" % p)
    ax[1].grid()

    x0 = 4
    ymin = pts[:, 1].min()
    ymax = pts[:, 1].max()

    np.random.seed(0)
    y = np.linspace(ymin, ymax, 201)
    x = np.ones_like(y) * x0
    z1 = np.zeros_like(y)
    z2 = np.zeros_like(y)
    for i, fun in enumerate(funcs_fit):
        if pts[i][0] == x0:
            z1 += np.random.rand() * fun(x, y)
        else:
            z2 += np.random.rand() * fun(x, y)
            ax[1].plot(pts[i][0], pts[i][1], "bo")
    ax[1].plot([x0, x0], [ymin, ymax], "--r")

    ax[2].plot(y, z1)
    ax[2].set_title(
        "Interpolation along red dash line\nusing red nodes and random weights"
    )

    ax[3].plot(y, z2)
    ax[3].set_title(
        "Interpolation along red dash line\nusing blue nodes and random weights"
    )

    plt.show()

    return


if __name__ == "__main__":
    # demo_plot_1d_lagrange_polys(p=4)
    # demo_plot_2d_lagrange_polys(p=2, i=0, j=0)
    # demo_plot_high_order_1d_gd_shape_function(p=5)
    # demo_plot_high_order_2d_gd_shape_function(p=3)
    # demo_poly_fit_1d(p=3)
    # demo_poly_fit_2d(p=3, i=0, j=0)
    # demo_poly_fit_2d_test_kronecker(p=3)
    demo_poly_fit_2d_test_edge_interpolation()
    pass
