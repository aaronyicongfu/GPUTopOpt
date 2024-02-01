import numpy as np
import matplotlib.pyplot as plt


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


def demo_poly_fit_2d_test_edge_interpolation(p, pts):
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

    return


if __name__ == "__main__":
    p = 5

    # This is a good example
    pts = np.array([(i, j) for i in range(p + 1) for j in range(p + 1)])
    pts[4] = (-1, 4)
    pts[5] = (0, 6)
    pts[6] = (1, -2)
    pts[30] = (7, -2)
    demo_poly_fit_2d_test_edge_interpolation(p, pts)

    # This is a bad example
    pts = np.array([(i, j) for i in range(p + 1) for j in range(p + 1)])
    pts[4] = (-1, 0)
    demo_poly_fit_2d_test_edge_interpolation(p, pts)

    plt.show()
