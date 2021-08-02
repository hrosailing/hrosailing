import hrosailing.processing.pipelinecomponents as pc
import matplotlib.pyplot as plt
import numpy as np


def visualize_ball():
    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    x, y = np.meshgrid(x, y)
    x = np.ravel(x)
    y = np.ravel(y)

    ball = pc.Ball(d=2, radius=1)
    mask = ball.is_contained_in(np.column_stack((x, y)))
    x = x[mask]
    y = y[mask]

    plt.plot(x, y, ms=0.6, color='blue', ls='', marker='o')
    plt.show()


def visualize_ellipsoid():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    x, y = np.meshgrid(x, y)
    x = np.ravel(x)
    y = np.ravel(y)

    ellipsoid = pc.Ellipsoid(d=2, lin_trans=np.array([[2, 1], [1, 2]]), radius=1)

    mask = ellipsoid.is_contained_in(np.column_stack((x, y)))
    x = x[mask]
    y = y[mask]

    plt.plot(x, y, ms=0.6, color='blue', ls='', marker='o')
    plt.show()


def visualize_cuboid():
    x = np.linspace(-1.5, 1.5, 1000)
    y = np.linspace(-1.5, 1.5, 1000)

    x, y = np.meshgrid(x, y)
    x = np.ravel(x)
    y = np.ravel(y)

    cuboid = pc.Cuboid()

    mask = cuboid.is_contained_in(np.column_stack((x, y)))
    x = x[mask]
    y = y[mask]

    plt.plot(x, y, ms=0.6, color='blue', ls='', marker='o')
    plt.show()
