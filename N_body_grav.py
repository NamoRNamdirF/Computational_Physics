import matplotlib.pyplot as plt
from torch import Tensor
from numpy import dot, hstack


# import timeit
# code_to_test = """
def get_a(cord, mass, G):
    x = cord[:, 0:1]
    y = cord[:, 1:2]
    z = cord[:, 2:3]

    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    r = (dx ** 2 + dy ** 2 + dz ** 2 + 0.1) ** (-0.5)
    ax = dot(G * (dx * r), mass)
    ay = dot(G * (dy * r), mass)
    az = dot(G * (dz * r), mass)
    # соединяет массивы по горизонтали
    a = hstack((ax, ay, az))
    return a
# def get_a(pos, mass, G):
#
#     N = pos.shape[0]
#     a = zeros((N, 3))
#
#     for i in range(N):
#         for j in range(N):
#             dx = pos[j, 0] - pos[i, 0]
#             dy = pos[j, 1] - pos[i, 1]
#             dz = pos[j, 2] - pos[i, 2]
#             inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + 1)**(-1.5)
#             print(inv_r3)
#             a[i, 0] += G * (dx * inv_r3) * mass[j]
#             a[i, 1] += G * (dy * inv_r3) * mass[j]
#             a[i, 2] += G * (dz * inv_r3) * mass[j]
#
#     return a


def main():
    t = 0
    dt: float = 0.1
    G = 1.0
    mass = Tensor([[0.33],
                  [0.33],
                  [0.33]])
    cord = Tensor([[0.286031, 0., 0.],
                  [0., 0., 0.],
                  [-0.286031, 0., 0.]])
    v = Tensor([[-0.749442/2, -1.15078, 0.],
                [0.749442, 1.15078, 0.],
                [-0.749442/2, -1.15078, 0.]])

    acc = get_a(cord, mass, G)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    for i in range(1000):
        # Алгоритм верле:
        v += acc * dt
        cord += v * dt + (acc * dt ** 2) / 2
        acc = get_a(cord, mass, G)
        v += acc * dt / 2
        t += dt
        #############################################################
        plt.sca(ax1)
        plt.scatter(cord[:, 0], cord[:, 1], cord[:, 2], s=1, color='black')
        plt.pause(0.001)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
# """
# elapsed_time = timeit.timeit(code_to_test, number=100)/100
# print(elapsed_time)
