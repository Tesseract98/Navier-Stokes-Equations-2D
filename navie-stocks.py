import numpy as np
from matplotlib import pyplot as plt, animation, cm
from tqdm import tqdm


def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))
    return b


def pressure_poisson(p, dx, dy, b, iterations=50):
    for q in range(iterations):
        pn = p.copy()
        d2p = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) \
              / (2 * (dx ** 2 + dy ** 2))

        p[1:-1, 1:-1] = d2p - dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]

        # border condition on pressure
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        p[-1, :] = 0  # p = 0 at y = 2
    return p


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, gravity_x_y):
    b = np.zeros((ny, nx))

    # init condition on pressure
    # p[:ny // 2, nx // 4:] = 1
    # p[ny // 2:, nx // 4:] = 1
    # p[:ny // 2, nx // 4:] = -10

    res_u, res_v, res_p = [], [], []
    for _ in tqdm(range(nt)):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        res_p.append(np.copy(p))

        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) \
                        - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) \
                        - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) \
                        + nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy ** 2 *
                                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + gravity_x_y[0]

        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) \
                        - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) \
                        - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) \
                        + nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])) + gravity_x_y[1]

        # border condition on velocity field
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

        res_u.append(np.copy(u))
        res_v.append(np.copy(v))

    return res_u, res_v, res_p


def get_markers(dx, dy, nt, u, v):
    marker = np.zeros((ny, nx))
    marker[ny // 2:, 0: nx // 4] = 1
    marker[ny // 2 - 1, 0: nx // 4] = 0.5
    marker[ny // 2:, nx // 4] = 0.5
    markers = [marker]
    for z in tqdm(range(nt)):
        marker = np.copy(markers[-1])
        marker[1:-1, 1:-1] = marker[1:-1, 1:-1] - dt * (
                u[z][1:-1, 1:-1] * (marker[1:-1, 1:-1] - marker[1:-1, 0:-2]) / dx +
                v[z][1:-1, 1:-1] * (marker[1:-1, 1:-1] - marker[0:-2, 1:-1]) / dy)

        marker[marker < 0] = 0
        markers.append(marker)
    return markers


# def get_markers(dx, dy, nt, u, v):
#     marker = np.zeros((ny, nx))
#     marker[ny // 2:, 0: nx // 4] = 1
#     marker[ny // 2 - 1, 0: nx // 4] = 0.5
#     marker[ny // 2:, nx // 4] = 0.5
#     markers = [marker]
#     # for z in tqdm(range(nt)):
#     #     marker = np.copy(markers[-1])
#     #     # marker[1:-1, 1:-1] = marker[1:-1, 1:-1] - dt * (
#     #     #         u[z][1:-1, 1:-1] * (marker[1:-1, 1:-1] - marker[1:-1, 0:-2]) / dx +
#     #     #         v[z][1:-1, 1:-1] * (marker[1:-1, 1:-1] - marker[0:-2, 1:-1]) / dy)
#     for z in tqdm(range(nt)):
#         for i in range(1, len(marker) - 1):
#             for j in range(1, len(marker[i]) - 1):
#                 marker[i, j] = u[z][i, j] * v[z][i, j] * marker[i, j] \
#                                - (u[z][i, j] * marker[i, j] - u[z][i, j] * marker[1:-1, 0:-2]) / dx ** 2 \
#                                 + (v[z][1:-1, 1:-1] * marker[1:-1, 1:-1] - marker[0:-2, 1:-1]) / dy ** 2
#
#                 # marker[i, j] = marker[i, j] - dt * (u[z][i, j] * marker[i, j] - u[z][i, j] * marker[1:-1, 0:-2]) / dx +
#                 #                v[z][1:-1, 1:-1] * (marker[1:-1, 1:-1] - marker[0:-2, 1:-1]) / dy)
#                 marker[marker < 0] = 0
#         markers.append(marker)
#     return markers


def show_streamplot(X, Y, u, v, save_to_file=False):
    fig, axis = plt.subplots()
    axis.streamplot(X, Y, np.rot90(u[-2], k=2), np.rot90(v[-2], k=2))

    if save_to_file:
        plt.savefig("data/stream-lines.png")
    else:
        plt.show()


def show_quiver_animation(X, Y, u, v, p, save_to_file=False):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    # plotting the pressure field as a contour
    plt.contourf(X, Y, p[-1], alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    # plotting the pressure field outlines
    plt.contour(X, Y, p[-1], cmap=cm.viridis)

    output_step = 2

    def animate(i):
        # plotting velocity field
        return [plt.quiver(X[::output_step, ::output_step], Y[::output_step, ::output_step],
                           u[i][::output_step, ::output_step], v[i][::output_step, ::output_step])]

    anim = animation.FuncAnimation(fig, animate, frames=len(u), interval=1, blit=True, repeat=False)

    if save_to_file:
        plt.quiver(X[::output_step, ::output_step], Y[::output_step, ::output_step],
                   np.rot90(u[-1][::output_step, ::output_step], k=2),
                   np.rot90(v[-1][::output_step, ::output_step], k=2))
        plt.savefig("data/velocity-field.png")

        # watcher = lambda x, y: print(x)
        # anim.save('data/velocity-field.gif')
    else:
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


def show_animation_of_dam_destruction_return_markers(u, v, dx, dy, nt, save_to_file=False):
    frame_rate = 20
    frames = nt // frame_rate

    markers = get_markers(dx, dy, nt, u, v)[::frame_rate]

    def animate(i):
        im.set_array(markers[i][1:-1, 1:-1] > 0.6)
        # im.set_array(markers[i][1:-1, 1:-1])
        return [im]

    fig = plt.figure()
    im = plt.imshow(markers[0][1:-1, 1:-1] > 0.6, cmap='Blues', interpolation='spline36')
    # im = plt.imshow(markers[0][1:-1, 1:-1] > 0.6, cmap='Blues')
    # im = plt.imshow(markers[0][1:-1, 1:-1], cmap='PuBu')
    # im = plt.imshow(markers[0][1:-1, 1:-1], cmap='plasma', interpolation='spline36')
    # im = plt.imshow(-markers[0][1:-1, 1:-1], cmap='ocean', interpolation='spline36')
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=20, blit=True)

    if save_to_file:
        # plt.imshow(markers[0][1:-1, 1:-1] > 0.6, cmap='Blues', interpolation='spline36')
        # plt.savefig("data/start-dam.png")
        #
        # markers_len = len(markers)
        #
        # plt.imshow(markers[markers_len // 4][1:-1, 1:-1] > 0.6, cmap='Blues', interpolation='spline36')
        # plt.savefig("data/25t-dam.png")
        #
        # plt.imshow(markers[markers_len // 2][1:-1, 1:-1] > 0.6, cmap='Blues', interpolation='spline36')
        # plt.savefig("data/50t-dam.png")
        #
        # plt.imshow(markers[markers_len // 4 * 3][1:-1, 1:-1] > 0.6, cmap='Blues', interpolation='spline36')
        # plt.savefig("data/75t-dam.png")
        #
        # plt.imshow(markers[-1][1:-1, 1:-1] > 0.6, cmap='Blues', interpolation='spline36')
        # plt.savefig("data/end-dam.png")

        anim.save('data/dam-destruction.gif')
    else:
        plt.show()


def show_animation_of_pressure_contour(u, v, dx, dy, nt, save_to_file=False):
    pressures = []
    markers = get_markers(dx, dy, nt, u, v)
    for i in range(nt):
        press = np.copy(p[i])
        press[markers[i] < 0.5] = 0
        pressures.append(press)

    fig, axis = plt.subplots(3, 1)
    axis[0].contour(np.rot90(pressures[2].T), cmap=cm.viridis)
    axis[1].contour(np.rot90(pressures[len(pressures) // 2].T), cmap=cm.viridis)
    axis[2].contour(np.rot90(pressures[-1].T), cmap=cm.viridis)

    if save_to_file:
        plt.savefig('data/watter-pressure.png')
    else:
        plt.show()


def get_modeling_results(save_to_file: bool = False):
    if save_to_file:
        import os

        if not os.path.exists("data"):
            os.makedirs("data")

    print("show dam destruction")
    show_animation_of_dam_destruction_return_markers(u, v, dx, dy, nt, save_to_file=save_to_file)
    print("show watter pressure")
    show_animation_of_pressure_contour(u, v, dx, dy, nt, save_to_file=save_to_file)
    print("show velocity field")
    show_quiver_animation(X, Y, u, v, p, save_to_file=save_to_file)
    print("show stream direction")
    show_streamplot(X, Y, u, v, save_to_file=save_to_file)


if __name__ == '__main__':
    save_result_to_file = True

    # resolution
    nx = 100
    ny = 100

    # density
    rho = 1

    # viscosity
    # nu = 0.05
    nu = 0.1

    # steps by t
    nt = 3000

    # delta t
    dt = .001

    gravity_x_y = (0, 0.01)

    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, gravity_x_y)

    get_modeling_results(save_result_to_file)
