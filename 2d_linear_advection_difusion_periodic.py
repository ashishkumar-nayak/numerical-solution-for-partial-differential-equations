import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

nx, ny = 101, 101

xmax, xmin = 2.0, -2.0
ymax, ymin = 2.0, -2.0

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(x, y, indexing='ij')

ax = 0.8
ay = 0.5
mu = 0.01

cfl = 0.2
dt = cfl * min(dx / abs(ax), dy / abs(ay))

t_end = 5.0
nt = int(t_end / dt)
t = np.linspace(0, nt * dt, nt + 1)

u = np.zeros((nt + 1, nx, ny))
u[0, :, :] = np.exp(-0.5 * ((X / 0.4) ** 2 + (Y / 0.4) ** 2))

Cx = ax * dt / dx
Cy = ay * dt / dy
Dx = mu * dt / dx**2
Dy = mu * dt / dy**2

for n in range(nt):
    for i in range(1, nx):
        for j in range(1, ny):
            adv = (
                - Cx * (u[n, i, j] - u[n, i - 1, j])
                - Cy * (u[n, i, j] - u[n, i, j - 1])
            )
            diff = (
                Dx * (u[n, (i + 1) % nx, j] + u[n, i - 1, j] - 2 * u[n, i, j]) +
                Dy * (u[n, i, (j + 1) % ny] + u[n, i, j - 1] - 2 * u[n, i, j])
            )
            u[n + 1, i, j] = u[n, i, j] + adv + diff

    for j in range(1, ny):
        adv = (
            - Cx * (u[n, 0, j] - u[n, nx - 1, j])
            - Cy * (u[n, 0, j] - u[n, 0, j - 1])
        )
        diff = (
            Dx * (u[n, 1, j] + u[n, nx - 1, j] - 2 * u[n, 0, j]) +
            Dy * (u[n, 0, (j + 1) % ny] + u[n, 0, j - 1] - 2 * u[n, 0, j])
        )
        u[n + 1, 0, j] = u[n, 0, j] + adv + diff

    for i in range(1, nx):
        adv = (
            - Cx * (u[n, i, 0] - u[n, i - 1, 0])
            - Cy * (u[n, i, 0] - u[n, i, ny - 1])
        )
        diff = (
            Dx * (u[n, (i + 1) % nx, 0] + u[n, i - 1, 0] - 2 * u[n, i, 0]) +
            Dy * (u[n, i, 1] + u[n, i, ny - 1] - 2 * u[n, i, 0])
        )
        u[n + 1, i, 0] = u[n, i, 0] + adv + diff

    adv = (
        - Cx * (u[n, 0, 0] - u[n, nx - 1, 0])
        - Cy * (u[n, 0, 0] - u[n, 0, ny - 1])
    )
    diff = (
        Dx * (u[n, 1, 0] + u[n, nx - 1, 0] - 2 * u[n, 0, 0]) +
        Dy * (u[n, 0, 1] + u[n, 0, ny - 1] - 2 * u[n, 0, 0])
    )
    u[n + 1, 0, 0] = u[n, 0, 0] + adv + diff

fig, ax_fig = plt.subplots()

im = ax_fig.imshow(
    u[0, :, :].T,
    extent=[xmin, xmax, ymin, ymax],
    origin='lower',
    vmin=-0.1,
    vmax=1.1,
    aspect='auto'
)

cbar = fig.colorbar(im, ax=ax_fig)
cbar.set_label("u(x, y, t)")

ts = ax_fig.text(0.05, 0.9, '', transform=ax_fig.transAxes, color='white')
ax_fig.set_xlabel('x')
ax_fig.set_ylabel('y')
ax_fig.set_title('2D Advection-Diffusion (Periodic BC)')

def update(frame):
    im.set_data(u[frame, :, :].T)
    ts.set_text(f't = {frame * dt:.3f}')
    return im, ts

anim = FuncAnimation(
    fig,
    update,
    frames=range(0, nt + 1, max(1, nt // 200)),
    interval=30,
    blit=True,
)

anim.save("advection_diffusion_2d_periodic.gif", writer=PillowWriter(fps=30))
plt.show()

