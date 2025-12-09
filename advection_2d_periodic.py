import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation, PillowWriter

# =====================
# Grid and parameters
# =====================
nx, ny = 101, 101  # grid points in x and y

xmax, xmin = 2.0, -2.0
ymax, ymin = 2.0, -2.0

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(x, y, indexing='ij')

ax = 0.8   # advection speed in x
ay = 0.5   # advection speed in y

cfl = 0.2
dt = cfl * min(dx / abs(ax), dy / abs(ay))

print("dx, dy =", dx, dy)
print("dt =", dt)

t_end = 5.0
nt = int(t_end / dt)
t = np.linspace(0, nt * dt, nt + 1)
print("t_final (approx) =", t[-1], "nt =", nt)

# =====================
# Initial condition
# =====================
u = np.zeros((nt + 1, nx, ny))
u[0, :, :] = np.exp(-0.5 * ((X / 0.4) ** 2 + (Y / 0.4) ** 2))  # 2D Gaussian

Cx = ax * dt / dx
Cy = ay * dt / dy

# ==========================================
# Upwind scheme with PERIODIC BC in 2D
# Assume ax > 0, ay > 0 (like c > 0 case)
# ==========================================
for n in range(nt):
    # interior points: i = 1..nx-1, j = 1..ny-1
    for i in range(1, nx):
        for j in range(1, ny):
            u[n + 1, i, j] = (
                u[n, i, j]
                - Cx * (u[n, i, j] - u[n, i - 1, j])
                - Cy * (u[n, i, j] - u[n, i, j - 1])
            )

    # periodic in x at i = 0, for j = 1..ny-1 (use i-1 = nx-1)
    for j in range(1, ny):
        u[n + 1, 0, j] = (
            u[n, 0, j]
            - Cx * (u[n, 0, j] - u[n, nx - 1, j])
            - Cy * (u[n, 0, j] - u[n, 0, j - 1])
        )

    # periodic in y at j = 0, for i = 1..nx-1 (use j-1 = ny-1)
    for i in range(1, nx):
        u[n + 1, i, 0] = (
            u[n, i, 0]
            - Cx * (u[n, i, 0] - u[n, i - 1, 0])
            - Cy * (u[n, i, 0] - u[n, i, ny - 1])
        )

    # corner (i=0, j=0): periodic in both x and y
    u[n + 1, 0, 0] = (
        u[n, 0, 0]
        - Cx * (u[n, 0, 0] - u[n, nx - 1, 0])
        - Cy * (u[n, 0, 0] - u[n, 0, ny - 1])
    )

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
ax_fig.set_title('2D Advection (Upwind, Periodic BC)')

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

anim.save("advection_2d_periodic.gif", writer=PillowWriter(fps=30))
plt.show()
