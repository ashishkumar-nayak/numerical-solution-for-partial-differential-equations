import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#grid 
nx,ny=101,101

xmax,xmin=2.0,-2.0
ymax,ymin=2.0,-2.0

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)

ax = 0.8   # velocity in x
ay = 0.5   # velocity in y

cfl = 0.2
dt = cfl * min(dx / abs(ax), dy / abs(ay))

t_end = 5.0
nt = int(t_end / dt)

u = np.zeros((nt + 1, nx, ny))
u[0, :, :] = np.exp(-0.5 * ((X / 0.4) ** 2 + (Y / 0.4) ** 2))

Cx = ax * dt / dx
Cy = ay * dt / dy

# Time stepping
for n in range(nt):
    for i in range(1, nx):
        for j in range(1, ny):
            u[n+1, i, j] = (
                u[n, i, j]
                - Cx * (u[n, i, j] - u[n, i-1, j])
                - Cy * (u[n, i, j] - u[n, i, j-1])
            )

    # Dirichlet boundaries
    u[n+1, 0, :] = u[0, 0, :]
    u[n+1, :, 0] = u[0, :, 0]

# --- Plot & Animation ---
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
ax_fig.set_title('2D Advection (Upwind)')

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

anim.save("advection_2d.gif", writer=PillowWriter(fps=30))
plt.show()
