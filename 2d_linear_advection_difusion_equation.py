import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
#grid
nx,ny=101,101
xmax, xmin = 2.0, -2.0
ymax, ymin = 2.0, -2.0
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

X, Y = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)

Vx=0.8
Vy=0.5
mu=0.01

mu = 0.01   # diffusion coefficient

cfl_adv=0.2
dt_adv_x = cfl_adv * dx / (abs(Vx) + 1e-16)

dt_adv_y = cfl_adv * dy / (abs(Vy) + 1e-16)

dt_diff = 1.0 / (2.0 * mu * (1.0 / dx**2 + 1.0 / dy**2) + 1e-16)

dt = min(dt_adv_x, dt_adv_y, dt_diff)

print("dt adv x, adv y, diff =", dt_adv_x, dt_adv_y, dt_diff)
print("chosen dt =", dt)


t_end=5.0
nt=int(t_end/dt)
print("nt=", nt)

u = np.zeros((nt + 1, nx, ny))
sigma = 0.4
u0 = np.exp(-0.5 * ((X / sigma) ** 2 + (Y / sigma) ** 2))
u[0, :, :] = u0.copy()


Cx = Vx * dt / dx
Cy = Vy * dt / dy
alpha_x = mu * dt / dx**2
alpha_y = mu * dt / dy**2


# helpers for upwind coefficients (scalar Vx, Vy; sign-robust)
Vx_plus = max(Vx, 0.0)
Vx_minus = min(Vx, 0.0)
Vy_plus = max(Vy, 0.0)
Vy_minus = min(Vy, 0.0)  ## mainly we dont know the function in case non monoton#ic or monotonic


for n in range(nt):
    # interior points: i = 1..nx-2, j = 1..ny-2
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # upwind advection (compact sign-robust form)
            adv_x = (Vx_plus * (u[n, i, j] - u[n, i-1, j]) +
                     Vx_minus * (u[n, i+1, j] - u[n, i, j])) / dx
            adv_y = (Vy_plus * (u[n, i, j] - u[n, i, j-1]) +
                     Vy_minus * (u[n, i, j+1] - u[n, i, j])) / dy

            # diffusion (second-order central)
            diff = mu * ((u[n, i+1, j] - 2.0 * u[n, i, j] + u[n, i-1, j]) / dx**2 +
                         (u[n, i, j+1] - 2.0 * u[n, i, j] + u[n, i, j-1]) / dy**2)

            u[n+1, i, j] = u[n, i, j] - dt * (adv_x + adv_y) + dt * diff

    # Dirichlet boundaries: keep boundary fixed to initial boundary values (like your original)
    u[n+1, 0, :] = u0[0, :].copy()        # left edge x = xmin
    u[n+1, -1, :] = u0[-1, :].copy()      # right edge x = xmax
    u[n+1, :, 0] = u0[:, 0].copy()        # bottom edge y = ymin
    u[n+1, :, -1] = u0[:, -1].copy()      # top edge y = ymax


fig, ax_fig = plt.subplots(figsize=(6,5))

im = ax_fig.imshow(
    u[0, :, :].T,
    extent=[xmin, xmax, ymin, ymax],
    origin='lower',
    vmin=0.0,
    vmax=1.1,
    aspect='auto'
)

cbar = fig.colorbar(im, ax=ax_fig)
cbar.set_label("u(x, y, t)")

ts = ax_fig.text(0.05, 0.9, '', transform=ax_fig.transAxes, color='white')
ax_fig.set_xlabel('x')
ax_fig.set_ylabel('y')
ax_fig.set_title('2D Advection-Diffusion (Upwind + Central Diffusion)')

# reduce number of frames for gif to keep file size reasonable
frame_indices = list(range(0, nt + 1, max(1, nt // 200)))

def update(frame):
    im.set_data(u[frame, :, :].T)
    ts.set_text(f't = {frame * dt:.3f}')
    return im, ts

anim = FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=30,
    blit=True,
)

anim.save("advection_diffusion_2d.gif", writer=PillowWriter(fps=30))
plt.show()

