import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation, PillowWriter

nx = 101  # no of grid points
xmax = 2
xmin = -2

dx = (xmax - xmin) / (nx - 1)  # mesh size (slight fix)

x = np.linspace(xmin, xmax, nx)
print(x[100])
print(x)

c = 0.8                # advection speed
cfl = 0.2

dt = cfl * dx / abs(c)  # CFL condition
print(cfl)
print(dt)

t_end = 5.0
nt = int(t_end / dt)
t = np.linspace(0, nt * dt, nt + 1)
print(t[-1], nt)

u = np.zeros((nt + 1, nx))
u[0, :] = np.exp(-0.5 * (x / 0.4) ** 2)  # initial condition: Gaussian

uex = u[0, :].copy()

# Upwind scheme with PERIODIC boundary conditions
for n in range(nt):
    if c > 0:
        # interior points i = 1 ... nx-1
        for i in range(1, nx):
            u[n + 1, i] = u[n, i] - (c * dt / dx) * (u[n, i] - u[n, i - 1])
        # periodic BC at i = 0, use i-1 = nx-1
        u[n + 1, 0] = u[n, 0] - (c * dt / dx) * (u[n, 0] - u[n, nx - 1])
    else:
        # interior points i = 0 ... nx-2
        for i in range(0, nx - 1):
            u[n + 1, i] = u[n, i] - (c * dt / dx) * (u[n, i + 1] - u[n, i])
        # periodic BC at i = nx-1, use i+1 = 0
        u[n + 1, nx - 1] = u[n, nx - 1] - (c * dt / dx) * (u[n, 0] - u[n, nx - 1])

# Animation
fig, ax = plt.subplots()
line, = ax.plot(x, u[0, :])
ts = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.1, 1.1)

def update(frame):
    line.set_ydata(u[frame, :])
    ts.set_text(f't = {frame * dt:.3f}')
    return line, ts

anim = FuncAnimation(
    fig,
    update,
    frames=range(0, nt + 1, max(1, nt // 200)),
    interval=30,
    blit=True,
)

anim.save("advection_periodic.gif", writer=PillowWriter(fps=30))
plt.show()
