import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from matplotlib.animation import FuncAnimation, PillowWriter

import matplotlib.pyplot as plt

nx=101
xmax=2.0
xmin=-2.0

dx = (xmax - xmin) / (nx - 1)   # mesh size

x = np.linspace(xmin, xmax, nx)

u0 = np.sin(np.pi * x) #initialzation

cfl=0.2

umax = np.max(np.abs(u0))
dt = cfl * dx / umax
t_end = 1.5
nt = int(t_end / dt)
t = np.linspace(0, nt * dt, nt + 1)


u = np.zeros((nt + 1, nx))

u[0, :] = u0

uex = u[0, :].copy()

for n in range(0, nt):
    # interior points: choose upwind based on sign of u[n, i]
    for i in range(1, nx - 1):
        un = u[n, i]
        if un > 0.0:
            # backward difference (flow to the right)
            dudx = (u[n, i] - u[n, i - 1]) / dx
        else:
            # forward difference (flow to the left)
            dudx = (u[n, i + 1] - u[n, i]) / dx

        u[n + 1, i] = u[n, i] - dt * un * dudx

    # Dirichlet boundary conditions (keep fixed values)
    u[n + 1, 0]      = u[0, 0]
    u[n + 1, nx - 1] = u[0, nx - 1]
fig, ax = plt.subplots()
line, = ax.plot(x, u[0, :])
ts = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-1.5, 1.5)

def update(frame):
    line.set_ydata(u[frame, :])
    ts.set_text(f't = {frame*dt:.3f}')
    return line, ts

anim = FuncAnimation(
    fig,
    update,
    frames=range(0, nt + 1, max(1, nt // 200)),
    interval=30,
    blit=True,
)

anim.save("burgers.gif", writer=PillowWriter(fps=30))
plt.show()


