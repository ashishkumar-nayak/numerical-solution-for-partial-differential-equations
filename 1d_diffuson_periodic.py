import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

nx = 101          # number of grid points
xmax = 2
xmin = -2

dx = (xmax - xmin) / (nx - 1)

x = np.linspace(xmin, xmax, nx)

# Same dt as in your advection code
c = 0.8
cfl = 0.2
dt = cfl * dx / abs(c)

t_end = 5.0
nt = int(t_end / dt)

nu = 0.01                      # diffusion coefficient
r = nu * dt / dx**2
print("r =", r)

# Allocate array for solution
u = np.zeros((nt+1, nx))

# Initial condition (Gaussian)
u[0, :] = np.exp(-0.5 * (x/0.4)**2)


#-
for n in range(0, nt):

    # interior points: 1 ... nx-2
    for i in range(1, nx-1):
        u[n+1, i] = (
            u[n, i]
            + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        )

    # periodic boundary at i = 0
    u[n+1, 0] = (
        u[n, 0]
        + r * (u[n, 1] - 2*u[n, 0] + u[n, nx-1])
    )

    # periodic boundary at i = nx-1
    u[n+1, nx-1] = (
        u[n, nx-1]
        + r * (u[n, 0] - 2*u[n, nx-1] + u[n, nx-2])
    )



fig, ax = plt.subplots()
line, = ax.plot(x, u[0, :])
ts = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("1D Diffusion (Periodic BC)")

def update(frame):
    line.set_ydata(u[frame, :])
    ts.set_text(f't = {frame*dt:.3f}')
    return line, ts

anim = FuncAnimation(
    fig,
    update,
    frames=range(0, nt+1, max(1, nt//200)),
    interval=30,
    blit=True,
)

anim.save("diffusion_periodic.gif", writer=PillowWriter(fps=30))
plt.show()

