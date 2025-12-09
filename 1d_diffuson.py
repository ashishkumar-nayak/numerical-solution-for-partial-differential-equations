import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation, PillowWriter


nx = 101          # no of grid points
xmax = 2
xmin = -2
dx = (xmax - xmin) / 100   # mesh size

x = np.linspace(xmin, xmax, nx)
print(x[100])
print(x)

c = 0.8
cfl = 0.2

dt = cfl * dx / abs(c)     # keep SAME dt as advection code
print(cfl)
print(dt)

t_end = 5.0
nt = int(t_end / dt)
t = np.linspace(0, nt*dt, nt+1)
print(t[-1], nt)

nu = 0.01   # diffusion coefficient
r = nu * dt / dx**2
print("r (nu*dt/dx^2) =", r)

u = np.zeros((nt+1, nx))
u[0, :] = np.exp(-0.5 * (x/0.4)**2)   # initial condition (same Gaussian)
uex = u[0, :].copy()

for n in range(0, nt):

    # update interior points
    for i in range(1, nx-1):
        u[n+1, i] = (
            u[n, i]
            + r * (u[n, i+1] - 2.0*u[n, i] + u[n, i-1])
        )

    # Dirichlet boundary conditions
    # keep boundary values fixed in time (like your advection code idea)
    #u[n+1, 0]     = u[0, 0]
    #u[n+1, nx-1]  = u[0, nx-1]
    u[n+1,0]=0
    u[n+1,nx-1]=0



# -----------------------------
fig, ax = plt.subplots()
line, = ax.plot(x, u[0, :])
ts = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("1D Diffusion")

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

anim.save("diffusion.gif", writer=PillowWriter(fps=30))
plt.show()
