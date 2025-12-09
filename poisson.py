import numpy as np
import matplotlib.pyplot as plt

m = 100  # number of squares so grid is (m+1) x (m+1)
v = 1.0  # boundary potential
targ = 1e-6  # accuracy

# grid spacing (assuming domain length = 1 in both x,y)
Lx = 1.0
Ly = 1.0
hx = Lx / m
hy = Ly / m   # we'll assume hx = hy, so h = hx

h2 = hx * hx  # h^2

# potential grid
phi = np.zeros([m+1, m+1])

# boundary conditions
phi[0, :] = v
phi[m, :] = v
phi[:, 0] = v
phi[:, m] = v

phiprime = np.zeros_like(phi)

# define source term f(x,y) on the grid
# example: a localized positive "charge" near the center
x = np.linspace(0, Lx, m+1)
y = np.linspace(0, Ly, m+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# Example f(x,y): Gaussian source in the middle
sigma = 0.05
# f = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2*sigma**2))
f=(np.sin(X)-np.cos(X))**2 - np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2*sigma**2))

delta = 1.0

while delta > targ:

    for i in range(m+1):
        for j in range(m+1):

            # keep boundaries fixed
            if i == 0 or i == m or j == 0 or j == m:
                phiprime[i, j] = phi[i, j]
            else:
                # Jacobi update for Poisson: Laplacian(phi) = f
                phiprime[i, j] = 0.25 * (
                    phi[i+1, j] + phi[i-1, j] +
                    phi[i, j+1] + phi[i, j-1] -
                    h2 * f[i, j]
                )

    delta = np.max(np.abs(phi - phiprime))
    phi, phiprime = phiprime, phi

plt.imshow(phi, origin='lower', extent=[0, Lx, 0, Ly])
plt.colorbar(label='Potential φ')
plt.title('Solution of Poisson Equation (Jacobi)')
plt.show()
