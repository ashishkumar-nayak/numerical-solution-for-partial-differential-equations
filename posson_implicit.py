import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# -----------------------------
# Grid and problem setup
# -----------------------------
L = 1.0
N = 50  # number of intervals  -> N+1 grid points

dx = L / N
x = np.linspace(0, L, N + 1)

# Gaussian source term f(x)
A0 = 1.0
x0 = 0.5
sigma = 0.1
f = A0 * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Boundary conditions
phi_left = 0.0
phi_right = 0.0

# -----------------------------
# Build matrix A (size (N-1)x(N-1))
# -----------------------------
m = N - 1  # number of unknown interior points: phi_1,...,phi_{N-1}
A = np.zeros((m, m))

# Tridiagonal: -1, 2, -1  (from -phi_{i-1} + 2 phi_i - phi_{i+1})
for i in range(m):
    A[i, i] = 2.0                    # main diagonal
    if i > 0:
        A[i, i - 1] = -1.0           # lower diagonal
    if i < m - 1:
        A[i, i + 1] = -1.0           # upper diagonal

# -----------------------------
# Build RHS vector b
# -----------------------------
b = - (dx**2) * f[1:N]              # f_1, ..., f_{N-1}

# Apply boundary contributions (Dirichlet)
b[0]  += phi_left                   # from -phi_0 term in first equation
b[-1] += phi_right                  # from -phi_N term in last equation

# -----------------------------
# Solve using A^{-1} b
# -----------------------------
A_inv = inv(A)                      # compute inverse (not efficient, but okay for learning)
phi_inner = A_inv @ b               # interior solution: phi_1,...,phi_{N-1}

# Reconstruct full phi including boundaries
phi = np.zeros(N + 1)
phi[0] = phi_left
phi[N] = phi_right
phi[1:N] = phi_inner

# -----------------------------
# Plot
# -----------------------------
plt.plot(x, phi, label="phi(x) solution")
plt.plot(x, f, '--', label="Gaussian f(x)")
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()
