# %% Preamble
import matplotlib.pyplot as plt
import numpy as np

# Use same line colors as in paper
from cycler import cycler
linecolors = ['#902A3C','#185477','#E69C24']
plt.rc('axes', prop_cycle=(cycler('color', linecolors)))

# %% Define DRS algorithm for single-valued operators A and B

def drs(A, B, z0, γ, λ, N):
    n = z0.shape[0]
    zs = np.NaN * np.ones((N, n))
    z̅s = np.NaN * np.ones((N, n))
    ẑs = np.NaN * np.ones((N, n))
    I = np.eye(n)

    z = z0
    for i in range(N):
        zs[i] = z
        z̅ = np.linalg.solve(I + γ * A, z)
        z̅s[i] = z̅
        ẑ = np.linalg.solve(I + γ * B, 2 * z̅ - z)
        ẑs[i] = ẑ
        z = z + λ * (ẑ - z̅)

    return zs, z̅s, ẑs

# %% Define problem data
a = 2
b = -1
A = np.array([[0, a], [-a, 0]])
B = b * np.eye(2)
Tp = lambda z_ : (A + B)@z_

# %% Compute upper bound on lambda
γ = 1/2
λ_max = 2 * (b + γ * a * a) * (1 + γ * b) / (γ * (a * a + b * b))

# %% Perform experiments
results = {}

_, z̅s, _ = drs(A, B, np.array([1, 1.1]), γ, (6/5)*λ_max, 64)
results[r'$\lambda = \frac{6}{5} \bar \lambda$'] = z̅s

_, z̅s, _ = drs(A, B, np.array([1, 1.1]), γ, λ_max, 64)
results[r'$\lambda = \bar \lambda$'] = z̅s

_, z̅s, _ = drs(A, B, np.array([1, 1.1]), γ, (4/5)*λ_max, 64)
results[r'$\lambda = \frac{4}{5} \bar \lambda$'] = z̅s

resolution = 64
xx = np.linspace(-3, 3, resolution, endpoint=True)
yy = np.linspace(-3, 3, resolution, endpoint=True)
X, Y = np.meshgrid(xx, yy)
mW = np.vectorize(lambda x, y: -Tp(np.array([x, y])), signature="(),()->(m)")
UV = mW(X, Y)

plt.figure(figsize=(7.5, 7.5))
plt.streamplot(X, Y, UV[:, :, 0], UV[:, :, 1], color='#ccc')

for lbl, z̅s in results.items():
    plt.plot(z̅s[:, 0], z̅s[:, 1], ".-", label=lbl)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xlim(-1.6,1.6)
plt.ylim(-1.6,1.6)
plt.legend()
plt.show()

# %%
