# %% Preamble
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Use same line colors as in paper
from cycler import cycler
linecolors = ['#185477','#E69C24']
plt.rc('axes', prop_cycle=(cycler('color', linecolors)))

# %% Define PPA algorithm for single-valued operator F

def ppa(F, z0, λ, N):
    n = z0.shape[0]
    zs = np.NaN * np.ones((N, n))
    z̅s = np.NaN * np.ones((N, n))

    z = z0
    for i in range(N):
        zs[i] = z
        res = scipy.optimize.root(lambda x : x + F(x) - z, z)
        if res['success']:
            z̅ = scipy.optimize.root(lambda x : x + F(x) - z, z)['x']
            if np.linalg.norm(z̅ + F(z̅) - z) > 1e-8:
                print(np.linalg.norm(z̅ + F(z̅) - z))
        else:
            print('PPA failed')
            print(res['message'])
            print(np.linalg.norm(z̅ + F(z̅) - z))
        z̅s[i] = z̅
        z = z + λ * (z̅ - z)
    return zs, z̅s

# %% Define problem data
a = 2
b = 1
M = np.array([[b, a], [-a, b]])

F_dict = {}

# f equal to 1
f_dist2 = lambda nrm_z : 1
F_dict["$f \equiv 1$"] = lambda z_: f_dist2(np.linalg.norm(z_)) * M @ z_

# f as in Figure 3
f_dist = lambda nrm_z : nrm_z if nrm_z <= 0.4 else 0.8 - nrm_z if nrm_z < 0.8 else 0 if nrm_z < 1 else 2.5*(nrm_z - 1) if nrm_z <= 1.4 else 1
F_dict["$f$ as in Fig. 3"] = lambda z_ : f_dist(np.linalg.norm(z_)) * M @ z_ 


# %% Compute weak Minty parameter and upper bound on lambda
ρ = b / (a*a + b*b)
λ_max = 2*(1+ρ)
assert(λ_max == 2.4)

# %% Perform experiments
for op_name, op in F_dict.items():
    for λ in [2.3, 2.5]:
        results = {}

        _, z̅s = ppa(op, np.array([4, 0]), λ, 50)
        results["x_0 = (4,0)"] = z̅s

        _, z̅s = ppa(op, np.array([1/2, 1/2]), λ, 50)
        results["x_0 = (1/2,1/2)"] = z̅s

        resolution = 64
        xx = np.linspace(-3, 3, resolution, endpoint=True)
        yy = np.linspace(-3, 3, resolution, endpoint=True)
        X, Y = np.meshgrid(xx, yy)
        mW = np.vectorize(lambda x, y: -op(np.array([x, y])), signature="(),()->(m)")
        UV = mW(X, Y)


        plt.figure(figsize=(7.5, 7.5))
        plt.streamplot(X, Y, UV[:, :, 0], UV[:, :, 1], color='#ccc')
        plt.scatter(0, 0, color='#228B22', zorder=5)
        if op_name == "$f$ as in Fig. 3":
            circle = plt.Circle((0, 0), 1, color='#228B22', fill=True, alpha = 0.33)
            plt.gca().add_patch(circle)
            circle = plt.Circle((0, 0), 0.8, color='w', fill=True)
            plt.gca().add_patch(circle)
            circle = plt.Circle((0, 0), 1, color='#228B22', fill=False)
            plt.gca().add_patch(circle)
            circle = plt.Circle((0, 0), 0.8, color='#228B22', fill=False)
            plt.gca().add_patch(circle)


        for lbl, z̅s in results.items():
            plt.plot(z̅s[:, 0], z̅s[:, 1], ".-", label=lbl)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(op_name + r', $\lambda = $'+ str(λ))
        plt.xlim(-1.6,1.6)
        plt.ylim(-1.6,1.6)
        plt.legend()
        plt.show()


# %%
