# %% Preamble
# %matplotlib qt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy

plt.rcParams['text.usetex'] = True

# Use same line colors as in paper
from cycler import cycler
linecolors = ['#185477','#E69C24']
plt.rc('axes', prop_cycle=(cycler('color', linecolors)))


# Define the colormap from the paper
colors = [(0, '#d2aab1fb'),   # Red
          (0.9, '#f4d6a8ff'), # Orange
          (1, '#bdddbdff')]   # Green

custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

# %% Define PPPA algorithm for single-valued operator F and the convex set C from the Von Neumann problem

def get_bounds(case):
    '''
        We will consider 9 different cases, depending on the active constraints.
        Parametrize z as [s, 1-s, t, 1-t].
        Define w as the different variables [s, t, a1, a2, b1, b2, b3, b4].
        Then, the cases are visualized as follows:

            1-----2-----3
            |           |
            |           |
            8     0     4
            |           |
            |           |
            7-----6-----5
        t
          s
    '''
    a_min = -np.inf*np.ones(2)
    a_max = np.inf*np.ones(2)

    b_min = np.zeros(4)
    b_max = np.zeros(4)

    def s_bounds(case):
        if case in [0,2,6]:
            s_min = 0
            s_max = 1

        elif case in [1,7,8]:
            s_min = 0
            s_max = 0

        elif case in [3,4,5]:
            s_min = 1
            s_max = 1
        
        return s_min, s_max

    def t_bounds(case):
        if case in [0,4,8]:
            t_min = 0
            t_max = 1

        elif case in [5,6,7]:
            t_min = 0
            t_max = 0

        elif case in [1,2,3]:
            t_min = 1
            t_max = 1
        
        return t_min, t_max

    s_min, s_max = s_bounds(case)
    t_min, t_max = t_bounds(case)

    z_min = np.array([s_min, t_min])
    z_max = np.array([s_max, t_max])


    if s_min == 0 and s_max == 0:
        b_min[0] = -np.inf

    if s_min == 1 and s_max == 1:
        b_min[1] = -np.inf

    if t_min == 0 and t_max == 0:
        b_min[2] = -np.inf

    if t_min == 1 and t_max == 1:
        b_min[3] = -np.inf

    lb = np.concatenate((z_min, a_min, b_min))
    ub = np.concatenate((z_max, a_max, b_max))

    return lb, ub

def N_Ci(a, b, zi):
    return a * np.ones(2) + (zi <= 1e-12)*b

def N_C(x, y, a1, b1, a2, b2):
    return np.concatenate((N_Ci(a1, b1, x), N_Ci(a2, b2, y)))
    
def get_resolvent(F, P):
    def rootfinding(w_bar, z):
        s_bar, t_bar = w_bar[0:2]
        a1_bar, a2_bar = w_bar[2:4]
        b12_bar = w_bar[4:6]
        b34_bar = w_bar[6:8]

        x_bar = np.array([s_bar, 1-s_bar])
        y_bar = np.array([t_bar, 1-t_bar])
        z_bar = np.concatenate((x_bar, y_bar))
        
        
        return P @ z_bar + F(z_bar) + N_C(x_bar, y_bar, a1_bar, b12_bar, a2_bar, b34_bar) - P @ z 
    
    def get_zbar(z):
        w_results = []
        fun_results = []
        
        s0_guess = z[0]
        t0_guess = z[2]
        w0_guess = np.concatenate((np.array([s0_guess, t0_guess]), np.zeros(6)))
        
        for case in range(9):
            lb, ub = get_bounds(case)
            w0 = np.clip(w0_guess, lb, ub)
            res = scipy.optimize.minimize(
                lambda w : np.linalg.norm(rootfinding(w, z))**2, w0,
                bounds = scipy.optimize.Bounds(lb, ub, keep_feasible=False),
                method = 'L-BFGS-B'
                )
            
            if res['success'] == True:
                w_results.append(res['x'])
                fun_results.append(res['fun'])
            else:
                w_results.append([])
                fun_results.append(np.inf)

        s, t = w_results[np.argmin(fun_results)][:2]
        return np.array([s, 1-s, t, 1-t])
    return get_zbar
    
def pppa(F, P, z0, λ, N):
    n = z0.shape[0]
    zs = np.NaN * np.ones((N, n))
    z̅s = np.NaN * np.ones((N, n))

    preconditioned_resolvent = get_resolvent(F, P)
    
    z = z0
    for i in range(N):
        zs[i] = z

        z̅ = preconditioned_resolvent(z)

        z̅s[i] = z̅
        z = z + λ * (z̅ - z)
    return zs, z̅s

# %% Define problem data
eps = 0.1

R = np.array([[0, 1 + eps/2], [2 - eps/2, 2]])
S = np.array([[1/2, 1/2], [1, 1]])

z_star = np.array([0, 1, 0, 1])

def F(z):
    x = z[:2]
    y = z[2:]

    f = x.T @ R @ y
    g = x.T @ S @ y

    dfx = R @ y
    dfy = R.T @ x

    dgx = S @ y
    dgy = S.T @ x

    Vx = (g * dfx - f * dgx)/g**2
    Vy = (g * dfy - f * dgy)/g**2

    return np.concatenate((Vx, -Vy))


# %% Perform experiments
gam = 1/4
P = 1/gam*np.eye(4)
λ = 1/3

s0 = 0.5
t0 = 0.9
z0 = np.array([s0, 1-s0, t0, 1-t0])

zs, z̅s = pppa(F, P, z0, λ, 300)

# %% Show visualizations of the Von Neumann problem
resolution = 64
ss = np.linspace(0, 1, resolution, endpoint=True)
tt = np.linspace(0, 1, resolution, endpoint=True)
S, T = np.meshgrid(ss, tt)

def rho_epsilon(s, t, eps):
    num = 4*(2-s)**2*(eps*(s+t) - s*t*(s+eps))
    denom = (2*eps - t*(4 + eps))**2 + (eps + 2*s)**2*(2-s)**2
    return num/denom

rho_epsilon_vec = np.vectorize(lambda s,t : rho_epsilon(s,t,eps), signature="(),()->()")
rho_epsilon_grid = rho_epsilon_vec(S, T)

plt.figure(figsize=(9, 7.5))
plt.imshow(np.minimum(rho_epsilon_grid, 0),
           extent=[0,1,0,1],
           origin="lower",
           cmap=custom_cmap,
           aspect='auto')
plt.colorbar().set_label(label=r'$\rho_{\epsilon}(s,t)$',size=20)
plt.plot(z̅s[:, 0], z̅s[:, 2], ".-", label=r'$\bar{z}^k, \lambda = \eta_{\rm min}$')
plt.plot(zs[:, 0], zs[:, 2], ".-", label=r'$z^k, \lambda = \eta_{\rm min}$')
plt.xlabel('$s$', fontsize=15)
plt.ylabel('$t$', fontsize=15)
plt.legend(fontsize=15)
plt.show()

plt.figure(figsize=(7.5, 7.5))
plt.plot(np.linalg.norm(z̅s - z_star, axis = 1), ".-", label=r'$\|\bar{z}^k - z^\star\|$')
plt.plot(np.linalg.norm(zs - z_star, axis = 1), ".-", label=r'$\|z^k - z^\star\|$')
plt.xlabel('$s$', fontsize=15)
plt.ylabel('$t$', fontsize=15)
plt.xlim(0,50)
plt.ylim(0.85,1.55)
plt.legend(fontsize=15)
plt.show()

# %% Show comparison between optimal weak Minty constant and its lower bound
rho_list = []
for e in np.linspace(0,1,101):
    rho = scipy.optimize.minimize(
                    lambda w : rho_epsilon(w[0], w[1], e),
                    np.array([1,1]),
                    bounds = scipy.optimize.Bounds([np.sqrt(e),e], [1,1], keep_feasible=False),
                    method = 'L-BFGS-B'
                    )
    rho_list.append(rho['fun'])

plt.figure
plt.plot(np.linspace(0,1,101), rho_list, label=r'$\rho_\epsilon$')
plt.plot(np.linspace(0,1,101), np.linspace(-1/4,0,101), label=r'$\frac{\epsilon -1}{4}$')
plt.xlabel(r'$\epsilon$')
plt.legend(fontsize=15)
plt.show()

# %%
