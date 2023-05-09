# %% Preamble
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np

# Use same line colors as in paper
from cycler import cycler
linecolors = ['#185477','#E69C24']
plt.rc('axes', prop_cycle=(cycler('color', linecolors)))

# %% Define problem data
a = 10
b = -1
A = np.array([[0,a],[-a,0]])
B = b * np.eye(2)

# %% Define free variables
rhoA = cp.Variable((2,2))
rhoB = cp.Variable((2,2))
sigmaA = cp.Variable((2,2))
sigmaB = cp.Variable((2,2))

beta_primal = cp.Variable(1)
beta_dual = cp.Variable(1)

# %% Construct LMI problem
M = cp.bmat([[sigmaA - beta_dual*np.eye(2), -beta_dual*np.eye(2), np.zeros((2,2)), np.zeros((2,2))],
    [-beta_dual*np.eye(2), sigmaB - beta_dual*np.eye(2), np.zeros((2,2)), np.zeros((2,2))],
    [np.zeros((2,2)), np.zeros((2,2)), rhoB - beta_primal*np.eye(2), -beta_primal*np.eye(2)],
    [np.zeros((2,2)), np.zeros((2,2)), -beta_primal*np.eye(2), rhoA - beta_primal*np.eye(2)]])

constraints = [M >> 0,                                          # beta_primal = rhoA Box rhoB and beta_dual = monA Box monB (cf. proof Lem. 5.1)
               0 >> (A.T + A)/2 + sigmaA + A.T @ rhoA @ A,      # Semimonotonicity of A
               0 >> -(B.T + B)/2 + sigmaB + B.T @ rhoB @ B,     # Semimonotonicity of B
               sigmaA + sigmaB >> 0,                            # Assumption III.A1
               rhoA + rhoB >> 0]                                # Assumption III.A1

nb_gammas = 250
gammas = np.linspace(0.01,1,nb_gammas)
λ_max_mosek = np.zeros(nb_gammas)
λ_max = np.zeros(nb_gammas)

for i, gamma in enumerate(gammas):
    cost = 2*(1 + gamma**(-1)*beta_primal + gamma*beta_dual)    # Maximize upper bound on lambda
    problem = cp.Problem(cp.Maximize(cost), constraints)
    λ_max_mosek[i] = problem.solve(solver='MOSEK')
    λ_max[i] = 2 * (b + gamma * a**2)*(1 + gamma * b)/(gamma*(a**2 + b**2))

plt.figure(figsize=(4, 3))
plt.plot(gammas, λ_max, label=r'$\bar \lambda$')
plt.plot(gammas, λ_max_mosek, label=r'$\lambda_\mathrm{max}^\mathrm{semi}$')
plt.ylim(bottom=0)
plt.xlabel('$\gamma$')
plt.ylabel('$\lambda$')
plt.legend()
plt.show()

# %%
