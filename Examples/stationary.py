# %% Preamble
import matplotlib.pyplot as plt
import numpy as np

# %% Problem definition
muA = -3/10
rhoA = -1/10
muB = 4/10
rhoB = 4/10

def A(x):
    if x < -3 or x > 3:
        return (1 + np.sqrt(1-4*muA*rhoA))/(2*rhoA)*x - 1
    else:
        return (1 - np.sqrt(1-4*muA*rhoA))/(2*rhoA)*x - 1

def B(x):
    if x < -1 or x > 1:
        return (1 - np.sqrt(1-4*muB*rhoB))/(2*rhoB)*x + 1
    else:
        return (1 + np.sqrt(1-4*muB*rhoB))/(2*rhoB)*x + 1
    
beta_mu = muA*muB/(muA+muB)
beta_rho = rhoA*rhoB/(rhoA+rhoB)


γ_min_DRS = - 2*beta_rho / (1+np.sqrt(1-4*beta_mu*beta_rho))
γ_max_DRS = - (1+np.sqrt(1-4*beta_mu*beta_rho)) / (2*beta_mu)
γ_max = 1/5

γ = (γ_min_DRS + γ_max)/2

λ_max = 2 + 2 * (1/γ)*beta_rho + 2*γ*beta_mu

# %% Compute resolvent of A and B
sA1 = -3*(1+γ*(1 - np.sqrt(1-4*muA*rhoA))/(2*rhoA)) - γ
sA2 = 3*(1+γ*(1 + np.sqrt(1-4*muA*rhoA))/(2*rhoA)) - γ
sA3 = -3*(1+γ*(1 + np.sqrt(1-4*muA*rhoA))/(2*rhoA)) - γ
sA4 = 3*(1+γ*(1 - np.sqrt(1-4*muA*rhoA))/(2*rhoA)) - γ

def resA(s, g):
    if s < sA1 or s > sA4:
        return (1+g*(1 + np.sqrt(1-4*muA*rhoA))/(2*rhoA))**(-1)*(s+g)
    elif s > sA2 and s < sA3:
        return (1+g*(1 - np.sqrt(1-4*muA*rhoA))/(2*rhoA))**(-1)*(s+g)
    else:
        # print("random A")
        if np.random.choice([0,1]):
            return (1+g*(1 + np.sqrt(1-4*muA*rhoA))/(2*rhoA))**(-1)*(s+g)
        else:
            return (1+g*(1 - np.sqrt(1-4*muA*rhoA))/(2*rhoA))**(-1)*(s+g)
        
sB1 = -(1+γ*(1 + np.sqrt(1-4*muB*rhoB))/(2*rhoB)) + γ
sB2 = -(1+γ*(1 - np.sqrt(1-4*muB*rhoB))/(2*rhoB)) + γ
sB3 = (1+γ*(1 - np.sqrt(1-4*muB*rhoB))/(2*rhoB)) + γ
sB4 = (1+γ*(1 + np.sqrt(1-4*muB*rhoB))/(2*rhoB)) + γ

def resB(s, g):
    if s < sB1 or s > sB4:
        return (1+g*(1 - np.sqrt(1-4*muB*rhoB))/(2*rhoB))**(-1)*(s-g)
    elif s > sB2 and s < sB3:
        return (1+g*(1 + np.sqrt(1-4*muB*rhoB))/(2*rhoB))**(-1)*(s-g)
    else:
        # print("random B")
        if np.random.choice([0,1]):
            return (1+g*(1 - np.sqrt(1-4*muB*rhoB))/(2*rhoB))**(-1)*(s-g)
        else:
            return (1+g*(1 + np.sqrt(1-4*muB*rhoB))/(2*rhoB))**(-1)*(s-g)

# %% Perform DRS
def DRS(s0, g, lam, N):
    x_bar = np.zeros(N)
    x_hat = np.zeros(N)
    s = np.zeros(N+1)
    s[0] = s0
    for i in range(N):
        x_bar[i] = resA(s[i], g)
        x_hat[i] = resB(2*x_bar[i] - s[i], g)
        s[i+1] = s[i] + lam * (x_hat[i] - x_bar[i])
    return x_bar, x_hat, s

# %%

plt.figure(figsize=(4, 3))
for s0 in np.arange(-5,5,0.05):
    x_bar = DRS(s0, γ, 0.9*λ_max, 400)[0]
    plt.plot(x_bar, 'black', alpha=0.15)
# plt.ylim(bottom=0)
plt.xlabel('Iterations')
plt.ylabel('$x$')
plt.show()

# %%
