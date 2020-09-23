import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def model(t, z):
    H,I,V = z
    kr1 = 1e5    # new healthy cells per year
    kr2 = 0.1    # death rate of healthy cells
    kr3 = 2e-7   # healthy cells converting to infected cells
    kr4 = 0.5    # death rate of infected cells
    kr5 = 5      # death rate of virus
    kr6 = 100    # production of virus by infected cells

    dHdt = kr1 - kr2 * H - kr3 * H * V
    dIdt = kr3 * H * V - kr4 * I
    dVdt = -kr3 * H * V - kr5 * V + kr6 * I

    return [dHdt, dIdt, dVdt]

# Initial conditions
y0 = [1e6, 0, 100]

# Time span
t = [0, 15]

sol = solve_ivp(model, t, y0, method='LSODA') # LSODA is the same method used by ODEINT


fig, ax = plt.subplots(1, 1, figsize=(8,5))

ax.semilogy(sol.t, sol.y[0], 'b-', label=r'H')
ax.semilogy(sol.t, sol.y[1], 'g:', label=r'I')
ax.semilogy(sol.t, sol.y[2], 'r--', linewidth=2, label=r'V')
plt.xlabel('Time (years)')
plt.ylabel('States (log scale)')
plt.ylim(bottom=1)
plt.legend()
plt.show()
