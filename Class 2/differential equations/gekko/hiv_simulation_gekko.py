import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

# Requires model initialization strategies as seen here:
# https://apmonitor.com/do/index.php/Main/ModelInitialization

years = 15
resolution = 10
m = GEKKO()
m.time = np.linspace(0, years, years * resolution + 1)

H = m.Var(1e6)
I = m.Var(0)
V = m.Var(100)

kr1 = m.Param(1e5)    # new healthy cells per year
kr2 = m.Param(0.1)    # death rate of healthy cells
kr3 = m.Param(2e-7)   # healthy cells converting to infected cells
kr4 = m.Param(0.5)    # death rate of infected cells
kr5 = m.Param(5)      # death rate of virus
kr6 = m.Param(100)    # production of virus by infected cells

m.Equation(H.dt() == kr1 - kr2 * H - kr3 * H * V)
m.Equation(I.dt() == kr3 * H * V - kr4 * I)
m.Equation(V.dt() == -kr3 * H * V - kr5 * V + kr6 * I)

m.options.IMODE = 4
m.options.NODES = 3
m.solve()

fig, ax = plt.subplots(1, 1, figsize=(8,5))

ax.semilogy(m.time, H, 'b-', label=r'H')
ax.semilogy(m.time, I, 'g:', label=r'I')
ax.semilogy(m.time, V, 'r-', linewidth=2, label=r'V')
plt.xlabel('Time (years)')
plt.ylabel('States (log scale)')
plt.legend()
plt.show()
