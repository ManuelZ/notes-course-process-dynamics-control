import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

secs, resolution = 300, 10
m = GEKKO()
m.time = np.linspace(0, secs, secs * resolution + 1)

# Heater percentage points
Q_step = 50 * np.ones(secs * resolution + 1)
Q = m.Param(value=Q_step)  

Tau = 120 # seconds
Kp = 0.8  # degrees / percentage point
Ta = 23   # Initial temperature, degrees 

T = m.Var(23)
m.Equation(Tau * T.dt() == (Ta - T) + Kp * Q)

m.options.IMODE = 4
m.options.NODES = 3
m.solve()

plt.plot(m.time, T, 'r-', label=r'T(t)')
plt.plot(m.time, Q, 'k-', linewidth=2, label=r'Q(t)')
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()
