import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def model(t, y):
    u = 0 if t < 10 else 2
    dydt = (1 / 5) * (-y + u)
    return dydt

# Initial conditions
y0 = [1]

# Time span
t = [0, 40]

sol = solve_ivp(model, t, y0, method='LSODA') # LSODA is the same method used by ODEINT

plt.plot(sol.t, sol.y[0], 'r-', linewidth=2, label='Output y(t)')
plt.plot([0,10,10,40], [0,0,2,2], 'b-', label='Input u(t)')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.show()