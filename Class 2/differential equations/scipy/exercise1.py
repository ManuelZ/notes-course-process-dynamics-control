import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def model(t, y):
    dydt = -y + 1
    return dydt

# Initial conditions
y0 = [0]

# Time span
t = [0, 10]

sol = solve_ivp(model, t, y0, method='LSODA') # LSODA is the same method used by ODEINT

# plot results
plt.plot(sol.t, sol.y[0],'r-', linewidth=2, label='')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.show()