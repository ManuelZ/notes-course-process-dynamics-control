import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def model(t, z):
    x,y = z
    u = 0 if t < 5 else 2

    dxdt = (1/2) * (- x + u)
    dydt = (1/5) * (- y + x)
    return [dxdt, dydt]

# Initial conditions
y0 = [0, 0]

# Time span
t = [0, 40]

sol = solve_ivp(model, t, y0, method='LSODA') # LSODA is the same method used by ODEINT

plt.plot([0,5, 5,40], [0,0, 2,2], 'g:', label='Input u(t)')
plt.plot(sol.t, sol.y[0], 'r-', linewidth=2, label='Output x(t)')
plt.plot(sol.t, sol.y[1], 'b--', linewidth=2, label='Output y(t)')

plt.xlabel('Time')
plt.ylabel('y(t)')

plt.legend()
plt.show()