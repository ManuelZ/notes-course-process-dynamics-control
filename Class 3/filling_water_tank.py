import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# Problem
#
# Simulate the height of the tank by integrating the mass balance equation for 
# a period of 10 seconds. 
#
# - The valve opens to 100% at time=2 and shuts at time=7. 
# - Use a value of 1000 kg/m3 for density.
# - Use a value of 1.0 m2 for the cross-sectional area of the tank. 
# - For the valve, assume a valve coefficient of c=50.0 (kg/s / %open).
#
# Source: http://apmonitor.com/pdc/index.php/Main/DynamicModeling


def model(t, y):
    rho = 1000 # kg/m^3
    A = 1      # m^2
    c = 50     # kg/s / %open
    valve = 100 if (t >= 2) and (t < 7) else 0
    dHdt = c * valve / (rho * A)
    return [dHdt]

t0, tf = (0, 10) # s
y0 = [0]
t_eval= np.linspace(0, 10, 100)

sol = solve_ivp(model, (t0, tf), y0, method='LSODA', max_step=0.1)

plt.plot(sol.t, sol.y[0], 'r-', linewidth=2, label='')
plt.xlabel('Time')
plt.ylabel('Tank height y(t)')
plt.show()