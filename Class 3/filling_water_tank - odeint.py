import numpy as np
from scipy.integrate import odeint
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


def model(y, t):
    rho = 1000 # kg/m^3
    A = 1      # m^2
    c = 50     # kg/s / %open
    valve = 100 if (t >= 2) and (t < 7) else 0

    dHdt = c * valve / (rho * A)
    return dHdt

t = np.linspace(0, 10, 101)
y0 = [0]

sol = odeint(model, y0, t)

plt.plot(t, sol, 'r-', linewidth=2, label='')
plt.xlabel('Time')
plt.ylabel('Tank height y(t)')
plt.show()