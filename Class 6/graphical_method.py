import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

t_sim = 8 # seconds

# Simulation time
t = np.linspace(0, t_sim, t_sim*10)


###############################################################################
# VARIABLES
###############################################################################

Kp = 1     # Process Gain
θp = 0.7   # Process Dead time
τp = 2.1   # Process Time constant


###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

u = np.ones(len(t)) * 3
u[10:] = 2
# create linear interpolation of the u data versus time
uf = interp1d(t, u)


###############################################################################
# INITIAL CONDITIONS
###############################################################################

y0 = [5]


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

results  = np.ones(len(t)) * 5


###############################################################################
# SIMULATION
###############################################################################

def model(t, y, uf):
    y, = y
    
    # Magic from http://apmonitor.com/pdc/index.php/Main/FirstOrderGraphical
    if (t-θp) <= 0:
        um = uf(0)
    else:
        um = uf(t-θp)

    dydt = (-y + Kp * um) / τp
    
    return [dydt]


for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]

    inputs = [uf,]
    sol = solve_ivp(model, (t0, tf), y0, method='LSODA', max_step=0.01, args=inputs)  
    
    # Store results
    results[i+1] = sol.y[0, -1]
    
    # Adjust initial condition for next loop
    y0 = sol.y[:, -1]


###############################################################################
# PLOTTING
###############################################################################

plt.subplot(2,1,1)
plt.plot(t, results, 'r-', linewidth=2, label='First order model FOPDT')
plt.ylabel('Output')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, u, 'b-', linewidth=2, label='Input')
plt.xlabel('Time (sec)')
plt.ylabel('Input')
plt.legend()

plt.show()