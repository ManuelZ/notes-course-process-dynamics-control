import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Simulation time
t = np.linspace(0, 10, 100)


###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

# Feed Concentration (mol/m^3)
Caf = 1
# Feed Temperature (K)
Tf = 300


###############################################################################
# INITIAL CONDITIONS
###############################################################################

Ca0 = 0.0
T0 = 350
y0 = [Ca0, T0]


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

Ca = np.ones(len(t)) * Ca0
T  = np.ones(len(t)) * T0


###############################################################################
# SIMULATION
###############################################################################

def vessel(t, y, Caf, Tf):
    Ca,T = y

    # Constants 
    q = 100 # m^3 / hr
    V = 100 # m^3

    dCadt  = (q / V) * (Caf - Ca)
    dTdt = (q / V) * (Tf - T)

    return [dCadt, dTdt]


for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = (Caf, Tf)
    sol = solve_ivp(vessel, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)  
    y = sol.y
    
    # Store results
    Ca[i+1] = y[0, -1]
    T[i+1]  = y[1, -1]
    
    # Adjust initial condition for next loop
    y0 = y[:, -1]

###############################################################################
# PLOTTING
###############################################################################

# VOLUME
plt.subplot(2,1,1)
plt.plot(t, Ca, 'b-', linewidth=2, label='Concentration')
plt.ylabel('Concentration (mol/$m^3$)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, T, 'r--', linewidth=2, label='Temperature')
plt.ylabel('Temperature ($^\circ$K)')
plt.legend()

plt.show()