import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Exercise: http://apmonitor.com/pdc/index.php/Main/PhysicsBasedModels
#
# Use a mass, species, and energy balance to describe the dynamic response in:
#     - volume
#     - concentration
#     - temperature 
# of a well-mixed vessel.
#
# The inlet (qf) and outlet (q) volumetric flowrates, feed concentration (Caf),
# and inlet temperature (Tf) can be adjusted. 
#
# Initial conditions for the vessel are:
#     V  = 1.0 L
#     Ca = 0.0 mol/L
#     T  = 350 K
# 
# There is no reaction and no significant heat added by the mixer. 
# There is a cooling jacket that can be used to adjust the outlet temperature. 
# Show step changes in the process inputs. 


# Simulation time
t = np.linspace(0, 10, 100) # seconds

###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

# Inlet Volumetric Flowrate (L/min)
qf = np.ones(len(t))* 5.2
qf[50:] = 5.1

# Feed Concentration (mol/L)
Caf = np.ones(len(t))*1.0
Caf[30:] = 0.5

# Feed Temperature (K)
Tf = np.ones(len(t))*300.0
Tf[70:] = 325.0

# Outlet Volumetric Flowrate (L/min)
q = np.ones(len(t))*5.0


###############################################################################
# INITIAL CONDITIONS
###############################################################################

V0 = 1.0
Ca0 = 0.0
T0 = 350.0
y0 = [V0, Ca0, T0]


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

V  = np.ones(len(t))*V0
Ca = np.ones(len(t))*Ca0
T  = np.ones(len(t))*T0


###############################################################################
# SIMULATION
###############################################################################

def vessel(t, y, q, qf, Caf, Tf):
    V,Ca,T = y
    dVdt  = qf - q
    dCadt = (qf * Caf - q * Ca - Ca * dVdt) / V
    dTdt  = (qf * Tf - q * T - T * dVdt) / V
    return [dVdt, dCadt, dTdt]


for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = (q[i], qf[i], Caf[i], Tf[i])
    sol = solve_ivp(vessel, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)  
    y = sol.y
    
    # Store results
    V[i+1]  = y[0, -1]
    Ca[i+1] = y[1, -1]
    T[i+1]  = y[2, -1]
    
    # Adjust initial condition for next loop
    y0 = y[:, -1]


###############################################################################
# PLOTTING
###############################################################################

# VOLUME
plt.subplot(3,2,1)
plt.plot(t, qf, 'b--', linewidth=2, label='qf - Inlet ')
plt.plot(t, q, 'b:', linewidth=2, label='q - Outlet')
plt.ylabel('Flow rates (L/min)')
plt.legend()

plt.subplot(3,2,2)
plt.plot(t, V, 'b-', linewidth=2, label='V')
plt.ylabel('Tank volume (L)')
plt.legend()

# CONCENTRATION
plt.subplot(3,2,3)
plt.plot(t, qf, 'k--', linewidth=2, label='Caf')
plt.ylabel('Feed concentration (mol/L)')
plt.legend()

plt.subplot(3,2,4)
plt.plot(t, Ca, 'k-', linewidth=2, label='Ca')
plt.ylabel('Tank concentration (mol/L)')
plt.legend()

# TEMPERATURE
plt.subplot(3,2,5)
plt.plot(t, qf, 'r--', linewidth=2, label='Tf')
plt.xlabel('Time (min)')
plt.ylabel('Feed Temperature (°K)')
plt.legend()

plt.subplot(3,2,6)
plt.plot(t, T, 'r-', linewidth=2, label='T')
plt.xlabel('Time (min)')
plt.ylabel('Tank Temperature (°K)')
plt.legend()

plt.show()