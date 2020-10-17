import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# A proportional controller works really well with integrating systems (where 
# the steady state depends on prior history; it doesn't return to the same 
# steady state all the time). https://youtu.be/EDMGqB252Zs?t=712


#  Wether to animate or plot the final result
animate = False

# Total simulation time
t_sim = 10      # seconds

# Resolution of the simulation
resolution = 10 # every 0.1 sec 


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

# Setpoint storage
sps = np.zeros(t_sim * resolution + 1)

# Valve % open, input; it is now managed by the controller, not manually
u = np.zeros(t_sim * resolution + 1) # valve % open

# Error storage
es = np.zeros(t_sim * resolution + 1)

# Tank's water level
z = np.zeros(t_sim * resolution + 1)


###############################################################################
# INITIAL CONDITIONS
###############################################################################

level = 0  # initial tank level
valve = 10 # initial valve % openness
y0 = [level]


###############################################################################
# CONTROLLER PARAMETERS
###############################################################################

# Setpoint
SP = 10  # meters

# The bias term is a constant that is typically set to the value of the input 
# when the controller is first switched from manual to automatic mode.
ubias = 0

Kp = 1
θp = 0.1
τp = 5

# Proportional controller gain
Kc = 100


###############################################################################
# SIMULATION
###############################################################################

# Simulation time
t = np.linspace(0, t_sim, t_sim * resolution + 1)

c = 50.0     # valve coefficient (kg/s / %open)
rho = 1000.0 # water density (kg/m^3)
A = 1.0      # tank area (m^2)

def tank(t, y, c, valve):
    dLevel_dt = (c / (rho * A)) * valve
    return dLevel_dt

# Store the initial values
u[0]   = valve   # store the valve position
sps[0] = SP      # store the setpoint
z[0]   = level

plt.figure(1, figsize=(12,5))
if animate:
    plt.ion()
    plt.show()

for i in range(len(t) - 1):
    error = SP - level
    valve = ubias + (Kc * error) if i == 0 else (Kc * error)
    valve = 100 if valve > 100 else valve
    valve = 0 if valve < 0 else valve

    u[i+1]   = valve   # store the valve position
    es[i+1]  = error   # store the error 
    sps[i+1] = SP      # store the setpoint
    
    t0, tf = t[i], t[i+1]
    inputs = (c, valve)
    sol = solve_ivp(tank, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)
    level = sol.y[0, -1]
    y0 = [level]
    z[i+1] = level  # store the level
    
    if animate:
        plt.clf()

        # Tank level
        plt.subplot(3,1,1)
        plt.plot(t[0:i+1], z[0:i+1],'r-',linewidth=3,label='level PV')
        plt.plot(t[0:i+1], sps[0:i+1],'k:',linewidth=3,label='level SP')
        plt.ylabel('Tank Level')
        plt.legend(loc='best')
        
        # Valve opening
        plt.subplot(3,1,2)
        plt.plot(t[0:i+1], u[0:i+1],'b--',linewidth=3,label='valve')
        plt.ylabel('Valve')    
        plt.legend(loc='best')
        
        # Error
        plt.subplot(3,1,3)
        plt.plot(t[0:i+1],es[0:i+1],'g-',linewidth=3,label='error')
        plt.ylabel('Error = SP-PV')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')
        plt.ylim(bottom=0)
        
        plt.pause(0.1)

if not animate:
    
    # Tank level
    plt.subplot(3,1,1)
    plt.plot(t,z,'r-',linewidth=3,label='level PV')
    plt.plot(t,sps,'k:',linewidth=3,label='level SP')
    plt.ylabel('Tank Level')
    plt.legend(loc='best')
    
    # Valve opening
    plt.subplot(3,1,2)
    plt.plot(t,u,'b--',linewidth=3,label='valve')
    plt.ylabel('Valve')    
    plt.legend(loc='best')
    
    # Error
    plt.subplot(3,1,3)
    plt.plot(t,es,'g-',linewidth=3,label='error')
    plt.ylabel('Error = SP-PV')    
    plt.xlabel('Time (sec)')
    plt.legend(loc='best')
    
    plt.show()