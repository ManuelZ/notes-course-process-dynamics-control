import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# Notes
# u: refers to the input to the process
# y: refers to the process variable 


animate = False

# Total simulation time
t_sim = 1200      # seconds

# Resolution of the simulation
resolution = 10 # every 0.1 sec 

# Simulation time
t = np.linspace(0, t_sim, t_sim * resolution + 1)


###############################################################################
# INITIAL CONDITIONS
###############################################################################

# Input at steady state
uss = 0

# Value at steady state
yss = 0

# Initial output value
y = yss

# Initial conditions for the solver
y0 = [yss]


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

# Input storage
us = np.ones(t_sim * resolution + 1) * uss

# Output storage
ys = np.ones(t_sim * resolution + 1) * yss

# Error storage
es = np.zeros(t_sim * resolution + 1)


###############################################################################
# CONTROLLER PARAMETERS
###############################################################################

# Define setpoint values
sps = np.zeros(t_sim * resolution + 1)
sps[500:6000] = 10

# The bias term is a constant that is typically set to the value of the input 
# when the controller is first switched from manual to automatic mode.
ubias = 0

Kp = 2     # Process Gain (Δy/Δu)
τp = 200   # Process Time constant (63.2% in one τ)
θp = 0     # Process Dead time (time it takes the output to respond to the input)

# Proportional controller gain
Kc = 2

# Limits for the actuator
ACTUATOR_MAX = 100
ACTUATOR_MIN = 0


###############################################################################
# SIMULATION
###############################################################################

def model(t, y, u):
    """
    """
    # Extract values
    y = y[0]

    dydt = (1 / τp) * (-(y - yss) + Kp * (u - uss))

    return [dydt]


plt.figure(1, figsize=(12,5))
if animate:
    plt.ion()
    plt.show()


for i in range(len(t) - 1):
    SP = sps[i]
    error = SP - y
    u = ubias + (Kc * error) if i == 0 else (Kc * error)
    u = ACTUATOR_MAX if u > ACTUATOR_MAX else u
    u = ACTUATOR_MIN if u < ACTUATOR_MIN else u

    us[i+1]  = u       # Store the input
    es[i+1]  = error   # Store the error 
    
    t0, tf = t[i], t[i+1]
    inputs = [u, ]
    
    sol = solve_ivp(
        model,           # A callable defined above
        (t0, tf),        # Interval of integration
        y0,              # Initial conditions
        method='LSODA',  # 'LSODA': equivalent to odeint; 'RK45': good
        max_step=0.1,    # Required if the state is zero for too long
        args=inputs      # Additional arguments for the model
    )
    
    y = sol.y[0, -1]
    y0 = [y]
    ys[i+1] = y  # store the output
    
    if animate:
        plt.clf()

        # Output
        plt.subplot(3,1,1)
        plt.plot(t[0:i+1], ys[0:i+1],  'r-', linewidth=3, label='Process Value - PV')
        plt.plot(t[0:i+1], sps[0:i+1], 'k:', linewidth=3, label='Setpoint - SP')
        plt.ylabel('Output')
        plt.legend()
        
        # Input
        plt.subplot(3,1,2)
        plt.plot(t[0:i+1], us[0:i+1], 'b--', linewidth=3, label='Input')
        plt.ylabel('Input')    
        plt.legend()
        
        # Error
        plt.subplot(3,1,3)
        plt.plot(t[0:i+1], es[0:i+1], 'g-', linewidth=3, label='Error')
        plt.ylabel('Error = SP-PV')
        plt.xlabel('Time (sec)')
        plt.legend()
        plt.ylim(bottom=0)
        
        plt.pause(0.01)


if not animate:
    
    # Output
    plt.subplot(3,1,1)
    plt.plot(t, ys,  'r-', linewidth=3, label='Process Value - PV')
    plt.plot(t, sps, 'k:', linewidth=3, label='Setpoint - SP')
    plt.ylabel('')
    plt.legend()
    
    # Input
    plt.subplot(3,1,2)
    plt.plot(t, us, 'b--', linewidth=3, label='Input to the process')
    plt.ylabel('Input')    
    plt.legend()
    
    # Error
    plt.subplot(3,1,3)
    plt.plot(t, es, 'g-', linewidth=3, label='Error')
    plt.ylabel('Error = SP-PV')
    plt.xlabel('Time (sec)')
    plt.legend()
    
    plt.show()