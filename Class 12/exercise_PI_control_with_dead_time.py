import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# Notes
# u: refers to the controller output (CO), also named output (OP), also named Manipulated Variable
# y: refers to the process variable (PV)


# Total simulation time
t_sim = 1200      # seconds

# Resolution of the simulation (ticks per second)
resolution = 10

# Simulation time
t = np.linspace(0, t_sim, t_sim * resolution + 1)
dt = t[1]-t[0]


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
error = np.zeros(t_sim * resolution + 1)

# Integral of error
ierror = np.zeros(t_sim * resolution + 1)


###############################################################################
# CONTROLLER PARAMETERS
###############################################################################

# Define setpoint values
sps = np.zeros(t_sim * resolution + 1)
sps[50*resolution: 600*resolution] = 10

# The bias term is a constant that is typically set to the value of the input 
# when the controller is first switched from manual to automatic mode.
ubias = 0

# First Order Process Plus Dead Time model parameters
Kp = 2     # Process Gain (Δy/Δu)
τp = 200   # Process Time constant (process goes from one ss to 63.2%  of other ss in one τ)
θp = 100   # Process Dead time (time it takes the output to respond to the input)

# Controller's Proportional gain
Kc = 2
# Controller's integral time constant
τI = 200

# Simulate time delay
ndelay = int(np.ceil(θp / dt))

# Limits for the actuator
CONTROLLER_OUPUT_MAX = 10
CONTROLLER_OUPUT_MIN = 0

###############################################################################
# SIMULATION
###############################################################################

def process(t, y, u, Kp, τp):
    y = y[0] # Extract values
    dydt = (1 / τp) * (-(y - yss) + Kp * (u - uss))
    return [dydt]


for i in range(len(t) - 1):
    
    # Store the error in the ith slot 
    error[i] = sps[i] - y
    
    # Store accumulated error in the ith slot 
    if i > 0:
        ierror[i] = ierror[i-1] + error[i] * dt
    
    P = Kc * error[i]
    I = (Kc / τI) * ierror[i]
    
    # The bias is added only the first time
    if i == 0:
        u = ubias + P + I
    else:
        u = P + I

    if u > CONTROLLER_OUPUT_MAX:
        # Avoid controller output saturating the actuator
        u = CONTROLLER_OUPUT_MAX
        # Anti integration windup
        ierror[i] = ierror[i] - error[i] * dt
    if u < CONTROLLER_OUPUT_MIN:
        # Avoid controller output saturating the actuator
        u = CONTROLLER_OUPUT_MIN
        # Anti integration windup
        ierror[i] = ierror[i] - error[i] * dt
    
    # Store the controller output in the ith slot
    us[i] = u

    # Implement time delay
    iu = max(0, i - ndelay)

    # Prepare inputs for the solver: 
    # Lagged controller output, model parameters
    inputs = [us[iu], Kp, τp]
    
    t0, tf = t[i], t[i+1]

    sol = solve_ivp(
        fun      = process,  # A callable defined above
        t_span   = (t0, tf), # Interval of integration
        y0       = y0,       # Initial conditions
        method   = 'LSODA',  # 'LSODA': equivalent to odeint; 'RK45': good
        max_step = 0.1,      # Required if the state is zero for too long
        args     = inputs    # Additional arguments for the model
    )
    
    y = sol.y[0, -1]
    y0 = [y, ]
    
    # Store the output in the i+1 slot
    ys[i+1] = y  

# Fill the last unused slot
us[t_sim * resolution]     = us[t_sim * resolution - 1]
ierror[t_sim * resolution] = ierror[t_sim * resolution - 1]
error[t_sim * resolution]  = error[t_sim * resolution - 1]


###############################################################################
# PLOTTING
###############################################################################

plt.figure(1, figsize=(12,5))

# Output
plt.subplot(4,1,1)
plt.plot(t, ys,  'r-', linewidth=3, label='Process Variable (PV)')
plt.plot(t, sps, 'k:', linewidth=3, label='Setpoint (SP)')
plt.ylabel('Process output')
plt.legend()

# Input
plt.subplot(4,1,2)
plt.plot(t, us, 'b-', linewidth=1, label='Controller Output (OP)')
plt.ylabel('Process input')    
plt.legend()

# Error
plt.subplot(4,1,3)
plt.plot(t, error, 'g-', linewidth=3, label='Error')
plt.ylabel('Error')
plt.xlabel('Time (sec)')
plt.legend()

# Integral of error
plt.subplot(4,1,4)
plt.plot(t, ierror, 'g-', linewidth=3, label='Integral of error')
plt.ylabel('Integral of error')
plt.xlabel('Time (sec)')
plt.legend()

plt.show()