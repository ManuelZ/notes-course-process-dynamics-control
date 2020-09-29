import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Simulate a step change in the gas pedal position, comparing the nonlinear 
# response to the linear response. 
# Comment on the accuracy of the linearized model.

t_sim = 60 # seconds

# Simulation time
t = np.linspace(0, t_sim, t_sim * 10) # seconds


###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

# Pedal %
u = np.zeros(len(t))
u[100:] = 50


###############################################################################
# INITIAL CONDITIONS
###############################################################################

v0 = 0
y0 = [v0]


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

v  = np.ones(len(t)) * v0
vlinear  = np.ones(len(t)) * v0


###############################################################################
# SIMULATION
###############################################################################

ρ = 1.225   # air density ; kg / m^3
A = 5       # car cross-sectional area ; m^2
Cd = 0.24   # drag coefficient ; no-units
m = 700     # vehicle mass ; kg
Fp = 30     # thrust parameter ; N/%

def automobile(t, y, u):
    v, = y
    dvdt = (1 / m) * (Fp * u - 0.5 * ρ * A * Cd * (v ** 2))
    return [dvdt]

for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = (u[i],)
    sol = solve_ivp(automobile, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)
    
    # Store results
    v[i+1]  = sol.y[0, -1]
    
    # Adjust initial condition for next loop
    y0 = sol.y[:, -1]


# Reset initial conditions
v0 = 0
y0 = [v0]
def linear_automobile(t, y, u):
    v, = y
    dvdt = -0.0848526 * (v - 40.406) + 0.042857 * (u - 40)
    return [dvdt]

for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = (u[i],)
    sol = solve_ivp(linear_automobile, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)
    
    # Store results
    vlinear[i+1]  = sol.y[0, -1]
    
    # Adjust initial condition for next loop
    y0 = sol.y[:, -1]



###############################################################################
# PLOTTING
###############################################################################

plt.plot(t, u, 'k--', linewidth=2, label='Pedal %')
plt.plot(t, v, 'b-', linewidth=2, label='Model')
plt.plot(t, vlinear, 'r:', linewidth=2, label='Linear model')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()

plt.show()