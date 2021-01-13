import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# Total simulation time
t_sim = 1200      # seconds

# Resolution of the simulation
resolution = 10 # every 0.1 sec 

# Simulation time
t = np.linspace(0, t_sim, t_sim * resolution + 1)

###############################################################################
# INITIAL CONDITIONS
###############################################################################

yss = 0
y0 = [yss]


###############################################################################
# DEFINE THE INPUT
###############################################################################

uss = 0
u = np.ones(len(t)) * uss
u[10*resolution:600*resolution] = 10

# Create linear interpolation of the data versus time
uf = interp1d(t, u)


###############################################################################
# STORAGE FOR THE OUTPUT
###############################################################################

y = np.ones(len(t)) * yss


###############################################################################
# SIMULATION
###############################################################################

Kp = 2     # Process Gain (Δy/Δu)
τp = 200   # Process Time constant (63.2% in one τ)

# CHANGE THIS TO ADD DEAD-TIME
θp = 100     # Process Dead time (time it takes the output to respond to the input)

def model(t, y, uf, Kp, τp, θp):
    """
    """
    # Extract values
    y = y[0]

    # Magic from http://apmonitor.com/pdc/index.php/Main/FirstOrderGraphical
    if (t - θp) <= 0:
        um = uf(0)
    else:
        um = uf(t-θp)

    dydt = (1 / τp) * (-(y - yss) + Kp * (um - uss))

    return [dydt]


for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = [uf, Kp, τp, θp]
    
    sol = solve_ivp(
        fun      = model,    # A callable defined above
        t_span   = (t0, tf), # Interval of integration
        y0       = y0,       # Initial conditions
        method   = 'LSODA',  # 'LSODA': equivalent to odeint; 'RK45': good
        max_step = 0.1,      # Required if the state is zero for too long
        args     = inputs    # Additional arguments for the model
    )
    
    # Store results
    y[i+1] = sol.y[0, -1]
    
    # Adjust initial condition for next loop
    y0 = sol.y[:, -1]


###############################################################################
# PLOTTING
###############################################################################
plt.figure()

plt.subplot(2,1,1)
plt.plot(t, y, 'r-', linewidth=2,  label='y(t)')
plt.xlabel('Time [s]')
plt.ylabel('Output [units]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, u, 'k:', linewidth=2,  label='Input')
plt.xlabel('Time [s]')
plt.legend()

plt.show()