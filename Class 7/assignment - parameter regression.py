import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d


###############################################################################
# PROCESS DATA
###############################################################################

url = 'https://apmonitor.com/pdc/uploads/Main/pipeline_data.txt'

try:
    with open('pipeline_data.txt') as f:
        data = pd.read_csv(f)
except FileNotFoundError:
    data = pd.read_csv(url)
    data.to_csv('pipeline_data.txt')

time = 'Time (min)'
valve = 'Valve Position (% open)'
TC = 'Temperature (degC)'

t = data.loc[:, time] / 60
u = data.loc[:, valve]
y = data.loc[:, TC]
# Create linear interpolation of the u data versus time
uf = interp1d(t, u)

###############################################################################
# INITIAL CONDITIONS
###############################################################################

u0  = data.loc[0, valve]
y0 = data.loc[0, TC]


###############################################################################
# SIMULATION
############################################################################### 

def fopdt_model(t, y, uf, Kp, θp, τp):
    y, = y
    
    try:
        if (t - θp) <= 0:
            um = uf(0)
        else:
            um = uf(t - θp)
    except:
        um = u0

    dydt = (-(y - y0) + Kp * (um - u0)) / τp
    
    return dydt


def sim_model(model, x, y0):
    
    # The current parameters to evaluate
    Kp, θp, τp = x

    # Storage for model values
    ym = np.zeros(len(t))

    for i in range(len(t) - 1):
        t0, tf = t[i], t[i+1]
        inputs = [uf, Kp, θp, τp]
        sol = solve_ivp(model, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)

        # Store results
        ym[i+1] = sol.y[0, -1]
        
        # Adjust initial condition for next loop
        y0 = sol.y[:, -1]
    
    return ym


def objective(x):
    ym = sim_model(fopdt_model, x, [y0])

    obj = np.sum(np.power(ym - y, 2))

    return obj


###############################################################################
# INITIAL GUESS
###############################################################################

x0 = [
    0.45, # Kp
    0.1,  # θp
    3     # τp
]

###############################################################################
# OPTIMIZE
###############################################################################

print(f'Initial SSE Objective: {objective(x0):.3f}')

# Optimize Kp, θp, τp
solution = minimize(objective, x0)
x = solution.x

print(f'Final SSE Objective: {objective(x)}')

print(f'Kp: {x[0]:.3f}')
print(f'θp: {x[1]:.3f}')
print(f'τp: {x[2]:.3f}')

# Calculate model with updated parameters
ym1 = sim_model(fopdt_model, x0, [y0])
ym2 = sim_model(fopdt_model, x, [y0])


###############################################################################
# PLOT RESULTS
###############################################################################

plt.figure()

plt.subplot(2,1,1)
plt.plot(t, u, 'b--')
plt.ylabel(valve)

plt.subplot(2,1,2)
plt.plot(t, y, 'r.', label='Measurement')
plt.plot(t, ym1, 'k--', label='Graphical fit')
plt.plot(t, ym2, 'b-', label='Regression fit')
plt.ylabel(TC)
plt.xlabel('Time (hr)')
plt.legend()

plt.show()