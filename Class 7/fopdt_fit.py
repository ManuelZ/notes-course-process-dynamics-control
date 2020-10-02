import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d


###############################################################################
# PROCESS DATA
###############################################################################

# Import CSV data file
# Column 1 = time (t)
# Column 2 = input (u)
# Column 3 = output (yp)
data = np.loadtxt('simulated_data.txt', delimiter=',')

t   = data[:, 0].T - data[0, 0]
yp  = data[:, 2].T
u   = data[:, 1].T
# Create linear interpolation of the u data versus time
uf = interp1d(t, u)


###############################################################################
# INITIAL CONDITIONS
###############################################################################

u0  = data[0, 1]
y0 = data[0, 2]


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


def sim_model(model, x):

    y0 = [data[0, 2]]
    
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
    ym = sim_model(fopdt_model, x)

    obj = np.sum(np.power(ym - yp, 2))

    return obj


###############################################################################
# INITIAL GUESS
###############################################################################

x0 = [
    10, # Kp
    10, # θp
    10  # τp
]

###############################################################################
# OPTIMIZE
###############################################################################

start = time.time()
# Optimize Kp, θp, τp
bounds = ((0,5), (1e-6, 5), (1e-6, 5))
solution = minimize(objective, x0, bounds=bounds, method='SLSQP')
end = time.time()


# Another way to solve: with bounds on variables
# bnds = ((0.4, 0.6), (1.0, 10.0), (0.0, 30.0))
# solution = minimize(objective,x0,bounds=bnds,method='SLSQP')
x = solution.x

print(f'Initial SSE Objective: {objective(x0):.3f}')
print(f'Final SSE Objective: {objective(x)}')
print(f'Optimization took {end-start:.1f}s')

print(f'Kp: {x[0]:.3f}')
print(f'θp: {x[1]:.3f}')
print(f'τp: {x[2]:.3f}')

# Calculate model with updated parameters
ym1 = sim_model(fopdt_model, x0)
ym2 = sim_model(fopdt_model, x)


###############################################################################
# PLOT RESULTS
###############################################################################

plt.figure()

plt.subplot(2,1,1)
plt.plot(t, yp, 'kx-', linewidth=2, label='Process Data')
plt.plot(t, ym1, 'b-', linewidth=2, label='Initial Guess')
plt.plot(t, ym2, 'r--', linewidth=3, label='Optimized FOPDT')
plt.ylabel('Output')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, u, 'bx-', linewidth=2, label='Measured')
plt.plot(t, uf(t), 'r--', linewidth=3, label='Interpolated')
plt.ylabel('Input Data')
plt.legend()

plt.show()