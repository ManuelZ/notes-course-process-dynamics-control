import time
import tclab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d

t_heat = 14 * 60 # seconds

# Simulation time
t = np.linspace(0, t_heat, t_heat)


###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

Qss = 0
Q = np.zeros(len(t)) # %
Q[30:] = 35.0
Q[270:] = 70.0
Q[450:] = 10.0
Q[630:] = 60.0
Q[800:] = 0.0
# Create linear interpolation of the data versus time
Qf = interp1d(t, Q)


###############################################################################
# MEASURE DATA
###############################################################################

try:
    df = pd.read_csv('tclab_data.csv')
except FileNotFoundError:
    with tclab.TCLab() as lab:
        h = tclab.Historian(lab.sources)
        
        for i in tclab.clock(t_heat - 1, step=1, tol=0.5):
            lab.Q1(Q[int(i)])
            h.update(i)
            print(f"Time: {i:.0f} sec")
    df = pd.DataFrame(h.log, columns=h.columns)
    df.to_csv('tclab_data.csv')

y = df.T1


###############################################################################
# INITIAL CONDITIONS
###############################################################################

Tss = 23 # degrees celcius
y0 = [Tss]


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

T = np.ones(len(t)) * Tss


###############################################################################
# SIMULATION
############################################################################### 

def fopdt_model(t, y, Qf, Kp, θp, τp):
    T, = y

    # Magic from http://apmonitor.com/pdc/index.php/Main/FirstOrderGraphical
    if (t-θp) <= 0:
        Qm = Qf(0)
    else:
        Qm = Qf(t-θp)

    dTdt = (-(T - Tss) + Kp * Qm) / τp
    return [dTdt]


def sim_model(model, x, y0):
    
    # The current parameters to evaluate
    Kp, θp, τp = x

    # Storage for model values
    ym = np.ones(len(t)) * y0

    for i in range(len(t) - 1):
        t0, tf = t[i], t[i+1]
        inputs = [Qf, Kp, θp, τp]
        sol = solve_ivp(model, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)

        # Store results
        ym[i+1] = sol.y[0, -1]
        
        # Adjust initial condition for next loop
        y0 = sol.y[:, -1]

    return ym


def objective(x):
    ym = sim_model(fopdt_model, x, y0)
    obj = np.sum(np.power(ym - y, 2))
    return obj


###############################################################################
# INITIAL GUESS FROM GRAPHICAL FIT
###############################################################################

x0 = [
    0.63,   # Kp
    10,     # θp
    140     # τp
]


###############################################################################
# OPTIMIZE
###############################################################################

start = time.time()
# Optimize Kp, θp, τp
solution = minimize(objective, x0, method='SLSQP')
x = solution.x
end = time.time()

print(f'Initial SSE Objective: {objective(x0):.3f}')
print(f'Final SSE Objective: {objective(x):.3f}')
print(f'Optimization took {end-start:.1f}s')

print(f'Kp: {x[0]:.3f}')
print(f'θp: {x[1]:.3f}')
print(f'τp: {x[2]:.3f}')

# Calculate model with updated parameters
ym1 = sim_model(fopdt_model, x0, y0)
ym2 = sim_model(fopdt_model, x, y0)


###############################################################################
# PLOTTING
###############################################################################

fig, ax = plt.subplots()

plt.plot(t, Q, 'k:', linewidth=2, label='Heater input %')
plt.plot(t, ym1, 'b--', linewidth=2, label='FOPDT - Graphical fit')
plt.plot(t, ym2, 'k-', linewidth=2, label='FOPDT - Regression fit')
plt.plot(df.Time, df.T1, 'r.', label='Measured')
plt.ylabel('Temperature (°C)')
# From https://www.oreilly.com/library/view/matplotlib-plotting-cookbook/9781849513265/ch03s11.html
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.xlabel('Time (sec)')
plt.legend()

plt.show()

# With the default minimization method:
# Initial SSE Objective: 6932.468
# Final SSE Objective: 268.07724751728335
# Optimization took 247.9s
# Kp: 0.682
# θp: 22.517
# τp: 147.268

# With method SLSQP:
# Initial SSE Objective: 6932.468
# Final SSE Objective: 268.07724753736386
# Optimization took 58.7s
# Kp: 0.682
# θp: 22.517
# τp: 147.268