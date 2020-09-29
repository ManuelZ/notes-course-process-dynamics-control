import tclab
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Q_INPUT = 75

t_heat = 480 # seconds

# Simulation time
t = np.linspace(0, t_heat, t_heat*10)

###############################################################################
# MEASURE DATA
###############################################################################

try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    with tclab.TCLab() as lab:
        h = tclab.Historian(lab.sources)
        
        for i in tclab.clock(t_heat):
            if i < 30:
                lab.Q1(0)
            else:
                lab.Q1(Q_INPUT)
            
            h.update(i)
            print(f"Time: {i:.0f} sec")
    df = pd.DataFrame(h.log, columns=h.columns)
    df.to_csv('data.csv')

###############################################################################
# VARIABLES
###############################################################################

Kp = 0.63  # Process Gain (% / deg C)
θp = 10    # Process Dead time (secs)
τp = 140   # Process Time constant (secs)


###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

Qss = 0
Q = np.zeros(len(t)) # %
Q[300:] = Q_INPUT
# Create linear interpolation of the data versus time
Qf = interp1d(t, Q)


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

def heater(t, y, Q):
    T, = y

    # Magic from http://apmonitor.com/pdc/index.php/Main/FirstOrderGraphical
    if (t-θp) <= 0:
        Qm = Qf(0)
    else:
        Qm = Qf(t-θp)

    dTdt = (-(T - Tss) + Kp * Qm) / τp
    return [dTdt]


for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = [Q[i]]
    sol = solve_ivp(heater, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)  
    
    # Store results
    T[i+1] = sol.y[0, -1]
    
    # Adjust initial condition for next loop
    y0 = sol.y[:, -1]


###############################################################################
# PLOTTING
###############################################################################
import matplotlib.ticker as ticker

fig, ax = plt.subplots()
plt.plot(t, Q, 'k:', linewidth=2, label='Heater input %')
plt.plot(t, T, 'b-', linewidth=2, label='FOPDT')
plt.plot(df.Time, df.T1, 'r.', label='Measured')
plt.ylabel('Temperature (°C)')
# From https://www.oreilly.com/library/view/matplotlib-plotting-cookbook/9781849513265/ch03s11.html
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.xlabel('Time (sec)')
plt.legend()

plt.show()