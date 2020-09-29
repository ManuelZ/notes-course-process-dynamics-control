import tclab
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Q_INPUT = 75

t_heat = 300 # seconds

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
            lab.Q1(Q_INPUT)
            h.update(i)
            print("Time:", i, 'seconds')
    df = pd.DataFrame(h.log, columns=h.columns)
    df.to_csv('data.csv')

###############################################################################
# VARIABLES
###############################################################################

Ta = 21.5          # deg C
U  = 5             # W/m^2-K
A  = 0.0012        # m^2
TaK = Ta + 273.15  # deg K
ε  = 0.9           # no units
σ  = 5.67e-8       # W/m^2-K^4
α  = 0.01          # W/%
m  = 0.004         # kg
cp = 500           # J/kg-K


###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

Q = np.ones(len(t)) * Q_INPUT # %


###############################################################################
# INITIAL CONDITIONS
###############################################################################

y0 = [Ta, Ta]


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

T  = np.ones(len(t)) * Ta
Tlin  = np.ones(len(t)) * Ta


###############################################################################
# SIMULATION
###############################################################################

T0 = 23             # Temperature in steady state [K]
Q0 = 0              # %

def heater(t, y, Q):
    T,Tlin = y
    TK = T + 273.15
    dTdt = (1 / (m * cp)) * (U * A * (Ta - T) + 
                             ε * σ * A * (np.power(TaK, 4) - np.power(TK, 4)) + 
                             α * Q)

    γ = (-U*A - 4*ε*σ*A*((T0+273.15) ** 3)) / (m * cp)
    β = α / (m * cp)
    dTdtlin = γ * (Tlin - T0) + β * (Q - Q0)
    return [dTdt, dTdtlin]


for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = [Q[i]]
    sol = solve_ivp(heater, (t0, tf), y0, method='LSODA', max_step=0.1, args=inputs)  
    
    # Store results
    T[i+1]    = sol.y[0, -1]
    Tlin[i+1] = sol.y[1, -1]
    
    # Adjust initial condition for next loop
    y0 = sol.y[:, -1]


###############################################################################
# PLOTTING
###############################################################################

plt.plot(t, Q, 'k:', linewidth=2, label='Heater input %')
plt.plot(t, T, 'b-', linewidth=2, label='Simulated nonlinear')
plt.plot(t, Tlin, 'k:', linewidth=2, label='Simulated linear')
plt.plot(df.Time, df.T1, 'r.', label='Measured')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.show()