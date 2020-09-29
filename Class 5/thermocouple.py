import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


t_sim = 0.1 # seconds

# Simulation time
t = np.linspace(0, t_sim, 1000) # seconds


###############################################################################
# INITIAL CONDITIONS
###############################################################################

Tf0 = 1500      # degrees Kelvin
y0 = [Tf0, Tf0] # For the non-linear and linear models


###############################################################################
# ADJUSTABLE VARIABLES
###############################################################################

# Flame temperature
Tf = np.ones(len(t)) * Tf0
Tf[500:] = 1520

# Cyclical gas temperature due to pulsed acoustic wave
A = 75  # Amplitude, degrees Kelvin
f = 100 # hz
Tg = Tf + A * np.sin(2 * np.pi * f * t) # Degrees Kelvin


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

Tt     = np.ones(len(t)) * Tf0
Ttlin  = np.ones(len(t)) * Tf0


###############################################################################
# SIMULATION
###############################################################################

def area(d): return 4 * np.pi * (d/2) ** 2

def vol(d): return (4/3) * np.pi * (d/2) ** 3

dt   = 1e-04       # Thermocouple’s diameter [m]
A    = area(dt)    # Thermocouple’s surface area [m^2]
V    = vol(dt)     # Thermocouple’s volume [m^3]
ρt   = 20000       # Thermocouple’s density [kg/m^3]
Cpt  = 400         # Thermocouple’s specific heat capacity [J/(kg∙K)]
U    = 2800        # Heat transfer coefficient [W/(m^2∙K)]
ε    = 0.8         # Emissivity factor [no units]
σ    = 5.67e-8     # Stefan–Boltzmann constant (5.67×10−8) [W/(m^2∙K^4)]

Tt0 = 1500         # Thermocouple’s temperature in steady state [K]
Tg0 = 1500         # Gas temperature in steady state [K]
Tf0 = 1500         # Flame temperature in steady state [K]

def thermocouple(t, y, Tf, Tg):
    Tt, Ttlin = y
    
    dTdt = ( U * A * (Tg - Tt) + ε * σ * A * ((Tf ** 4) - (Tt ** 4)) ) / (ρt * V * Cpt)

    α = -U * A - 4 * ε * σ * A * (Tt0 ** 3)
    β = U * A 
    γ = 4 * ε * σ * A * (Tf0 ** 3)
    dTdtlin = (α * (Ttlin - Tt0 ) + β * (Tg - Tg0) + γ * (Tf - Tf0)) / (ρt * V * Cpt)
    
    return [dTdt, dTdtlin]


for i in range(len(t) - 1):
    t0, tf = t[i], t[i+1]
    inputs = (Tf[i], Tg[i])
    sol = solve_ivp(thermocouple, (t0, tf), y0, method='LSODA', args=inputs)
    
    # Store results
    Tt[i+1] =    sol.y[0, -1]
    Ttlin[i+1] = sol.y[1, -1]
    
    # Adjust initial condition for next loop
    y0 = sol.y[:, -1]


###############################################################################
# PLOTTING
###############################################################################

plt.plot(t, Tg, 'b-', linewidth=2,  label='Gas temperature')
plt.plot(t, Tf, 'k--', linewidth=2, label='Flame temperature')
plt.plot(t, Tt, 'r-', linewidth=2,  label='Thermocouple temp')
plt.plot(t, Ttlin, 'g--', linewidth=2, label='Thermocouple temp (linear)')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.legend()

plt.show()