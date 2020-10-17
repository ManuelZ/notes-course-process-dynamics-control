import tclab
import numpy as np
import matplotlib.pyplot as plt
from tuning import get_tuning_correlation

t_heat = 200 # seconds 
MAX_Q = 75


###############################################################################
# INITIAL CONDITIONS
###############################################################################

# Temperature in Steady state 
Tss = 23 # degrees celcius


###############################################################################
# STORAGE FOR THE RESULTS
###############################################################################

Ts     = np.ones(t_heat + 1) * Tss
Qs     = np.zeros(t_heat + 1)
errors = np.zeros(t_heat + 1)


###############################################################################
# CONTROLLER PARAMETERS
###############################################################################

# Setpoint with step after ten seconds
SPs = np.ones(t_heat + 1) * Tss
SPs[10:] = 60

# Since the initial input Q is zero before the controller is used, the bias is
# also zero when the controller kicks in
Qbias = 0 


Kp = 0.682
θp = 22.517
τp = 147.268

# The gain of the controller
Kc = get_tuning_correlation(Kp, θp, τp, method='setpoint_tracking')

print(f'Proportional gain: {Kc}')


###############################################################################
# CONTROL
############################################################################### 

# Simulation time
t = np.linspace(0, t_heat, t_heat + 1)

plt.figure(1, figsize=(12,5))
plt.ion()
plt.show()

with tclab.TCLab() as lab:
    h = tclab.Historian(lab.sources)

    for i in range(t_heat):
        
        SP = SPs[i]
        T = lab.T1
        error = SP - T
        
        Q = Qbias + Kc * error
        Q = MAX_Q if Q > MAX_Q else Q
        Q = 0 if Q < 0 else Q
        
        lab.Q1(Q)
        
        Ts[i+1]     = T     # Store temperature
        errors[i+1] = error # Store error
        Qs[i+1]     = Q     # Store Q
        
        print(f"Time: {i:.0f} s")
        print(f'Setpoint: {SP}; T: {T}; error: {error}; Q: {Q}')
        
        
        #######################################################################
        # ONLINE PLOTTING
        #######################################################################
        
        plt.clf()

        # Temperature and setpoint
        plt.subplot(3,1,1)
        plt.plot(t[0:i+1], Ts[0:i+1],'r-', linewidth=3, label='Temperature PV')
        plt.plot(t[0:i+1], SPs[0:i+1],'k:', linewidth=3, label='Temperature SP')
        plt.ylabel('Temperature [$^\circ$C]')
        plt.legend(loc='best')
        
        # Heater input
        plt.subplot(3,1,2)
        plt.plot(t[0:i+1], Qs[0:i+1],'b--', linewidth=3, label='Q input')
        plt.ylabel('Q')    
        plt.legend(loc='best')
        
        # Error
        plt.subplot(3,1,3)
        plt.plot(t[0:i+1], errors[0:i+1], 'g-', linewidth=3, label='error')
        plt.ylabel('Error = SP-PV')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')
        plt.ylim(bottom=0)
        
        plt.pause(1)

        
###############################################################################
# SAVE DATA
###############################################################################

data = np.vstack((t, Qs, Ts, SPs)).T
np.savetxt('P-only.csv', data, delimiter=',', header='Time,Q,T,SP', comments='')