import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def model(t, y, k):
    dydt = -k * y
    return dydt

# Initial conditions
y0 = [5]

# Time span
t = [0, 20]

k = 0.1
sol1 = solve_ivp(model, t, y0, method='LSODA', args=(k,)) # LSODA is the same method used by ODEINT

k = 0.2
sol2 = solve_ivp(model, t, y0, method='LSODA', args=(k,)) # LSODA is the same method used by ODEINT

k = 0.5
sol3 = solve_ivp(model, t, y0, method='LSODA', args=(k,)) # LSODA is the same method used by ODEINT

# plot results
plt.plot(sol1.t, sol1.y[0],'r-',linewidth=2,label='k=0.1')
plt.plot(sol2.t, sol2.y[0],'b--',linewidth=2,label='k=0.2')
plt.plot(sol3.t, sol3.y[0],'g:',linewidth=2,label='k=0.5')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()