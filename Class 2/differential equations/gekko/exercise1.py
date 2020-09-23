import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

m = GEKKO()    # create GEKKO model
y = m.Var(0) # create GEKKO variable
m.Equation(y.dt() == -y + 1) # create GEKKO equation
m.time = np.linspace(0, 20) # time points

# solve ODEs and plot
m.options.IMODE = 4
m.options.TIME_SHIFT=0

m.solve()
plt.plot(m.time, y, 'r-', linewidth=2, label='Exercise 1')

print(f'Final value: {np.max(y):.3f}')

plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
