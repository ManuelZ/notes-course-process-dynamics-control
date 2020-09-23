import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

m = GEKKO()
m.time = np.linspace(0, 40, 401)

u_step = np.zeros(401)
u_step[100:] = 2
u = m.Param(value=u_step)  

y = m.Var(1)

m.Equation(5 * y.dt() == -y + u)

# solve ODEs and plot
m.options.IMODE = 4
m.options.TIME_SHIFT=0

m.solve()
plt.plot(m.time, y, 'r-', linewidth=2, label='Output y(t)')
plt.plot(m.time, u, 'b-', linewidth=2, label='Input u(t)')

print(f'Final value: {np.max(y):.3f}')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()
