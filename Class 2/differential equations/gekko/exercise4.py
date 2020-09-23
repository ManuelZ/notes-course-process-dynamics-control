import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

m = GEKKO()
m.time = np.linspace(0, 40, 4001)

u_step = np.zeros(4001)
u_step[500:] = 2
u = m.Param(value=u_step)  

x = m.Var(0)
y = m.Var(0)

m.Equation(2 * x.dt() == -x + u)
m.Equation(5 * y.dt() == -y + x)

m.options.IMODE = 4
m.options.NODES = 3
m.solve()

plt.plot(m.time, x, 'b-', label=r'$2\frac{dx}{dt}= -x(t)\ +\ u(t)$')
plt.plot(m.time, y, 'r--', label=r'$5\frac{dy}{dt}= -y(t)\ +\ x(t)$')
plt.plot(m.time, u, 'k-', linewidth=2, label=r'u(t)')
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()
