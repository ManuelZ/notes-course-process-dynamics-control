import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

m = GEKKO()
m.time = np.linspace(0, 10)

t = m.Var(0)
x = m.Var(0)
y = m.Var(0)

m.Equation(t.dt() == 1)
m.Equation(x.dt() == 3 * m.exp(-t))
m.Equation(y.dt() == 3 - y)

m.options.IMODE = 4
m.options.NODES = 3
m.solve()

plt.plot(t, x, 'b-', label=r'$\frac{dx}{dt}= 3 \ \exp(-t)$')
plt.plot(t, y, 'r--', label=r'$\frac{dy}{dt}= -y + 3$')
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()
