import numpy as np
from matplotlib import pyplot as plt
from operators.derivative import diff_n1_e4

axis = np.linspace(0, 1, 1001)


def initial_f(a, sf):
    return np.exp(- sf * (a - 0.5) * (a - 0.5))


x = initial_f(axis, 500)
l = [x]
for i in range(5):
    l.append(diff_n1_e4(l[-1], 0.001))

fig = plt.figure()

for i, x in enumerate(l,1):
   graph = fig.add_subplot(2,3,i)
   graph.plot(axis,x)
   graph.set_xlim(0.3,0.7)
plt.show()