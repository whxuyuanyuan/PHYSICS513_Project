import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import curve_fit

t = []
time = 0
dt = 0.01
dx = []
x1 = 1.0
x2 = 1.0 + 2.0 * 10 ** -8
y1 = 0.0
y2 = 0.0 + 10 ** -8
z1 = 1.0
z2 = 1.0 + 10 ** -8

r = 28.0
sigma = 10.0
b = 8.0 / 3.0

def lorenz(x, y, z):
    return x + dt * sigma * (y - x), y + dt * (r * x - y - x * z), z + dt * (x * y - b * z)

def func(x, a, b):
    return a * x + b

while time < 30:
    t.append(time)
    time += dt
    dx.append(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))
    #dx.append(abs(x1 - x2))

    x1, y1, z1 = lorenz(x1, y1, z1)
    x2, y2, z2 = lorenz(x2, y2, z2)

plt.semilogy(t, dx)

t1 = np.array(t[int(12.5 / dt): int(18.5 / dt)])
dx = dx[int(12.5 / dt): int(18.5 / dt)]
dx = np.log(np.array(dx))
print len(dx), len(t1)
guess = np.array([0.9, -5])
popt, pcov = curve_fit(func, t1, dx, guess)
print popt[0], popt[1]
t = np.array(t)
y = np.exp(popt[0] * t + popt[1])
plt.semilogy(t, y, linestyle='--', color='r', linewidth=1.5, label='linear fit: y = 1.01x - 18.30')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\ln || \sigma ||$')
plt.title('9.3.9 Semilog Plot')
plt.savefig('9.3.9.pdf')