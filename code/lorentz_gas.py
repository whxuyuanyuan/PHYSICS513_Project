import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import brentq
from scipy.optimize import brenth
from scipy import optimize
from sympy import *
from sympy.geometry import *


def norm(angle):
    if angle > pi / 2:
        norm(angle - pi)
    elif angle < -pi / 2:
        norm(angle + pi)
    else:
        return angle

R = 1.0
a = 2.0
is_circle = False

steps = 105

x = [-a]
y = [-a / 3.0]
th = [np.pi * 0.2]

c = Circle(Point(0, 0), R)

for i in range(steps):
    print i
    k = tan(th[-1])
    if not is_circle:
        l = Line(Point(x[-1], y[-1]), Point(x[-1] + cos(th[-1]) * 10.0, y[-1] + sin(th[-1]) * 10.0))
        p_array = intersection(l, c)
        if len(p_array) > 1:
            is_circle = True
            if Point(x[-1], y[-1]).distance(p_array[0]) < Point(x[-1], y[-1]).distance(p_array[1]):
                p = p_array[0]
            else:
                p = p_array[1]
            x.append(float(p.x))
            y.append(float(p.y))
            normal = np.array([x[-1], y[-1]])
            tangent = np.array([-y[-1], x[-1]])
            velocity = np.array([cos(th[-1]), sin(th[-1])])
            n_c = -np.dot(normal, velocity)
            t_c = np.dot(tangent, velocity)
            new = n_c * normal + t_c * tangent
            th_ref = norm(atan(new[1] / new[0]))
            if new[0] > 0:
                th.append(th_ref)
            elif new[1] < 0:
                th.append(th_ref - np.pi)
            else:
                th.append(th_ref + np.pi)

        else:
            is_circle = False
            if 0 < th[-1] <= pi / 2:
                if (a - y[-1]) / (a - x[-1]) > k:
                    y.append(k * (a - x[-1]) + y[-1])
                    x.append(a)
                    th.append(pi - th[-1])
                    continue
                else:
                    x.append((a - y[-1]) / k + x[-1])
                    y.append(a)
                    th.append(-th[-1])
                    continue
            if pi / 2 < th[-1] <= pi:
                if (a - y[-1]) / (-a - x[-1]) > k:
                    x.append((a - y[-1]) / k + x[-1])
                    y.append(a)
                    th.append(-th[-1])
                    continue
                else:
                    y.append(k * (-a - x[-1]) + y[-1])
                    x.append(-a)
                    th.append(pi - th[-1])
                    continue
            if -pi / 2 < th[-1] <= 0:
                if (-a - y[-1]) / (a - x[-1]) > k:
                    x.append((-a - y[-1]) / k + x[-1])
                    y.append(-a)
                    th.append(-th[-1])
                    continue
                else:
                    y.append(k * (a - x[-1]) + y[-1])
                    x.append(a)
                    th.append(-(pi + th[-1]))
                    continue
            if -pi < th[-1] <= -pi / 2:
                if (-a - y[-1]) / (-a - x[-1]) > k:
                    y.append(k * (-a - x[-1]) + y[-1])
                    x.append(-a)
                    th.append(-(pi + th[-1]))
                    continue
                else:
                    x.append((-a - y[-1]) / k + x[-1])
                    y.append(-a)
                    th.append(-th[-1])
                    continue
    else:
        is_circle = False
        if 0 < th[-1] <= pi / 2:
            if (a - y[-1]) / (a - x[-1]) > k:
                y.append(k * (a - x[-1]) + y[-1])
                x.append(a)
                th.append(pi - th[-1])
                continue
            else:
                x.append((a - y[-1]) / k + x[-1])
                y.append(a)
                th.append(-th[-1])
                continue
        if pi / 2 < th[-1] <= pi:
            if (a - y[-1]) / (-a - x[-1]) > k:
                x.append((a - y[-1]) / k + x[-1])
                y.append(a)
                th.append(-th[-1])
                continue
            else:
                y.append(k * (-a - x[-1]) + y[-1])
                x.append(-a)
                th.append(pi - th[-1])
                continue
        if -pi / 2 < th[-1] <= 0:
            if (-a - y[-1]) / (a - x[-1]) > k:
                x.append((-a - y[-1]) / k + x[-1])
                y.append(-a)
                th.append(-th[-1])
                continue
            else:
                y.append(k * (a - x[-1]) + y[-1])
                x.append(a)
                th.append(-(pi + th[-1]))
                continue
        if -pi < th[-1] <= -pi / 2:
            if (-a - y[-1]) / (-a - x[-1]) > k:
                y.append(k * (-a - x[-1]) + y[-1])
                x.append(-a)
                th.append(-(pi + th[-1]))
                continue
            else:
                x.append((-a - y[-1]) / k + x[-1])
                y.append(-a)
                th.append(-th[-1])
                continue

fig = plt.gcf()
fig.set_size_inches(5, 5)
plt.plot([-a, a], [-a, -a], color='black')
plt.plot([a, a], [-a, a], color='black')
plt.plot([a, -a], [a, a], color='black')
plt.plot([-a, -a], [a, -a], color='black')
circle = plt.Circle((0, 0), R, color='black')
fig.gca().add_artist(circle)
plt.xlim([-2.2, 2.2])
plt.ylim([-2.2, 2.2])

plt.plot(x, y)
plt.show()
