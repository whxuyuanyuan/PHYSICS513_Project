#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Lorentz gas simulation.
author: Yuanyuan Xu
date: 04.16.2016
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import brentq
from scipy.optimize import brenth
from scipy import optimize
from sympy import *
from sympy.geometry import *
import os

def make_path(path):
    """
    Create the output diretory.
    :param path: path of the directory.
    :return: null.
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise Exception('Problem creating output dir %s !!!\nA file with the same name probably already exists, please fix the conflict and run again.' % output_path)


def norm(angle):
    if angle > pi / 2:
        norm(angle - pi)
    elif angle < -pi / 2:
        norm(angle + pi)
    else:
        return angle


def simulate(R, a, is_circle, steps, x0, y0, th0):
    """
    :param R: Radius of the inner circle.
    :param a: Half length of the side of the outside square.
    :param is_circle: Whether the start point is on the circle.
    :param steps: Total simulation steps.
    :param x0: Inital x-coordinate.
    :param y0: Initial y-coordinate.
    :param th0: Initial angle.
    :return: x, y, th arrays.
    """
    x = [x0]
    y = [y0]
    th = [th0]

    c = Circle(Point(0, 0), R)

    for i in range(steps):
        print 'step %d' % i
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
    print 'Done'
    return x, y, th

##################### Make Path ############################
make_path('../output')

##################### Test #################################
if True:
    # set parameters
    a = 2.0
    R = 1.5
    x0 = a
    y0 = a / 3
    th0 = -np.pi * 0.8
    steps = 100
    is_circle = False

    # simulate
    x, y, th = simulate(R, a, is_circle, steps, x0, y0, th0)

    # plot the configuration
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

    # plot the trajectory
    plt.plot(x, y, linestyle='--')
    plt.show()

    # save the data
    top_path = '../output/test/'
    make_path(top_path)
    np.save(top_path + 'x.npy', x)
    np.save(top_path + 'y', y)
    np.save(top_path + 'th', th)

    # load the data
    x = np.load(top_path + 'x.npy', x)
    y = np.load(top_path + 'y.npy', y)
    th = np.load(top_path + 'th.npy', th)