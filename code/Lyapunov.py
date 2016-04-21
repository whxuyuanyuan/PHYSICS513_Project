# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 00:18:42 2016
Lyapunov exponent
@author: Zhetao Jia
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from math import *
from scipy.optimize import curve_fit

def func(x, a, b):
    return a * x + b

def dist(x1, y1, x2, y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def findT(xList, yList):
    tList = [0]
    Len = len(xList)
    for i in range(1, Len):
        #tList.append(dist(xList[i-1], yList[i-1], xList[i], yList[i]))
        tList.append(tList[-1]+dist(xList[i-1], yList[i-1], xList[i], yList[i]))
    return tList

def findPos(xList, yList, tList, tDiff, num):
    tSelect = []    
    xSelect = []
    ySelect = []
    Len = len(xList)
    for i in range(num):
        tSelect.append(i * tDiff)
    tIdx = 0.0
    xTemp = 0.0
    yTemp = 0.0
    for time in tSelect:
        print time
        for i in range(Len):
            if tList[i] <= time and tList[i+1] > time:             
                tIdx = i
                break
        xtemp = xList[i] + (time - tList[i])/(tList[i+1] - tList[i])*(xList[i+1] - xList[i])
        ytemp = yList[i] + (time - tList[i])/(tList[i+1] - tList[i])*(yList[i+1] - yList[i])
        xSelect.append(xtemp) 
        ySelect.append(ytemp)         
    return tSelect, xSelect, ySelect        

if True:
    x1 = np.load('../output/Exp2/x1.npy')
    y1 = np.load('../output/Exp2/y1.npy')
    t1 = findT(x1,y1)
    x2 = np.load('../output/Exp2/x2.npy')
    y2 = np.load('../output/Exp2/y2.npy')
    t2 = findT(x1,y1)

    tIntval = 0.01
    pointNum = 10000
    tS1, xS1, yS1 = findPos(x1, y1, t1, tIntval, pointNum)
    tS2, xS2, yS2 = findPos(x2, y2, t2, tIntval, pointNum)

    #plt.plot(xS1, tS1, 'ro', xS2, tS2, '*')
    xS12 = []
    rS12 = []
    for i in range(len(tS1)):

        xS12.append(xS1[i]-xS2[i])
        rS12.append(sqrt((xS1[i] - xS2[i])**2 + (yS1[i] - yS2[i])**2))

    np.save('../output/Exp2/tS1', np.array(tS1))
    np.save('../output/Exp2/rS12', np.array(rS12))

# load data
tS1 = np.load('../output/Exp2/tS1.npy')
rS12 = np.load('../output/Exp2/rS12.npy')


plt.semilogy(tS1, rS12)

rS12 = np.log(rS12[360: 2300])
popt, pcov = curve_fit(func, tS1[360: 2300], rS12)

t_fit = np.linspace(2, 30, 1000)
y_fit = np.exp(popt[0] * t_fit + popt[1])

plt.title(r'$\lambda = %.3f$' % popt[0])
plt.semilogy(t_fit, y_fit, color='r', linestyle='--', label='Linear fit')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$|\mathbf{x(t)}|$')
plt.savefig('../output/Exp2/Lyapunov.pdf')

plt.show()