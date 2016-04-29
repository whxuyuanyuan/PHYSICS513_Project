# -*- coding: utf-8 -*-
# coding: utf-8
#
import random
#import numpy as np
#import matplotlib.pyplot as plt
from math import *
#from scipy.optimize import brentq
#from scipy.optimize import brenth
#from scipy import optimize
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import os


def dist(x1, y1, x2, y2):
    return sqrt((x1-x2)**2+(y1-y2)**2)


def findProb(pi, ri, rc, xList, yList):
    xi, yi = pi[0], pi[1]
    xc, yc = 0.0, 0.0
    if (dist(xi, yi, xc, yc) < ri + rc):
        return "Error",dist(xi, yi, xc, yc), ri + rc
    totLen = 0.0
    prevX = None
    prevY = None
    for idx in range(size(xList)):
        if prevX == None:
            prevX = xList[0]
            prevY = yList[0]
            continue
        else:
            currX = xList[idx]
            currY = yList[idx]
            a = currY - prevY
            b = currX - prevX
            linedist = abs(a*xi+b*yi+currX*prevY-currY*prevX)/sqrt(a**2+b**2)
            #print linedist,currX
            if abs(linedist) < ri:
                totLen += 2.0 * math.sqrt(ri**2.0-linedist**2.0)
            prevX = currX
            prevY = currY
    print ri
    return totLen/(math.pi*ri**2)

x1 = np.load('x.npy')
y1 = np.load('y.npy')

def testUni(rc, d, runTime):
    xData, yData = [], []
    probData = []
    for i in range(runTime):
        print i
        pxi = random.uniform(-d, d)
        pyi = random.uniform(-d, d)
        ri = 0.05*d*random.random() 
        if (dist(pxi, pyi, 0, 0) > ri + rc):
            xData.append(pxi)
            yData.append(pyi)
            probData.append(findProb([pxi,pyi], ri, rc, x1, y1))
    return xData, yData, probData

xList, yList, zList = testUni(1, 2, 15)
# save the data
#zList = [i/maxZ for i in zList]
#maxZ = max(zList)
np.save('xList', xList)
np.save('yList', yList)
np.save('zList', zList)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xList, yList, zList)
#pylab.pcolor(xList, yList, zList)

# and a color bar to show the correspondence between function value and color
plt.show()
plt.savefig('uniform.png')


#print findProb([-1.5,-1.5], 0.5, 1, x1, y1)