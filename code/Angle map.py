# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 06:59:37 2016

@author: Eric J
"""
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import brentq
from scipy.optimize import brenth
from scipy import optimize
from sympy import *
from sympy.geometry import *

def dist(x1, y1, x2, y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)
    
x1 = np.load('x1.npy')
y1 = np.load('y1.npy')

angleList = []
testx = []
testy = []
for i in range(size(x1)):
    xi = x1[i]
    yi = y1[i]
    if (xi!= -2.0) and (xi!=2.0) and (yi!=-2.0) and (yi!=2.0):
        angleList.append(math.atan2(yi,xi))   
        testx.append(xi)
        testy.append(yi)

        
#plt.scatter(testx,testy)
#plt.hist(angleList, bins = 30)
anglePairx = []
anglePairy = []
for idx in range(size(angleList)-1):
    anglePairx.append(angleList[idx])
    anglePairy.append(angleList[idx+1])
plt.scatter(anglePairx, anglePairy)
plt.xlabel("theta i")
plt.ylabel("theta i+1")