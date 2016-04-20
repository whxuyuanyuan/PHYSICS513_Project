from sympy import *
from sympy.geometry import *

l = Line(Point(-5, -5), Point(5, 5))
c = Circle(Point(0, 0), 1)
sol = intersection(l, c)

print sol
