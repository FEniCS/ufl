# Testing is_multilinear (will move to tests when ready)

from ufl import *
from ufl.algorithms import is_multilinear

element = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(element)
u = TrialFunction(element)

a = v*(u + v)*dx + v*ds
b = v/u*dx

print is_multilinear(a)
print is_multilinear(b)
