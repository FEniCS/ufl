# Testing is_multilinear (will move to tests when ready)

from ufl import *
from ufl.algorithms import is_multilinear

element = FiniteElement("Lagrange", "triangle", 1)

v = BasisFunction(element)
u = BasisFunction(element)
c = Function(element)

a = v*(u + v)*dx + v*ds
#b = v/u*dx
#a = c*v*(u + v)*dx + c*v*dx

print is_multilinear(a)
#print is_multilinear(b)
