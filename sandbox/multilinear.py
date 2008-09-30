# Testing is_multilinear (will move to tests when ready)

from ufl import *
from ufl.algorithms import is_multilinear, monomials
from ufl.common import lstr

element = FiniteElement("Lagrange", "triangle", 1)

v = BasisFunction(element)
u = BasisFunction(element)
c = Function(element)

a = v*(u + v)*dx + v*ds
b = v/u*dx
c = c*v*(u + 2*u)*dx + c*dot(grad(v), grad(u))*ds
d = v*dot(grad(v), grad(u))*dx

print a
print is_multilinear(a)
print ""

print b
print is_multilinear(b)
print ""

print c
print is_multilinear(c)
print ""

print d
print is_multilinear(d)
