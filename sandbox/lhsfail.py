
from ufl import *

element = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(element)
u = TrialFunction(element)
u0 = Function(element)

F = v*(u - u0)*dx 

a = lhs(F)
L = rhs(F)

print F
print a
print L

