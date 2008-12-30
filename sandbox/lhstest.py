
from ufl import *

e = FiniteElement("CG", triangle, 1)
v = TestFunction(e)
u = TrialFunction(e)
f = Function(e)

a = (u*v + f*v)*dx
print
print "a =", a
print "l =", lhs(a)
print "r =", rhs(a)

a = (u*v)*dx + (f*v)*ds
print
print "a =", a
print "l =", lhs(a)
print "r =", rhs(a)

