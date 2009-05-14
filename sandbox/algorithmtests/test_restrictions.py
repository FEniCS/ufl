
from ufl import *

e = FiniteElement("DG", triangle, 1)
f = Function(e)
g = Function(e)
a = (grad(f) + grad(g))('-')
print a

from ufl.algorithms import propagate_restrictions

b = propagate_restrictions(a)
print b
