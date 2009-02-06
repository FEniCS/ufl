
from ufl import *

element = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(element)
u = TrialFunction(element)
f = Function(element)

forms = [\
#   f*dx,
#   v*dx,
#   (f+v)*dx,
#   v*u*dx,
#   (v + u*v)*dx,
#   (f+v+v*u)*dx,
#   f*v*u*dx,
#   (f*v + f*u*v)*dx,
    v*(u + f)*dx,
    v*(u - f)*dx,
    v*(f - u)*dx,
    f*((f*v)*(f*u))*dx,
    f*((2*v)*(f+u))*dx,
    f*((2*v)*(f-u))*dx,
    ]

for b in forms:
    print
    print "Trying form:", str(b)
    a = lhs(b)
    print "Lhs =", str(a)
    L = rhs(b)
    print "Rhs =", str(L)
    c = functional(b)
    print "Functional =", str(c)
    print

