
from ufl import *

from ufl.algorithms import *

def show(a):
    print "------------------------------------------"
    print
    print a
    print
    da = expand_derivatives(a)
    print da
    print
    print tree_format(da)
    print
    print forms2latexdocument([("a", a), ("da", da)], "")
    print
cell = triangle
element = FiniteElement("CG", cell, 1)
w = Function(element)

print "1 ========================================================================="
f = w*dx
show(f)

F = derivative(f, w)
show(F)

print "2 ========================================================================="
f = w**2*dx
show(f)

F = derivative(f, w)
show(F)

J = derivative(F, w)
show(J)

print "3 ========================================================================="
# f = (Dw : Dw) / 2
f = inner(grad(w), grad(w))/2*dx
show(f)

# F = (Dv : Dw + Dw : Dv) / 2 = (2 * Dv : Dw) / 2
F = derivative(f, w)
show(F)

J = derivative(F, w)
show(J)

print "4 ========================================================================="
Jstar = adjoint(J)
show(Jstar)

Jstaraction = action(Jstar)
show(Jstaraction)

