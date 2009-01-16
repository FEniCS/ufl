
from ufl import *
from ufl.algorithms import *

def show(a):
    print "------------------------------------------"
    print
    print a
    print
    print tree_format(a)
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
u = Function(element)
I = Identity(cell.dim())

print "1 ========================================================================="
# Win:
#F = 0.5*u*dx
#show(F)

# Win:
#F = grad(u)[0]*dx
#show(F)

# Win:
#F = grad(u)[i] * grad(u)[i] * dx
#show(F)

# Win:
#F = as_vector(u.dx(i), i)[j] * as_vector(u.dx(k), k)[j] * dx
#show(F)

# Win:
#F = as_vector(u.dx(i), i)[i] * I[i,0] * dx
#show(F)

# :
F = (0.5*u).dx(i) * I[i,0] * dx
show(F)

# Fail:
#F = (0.5*u).dx(i) * I[i,0] * dx
#show(F)

# Fail:
#F = as_vector((0.5*u).dx(i), i)[i] * I[i,0] * dx
#show(F)

# Fail:
#F = grad(0.5*u)[0]*dx
#show(F)

# Fail:
#F = dot(grad(0.5*u), grad(0.5*u))*dx
#show(F)

