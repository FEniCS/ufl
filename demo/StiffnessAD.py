#
# Author: Martin Sandve Alnes
# Date: 2008-10-30
#
from utils import LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    action,
    adjoint,
    derivative,
    dx,
    grad,
    inner,
    triangle,
)

element = LagrangeElement(triangle, 1)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

w = Coefficient(space)

# H1 semi-norm
f = inner(grad(w), grad(w)) / 2 * dx
# grad(w) : grad(v)
b = derivative(f, w)
# stiffness matrix, grad(u) : grad(v)
a = derivative(b, w)

# adjoint, grad(v) : grad(u)
astar = adjoint(a)
# action of adjoint, grad(v) : grad(w)
astaraction = action(astar)

forms = [f, b, a, astar, astaraction]
