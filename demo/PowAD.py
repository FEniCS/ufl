#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from utils import LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    derivative,
    dx,
    triangle,
)

element = LagrangeElement(triangle, 1)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
w = Coefficient(space)

L = w**5 * v * dx
a = derivative(L, w)
