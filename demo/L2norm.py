#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, dx, triangle

element = LagrangeElement(triangle, 1)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

f = Coefficient(space)

a = f**2 * dx
