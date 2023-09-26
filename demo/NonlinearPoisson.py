from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, dot, dx, grad, triangle
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
u0 = Coefficient(space)
f = Coefficient(space)

a = (1 + u0**2) * dot(grad(v), grad(u)) * dx \
    + 2 * u0 * u * dot(grad(v), grad(u0)) * dx
L = v * f * dx - (1 + u0**2) * dot(grad(v), grad(u0)) * dx
