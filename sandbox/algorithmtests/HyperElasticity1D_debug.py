#!/usr/bin/env python
from ufl import *
from ufl.algorithms import *
set_level(DEBUG)

cell = interval
element = FiniteElement("CG", cell, 2)
u = Function(element)
b = Constant(cell)
K = Constant(cell)

E = u.dx(0) + u.dx(0)**2 / 2
E = variable(E)
Q = b*E**2
psi = K*(exp(Q)-1)

f = psi*dx
F = derivative(f, u)
J = derivative(-F, u)

forms = [f, F, J]


print(("=== f" + "="*70))
print((str(f)))

print(("=== F" + "="*70))
print((str(F)))

print(("=== J" + "="*70))
print((str(J)))


print(("=== f" + "="*70))
print((str(strip_variables(f.form_data().form))))

print(("=== F" + "="*70))
print((str(strip_variables(F.form_data().form))))

print(("=== J" + "="*70))
print((str(strip_variables(J.form_data().form))))
print() 
print((str(expand_indices(strip_variables(J.form_data().form)))))

