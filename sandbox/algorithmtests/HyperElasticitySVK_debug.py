#!/usr/bin/env python
from ufl import *
from ufl.algorithms import *
set_level(DEBUG)

# Cell and its properties
#cell = tetrahedron
cell = triangle

# Elements
u_element = VectorElement("CG", cell, 1)
p_element = FiniteElement("CG", cell, 1)

# Test and trial functions
v = TestFunction(u_element)
w = TrialFunction(u_element)

# Displacement at current and two previous timesteps
u   = Function(u_element)
up  = Function(u_element)
upp = Function(u_element)

# Time parameters
dt = Constant(cell)

# Material parameters
lamda = Constant(cell)
mu = Constant(cell)

# Deformation gradient
I = Identity(cell.d)
F = I + grad(u).T

# Right Cauchy-Green deformation tensor
C = F.T*F

# Green strain tensor
E = (C-I)/2
E = variable(E)

# Strain energy function psi(Q(Ef)), Saint Vernant-Kirchoff law
psi = lamda/2 * tr(E)**2 + mu * inner(E, E)

# First Piola-Kirchoff stress tensor
P = F*diff(psi, E)

# Energy functional (without acceleration term and external forces)
a_f = psi*dx

# Residual equation
a_F = inner(P, grad(v))*dx \

# Jacobi matrix of residual equation
a_J = derivative(a_F, u, w)

forms = [a_f, a_F, a_J]

print "=== f" + "="*70
print str(a_f)

print "=== F" + "="*70
print str(a_F)

print "=== J" + "="*70
print str(a_J)


print "=== f" + "="*70
print str(strip_variables(a_f.form_data().form))

print "=== F" + "="*70
print str(strip_variables(a_F.form_data().form))

print "=== J" + "="*70
print str(strip_variables(a_J.form_data().form))
print 
#print str(expand_indices(strip_variables(a_J.form_data().form))) # pretty large expression...

