
from ufl import *
cell = triangle
d = cell.geometric_dimension()
e = VectorElement("CG", cell, 2)
v = TestFunction(e)
u = Function(e)
I = Identity(d)
F = I + grad(u).T
C = F.T*F
C = variable(C)
f = diff(tr(C), C)
#a = inner(f, grad(v))*dx
a = inner(f, f)*dx

print("a:")
print(a)
fd = a.form_data()
print("fd:")
print(fd)
print("fd form:")
print(fd.form)

from ufl.algorithms import expand_indices
print("fd form expanded:")
print(expand_indices(fd.form))

