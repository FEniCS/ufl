from ufl import *

element = VectorElement("Lagrange", "triangle", 1)
v = TestFunction(element)
u = TrialFunction(element)


print isinstance(v[i], Indexed)
print v[i].free_indices

#a = v[1]*u[0]*dx
#print a
