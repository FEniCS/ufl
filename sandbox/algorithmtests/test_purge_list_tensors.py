from ufl import *
from ufl.algorithms import purge_list_tensors

element = VectorElement("Lagrange", "triangle", 1)

v = TestFunction(element)
u = TrialFunction(element)

#a = inner(as_tensor([v[0], v[1]]), as_tensor([u[0], u[1]]))
a = inner(as_tensor([v[0], v[1]]), as_tensor([u[0], u[1].dx(0)]))*dx

fd = a.form_data()
b = fd.form
c = purge_list_tensors(b)
print str(c)

