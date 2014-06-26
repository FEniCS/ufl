

from ufl import *
from ufl.algorithms import *

cell = triangle
x, y = cell.x

e = FiniteElement("CG", cell, 1)
v = TestFunction(e)

f0 = as_vector((x, -y))
f1 = as_vector((x, y))

for f in (f0, f1):
    print(("f = ", f))
    print() 
    print("take 1:")
    a = (1+rot(f))*v*dx
    print(("a1 = ", a))
    print() 
    print((a.form_data()))
    print(("a2 = ", expand_indices(a.form_data().form)))

    print() 
    print("take 2:")
    a = rot(f)*v*dx
    print(("a1 = ", a))
    print() 
    print((a.form_data()))
    print(("a2 = ", expand_indices(a.form_data().form)))
