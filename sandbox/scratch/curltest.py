
from ufl import *
from ufl.classes import Form
from ufl.algorithms import *

cell = tetrahedron
x, y, z = cell.x

e = VectorElement("CG", cell, 1)
v = TestFunction(e)

f0 = as_vector(( y, -x,  0))
f1 = as_vector(( z,  0, -x))
f2 = as_vector(( 0,  x, -y))
f3 = as_vector(( x,  y,  z))

a0 = -2*v[2]*dx
a1 = +2*v[1]*dx
a2 = (v[2] - v[0])*dx
a3 = Form([])

for f, b in zip((f0, f1, f2, f3), (a0, a1, a2, a3)):
    print(("."*80))
    print(("f = ", f))
    
    a = dot(curl(f), v)*dx
    print(("\na = ", a))

    a = expand_compounds(a)
    print(("\na = ", a))
    
    a = expand_derivatives(a)
    print(("\na = ", a))

    a = renumber_indices(a)
    print(("\na = ", a))

    a = expand_indices(a)
    print(("\na = ", a))

    print((a == b))

#
#curl (y, -x, 0) = 
#|  i   j   k |
#| ,x  ,y  ,z |
#|  y  -x   0 |
#=
#0i + 0j + (-x,x -y,y)k = -2k
#

