
from ufl import *
fe = VectorElement("CG", triangle, 1)
f = Coefficient(fe)
u0, u1 = split(f)
f2 = as_vector((u1, u1))
print str(replace(u0*dx, {f: f2}))

