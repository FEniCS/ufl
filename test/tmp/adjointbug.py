
from dolfin import *

mesh = UnitSquare(3,3)

V1 = FunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V1)
v = TestFunction(V2)

u2 = Argument(V1)
v2 = Argument(V2)

print 'counts:'
print u.count(), v.count()
print u2.count(), v2.count()
#-1 -2
#1 2

a = u*v*dx
A = assemble(a)
print 'A size:'
print A.size(0), A.size(1)
#49 16

# Old buggy version, won't work with pydolfin and new ufl
if 1:
    b = adjoint(a)
    B = assemble(b)
    print 'B size:'
    print B.size(0), B.size(1)
    #49 16

# New with fix, won't work with old ufl
if 1:
    b = adjoint(a, (u2, v2))
    B = assemble(b)
    print 'B size again:'
    print B.size(0), B.size(1)
    #16 49

