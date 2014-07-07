
from ufl import *
from ufl.algorithms import *

# Function spaces
X = VectorElement('CG', triangle, 1)
Y = FiniteElement('CG', triangle, 1)
XY = X + Y

# Solution function (in mixed space)
u = Function(XY)

# Basis functions (in mixed space)
vv = TestFunction(XY)
uu = TrialFunction(XY)

# Form coefficients
for do_split in (True, False):
    if do_split:
        x, y = split(u)
    else:
        x = Function(X)
        y = Function(Y)

    # Forms (ok! means verified in detail by hand)
    #L = dot(x,x)*dx # ok!
    #L = dot(x,x)*y*dx # ok!
    #L = inner(grad(x),grad(x))*dx # ok!
    L = inner(grad(x), grad(x))*dx + dot(x, x)*y*dx # ok!
    
    if do_split:
        F = derivative(L, u, vv)
        J = derivative(F, u, uu)
    else:
        F = derivative(L, (x, y), vv)
        J = derivative(F, (x, y), uu)

    print(("="*80))
    print(do_split)
    print() 
    print() 
    print("F")
    print((str(F.form_data().form)))
    print() 
    print((str(expand_indices(F.form_data().form))))
    print() 
    print("J")
    print((str(J.form_data().form)))
    print() 
    print((str(expand_indices(J.form_data().form))))
    print() 

