
from ufl import *
from ufl.classes import *
from ufl.algorithms import *

def printit(a, tree=False):
    print(("-"*80))
    print("a =")
    print((str(a)))
    if isinstance(a, str):
        return
    if tree:
        print() 
        print((tree_format(a)))

    a = renumber_indices(a)
    print() 
    print("renumbered a =")
    print((str(a)))
    if tree:
        print() 
        print((tree_format(a)))

    a = expand_indices(a)
    print() 
    print("expanded a =")
    print((str(a)))
    if tree:
        print() 
        print((tree_format(a)))
    print() 

velement = VectorElement("Lagrange", triangle, 1)
v = Function(velement)
telement = TensorElement("Lagrange", triangle, 1)
A = Function(telement)
I = Identity(2)

#a = as_tensor(v[i], i)[k] * I[k,0]
if True:
    a = A[j,q]
    b = as_tensor(a, q)
    
    h = b[i]
    g = as_tensor( h, (j, i) )
    f = g[k,l]
    a = f * I[k,l]

printit(a, True)

#a = IndexSum( Product(  Indexed(  v , 
