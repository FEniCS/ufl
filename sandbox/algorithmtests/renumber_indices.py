
from ufl import *
from ufl.classes import *
from ufl.algorithms import *

velement = VectorElement("Lagrange", triangle, 1)
v = Function(velement)
telement = TensorElement("Lagrange", triangle, 1)
A = Function(telement)
I = Identity(2)

def printit(a, tree=False):
    print(("-"*80))
    print() 
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

if __name__ == "__main__":
    tests = [ \
        # constant
        v[0],
        A[0,0],
        # constant to index mapping
        as_tensor(v[i], i)[0] ,
        as_tensor(A[i,j], (j,i))[1,2] ,
        # index to index mapping
        as_tensor(v[i], i)[j] * v[j] ,
        as_tensor(A[k,l], (l,k))[j,i] * as_tensor(A[k,l], (l,k))[i,j] ,
        # hidden implicit sum!
        as_tensor(A[i,j], j)[i] , # => sum_i
        # partial mapping
        as_tensor(A[i,0], i)[1] ,
        as_tensor(A[1,i], i)[2] ,
        # double layers
        as_tensor(A[j,:][i], (j, i))[k,l]*I[k,l],
        # double index meaning
        (v[i]*v[i]) * (v[i]*v[i]),
        (v[i]*v[i]) * (2*v[i]*v[i]),
        ]
    
    for a in tests:
        printit(a, True)
    
    #test2()

# Failure on this one:
#mixed = MixedElement(*[VectorElement('Lagrange', Cell('triangle', 1), 2, 2),
#                       FiniteElement('Lagrange', Cell('triangle', 1), 1)],
#                     **{'value_shape': (3,) })
#Indexed(
#    ComponentTensor(
#        Indexed(
#            ListTensor(Indexed(SpatialDerivative(BasisFunction(mixed, -2), MultiIndex((Index(12),))), MultiIndex((FixedIndex(0),))),
#                       Indexed(SpatialDerivative(BasisFunction(mixed, -2), MultiIndex((Index(12),))), MultiIndex((FixedIndex(1),)))),
#            MultiIndex((Index(13),))
#        ),
#        MultiIndex((Index(12), Index(13)))
#    ),
#    MultiIndex((Index(14), Index(15)))
#)

