
from ufl import *
from ufl.classes import *
from ufl.algorithms import *

def test1():
    element = VectorElement("Lagrange", triangle, 1)
    f = Function(element)
    
    a = f[j]*as_tensor(f[i], i)[j]
    print str(a)
    print tree_format(a)

    a = expand_derivatives(a)
    print str(a)
    print tree_format(a)

    a = renumber_indices(a)
    print str(a)
    print tree_format(a)

    a = expand_indices(a)
    print str(a)
    print tree_format(a)

def test2():
    element = TensorElement("Lagrange", triangle, 1)
    f = Function(element)
    
    a = as_tensor(f[k,l], (l,k))[j,i] * as_tensor(f[k,l], (l,k))[i,j]
    print str(a)
    print tree_format(a)

    a = expand_derivatives(a)
    print str(a)
    print tree_format(a)

    a = renumber_indices(a)
    print str(a)
    print tree_format(a)

    a = expand_indices(a)
    print str(a)
    print tree_format(a)

def test():
    element = TensorElement("Lagrange", triangle, 1)
    A = Function(element)
    
    a = as_tensor(A[i,j], j)[i] # => sum_i
    
    print str(a)
    print tree_format(a)

    a = expand_derivatives(a)
    print str(a)
    print tree_format(a)

    a = renumber_indices(a)
    print str(a)
    print tree_format(a)

    a = expand_indices(a)
    print str(a)
    print tree_format(a)

if __name__ == "__main__":
    test()

