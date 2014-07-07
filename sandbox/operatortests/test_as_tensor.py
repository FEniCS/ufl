
from ufl import *
from ufl.tensors import ListTensor

n = FacetNormal(tetrahedron)

components = [ \
        (n, n),
        ((n, n), (n, n)),
        (((n, n), (n, n)), ((n, n), (n, n))),
        ((((n, n), (n, n)), ((n, n), (n, n))), (((n, n), (n, n)), ((n, n), (n, n)))),
    ]

for c in components:
    print() 
    print() 
    print(("c = ", c))
    t = ListTensor(*c)
    t2 = as_tensor(c)
    print(("equal =", t == t2))
    print(("shape =", t.shape()))
    print() 
    print((str(t)))
    print() 
    print((repr(t)))
    print() 

print() 

