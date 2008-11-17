
from ufl import *
from ufl.tensors import ListTensor

components = [ \
        (n, n),
        ((n, n), (n, n)),
        (((n, n), (n, n)), ((n, n), (n, n))),
        ((((n, n), (n, n)), ((n, n), (n, n))), (((n, n), (n, n)), ((n, n), (n, n)))),
    ]

for c in components:
    print 
    print "c = ", c
    t = ListTensor(*c)
    print str(t)
    print repr(t)

print 

