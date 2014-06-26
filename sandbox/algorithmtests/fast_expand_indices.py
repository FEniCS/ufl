
from ufl import *
from ufl.common import *
from ufl.classes import *
from ufl.algorithms import *

from ufl.testobjects import *
if __name__ == "__main__":
    v = BasisFunction(element)
    u = BasisFunction(element)
    print((expand_indices2(v)))
    print((expand_indices2(u*v)))
    print((expand_indices2(vv[0])))
    print((expand_indices2(u*vv[0])))
    print((expand_indices2(vu[0]*vv[0])))
    print((expand_indices2(u*v)))
    print((expand_indices2(vu[i]*vv[i])))
    print((expand_indices2(vu[i]*as_vector([2,3])[i])))
    print((expand_indices2(as_vector(vu[i], i)[j]*vv[j])))
    print((expand_indices2(expand_compounds( dot(dot(vu, as_tensor(vu[i]*vv[j], (i,j))), vv) ) )))

    print((expand_indices(expand_compounds( dot(dot(vu, as_tensor(vu[i]*vv[j], (i,j))), vv) ) )))

