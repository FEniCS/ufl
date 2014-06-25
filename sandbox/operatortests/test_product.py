
from ufl.testobjects import *
from ufl.algorithms import expand_compounds

def _show(x):
    print(x.shape())
    print(x.free_indices())
    print(str(x))
    print(repr(x))

def show(x):
    print("--------------")
    _show(x)
    y = expand_compounds(x)
    _show(y)

show(u*v)
show(u*vv)
show(vu*v)
show(u*tv)
show(tu*v)
show(tu*vv)
show(tu*tv)

