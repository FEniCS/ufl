"""Utility objects for pretty syntax in user code."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"

from base import *
from integral import *
from geometry import *
from indexing import Index
import math

# TODO: This is only the matrix identity, support higher order Id too?
class Identity(Terminal):
    __slots__ = ()
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return 2
    
    def __str__(self):
        return "I"
    
    def __repr__(self):
        return "Identity()"

I = Identity()

# Constants
e  = Number(math.e)
pi = Number(math.pi)

# Default indices
i, j, k, l = [Index() for _i in range(4)]
p, q, r, s = [Index() for _i in range(4)]

# Default integrals
dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9 = [Integral("cell", _domain_id)           for _domain_id in range(10)]
ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9 = [Integral("exterior_facet", _domain_id) for _domain_id in range(10)]
dS0, dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8, dS9 = [Integral("interior_facet", _domain_id) for _domain_id in range(10)]
dx, ds, dS = dx0, ds0, dS0

# Geometric entities
n = FacetNormal()
h = MeshSize()
