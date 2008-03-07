#!/usr/bin/env python

"""
Utility objects for pretty syntax in user code.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 8th 2008"

from base  import *
from geometry import *
import math

I = Identity()

# constants

e  = math.e
pi = math.pi

# default indices

i, j, k, l = [Index(name) for name in "ijkl"]
p, q, r, s = [Index(name) for name in "pqrs"]

# default integrals

dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9 = [Integral("cell", domain_id)           for domain_id in range(10)]
ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9 = [Integral("exterior_facet", domain_id) for domain_id in range(10)]
dS0, dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8, dS9 = [Integral("interior_facet", domain_id) for domain_id in range(10)]
dx, ds, dS = dx0, ds0, dS0

# geometric entities

n = FacetNormal()
h = CellRadius()

