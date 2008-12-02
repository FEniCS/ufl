"Types for quantities computed from cell geometry."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-12-02"

from ufl.base import Terminal
from ufl.common import domain2dim

class SpatialCoordinate(Terminal):
    __slots__ = ("_domain",)
    def __init__(self, domain):
        self._domain = domain

    def shape(self):
        return (domain2dim[self._domain],)
    
    def domain(self):
        return self._domain

    def __str__(self):
        return "x"
    
    def __repr__(self):
        return "SpatialCoordinate(%r)" % self._domain

    def __eq__(self, other):
        return isinstance(other, SpatialCoordinate) and other._domain == self._domain

class FacetNormal(Terminal):
    def __init__(self, domain):
        Terminal.__init__(self)
        self._domain = domain
    
    def shape(self):
        return (domain2dim[self._domain],)
    
    def domain(self):
        return self._domain
    
    def __str__(self):
        return "n"
    
    def __repr__(self):
        return "FacetNormal(%r)" % self._domain

    def __eq__(self, other):
        return isinstance(other, FacetNormal) and other._domain == self._domain
