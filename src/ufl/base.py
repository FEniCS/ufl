#!/usr/bin/env python

"""
This module contains the UFLObject base class and all expression tree node types,
ie all classes and objects needed to cover the UFL language specification.
"""

import operator


### Utility functions:

def _isnumber(o):
    return isinstance(o, (int, float))

def product(l):
    return reduce(operator.__mul__, l)


### UFLObject base class:

class UFLObject(object):
    def __init__(self):
        pass
    
    # ... Strings:
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        """It is required to implement repr for all UFLObject subclasses."""
        raise NotImplementedError(self.__class__.__repr__)
    
    # ... Access to subtree nodes for expression traversal:
    
    def ops(self):
        """Returns a sequence with all subtree nodes in expression tree.
           All UFLObject subclasses are required to implement ops ."""
        raise NotImplementedError(self.__class__.ops)
    
    def fromops(self, *ops):
        """Build a new object of the same type as self but with different ops.
           All UFLObject subclasses are required to implement fromops IFF their
           constructor takes a different list of arguments than ops() returns."""
        return self.__class__(*ops)
    
    # ... Algebraic operators:
    
    def __mul__(self, o):
        if _isnumber(o): o = Number(o)
        if isinstance(o, Integral):
            return Form( [Integral(o.domain_type, o.domain_id, self)] )
        return Product(self, o)
    
    def __rmul__(self, o):
        if _isnumber(o): o = Number(o)
        return Product(o, self)

    def __add__(self, o):
        if _isnumber(o): o = Number(o)
        return Sum(self, o)

    def __radd__(self, o):
        if _isnumber(o): o = Number(o)
        return Sum(o, self)

    def __sub__(self, o):
        if _isnumber(o): o = Number(o)
        return self + (-o)

    def __rsub__(self, o):
        if _isnumber(o): o = Number(o)
        return o + (-self)
    
    def __div__(self, o):
        if _isnumber(o): o = Number(o)
        return Division(self, o)
    
    def __rdiv__(self, o):
        if _isnumber(o): o = Number(o)
        return Division(o, self)
    
    def __pow__(self, o):
        if _isnumber(o): o = Number(o)
        return Power(self, o)
    
    def __rpow__(self, o):
        if _isnumber(o): o = Number(o)
        return Power(o, self)

    def __mod__(self, o):
        if _isnumber(o): o = Number(o)
        return Mod(self, o)

    def __rmod__(self, o):
        if _isnumber(o): o = Number(o)
        return Mod(o, self)

    def __neg__(self):
        return -1*self

    def __abs__(self):
        return Abs(self)

    def transpose(self):
        return Transpose(self)

    T = property(transpose)

    # ... Sequence protocol for vectors, matrices, tensors.
    #     (Iteration over small objects with len and [] is fast enough,
    #      no need to bother with creating iterator objects)

    def __len__(self):
        raise NotImplementedError(self.__len__)

    #def __getitem__(self, key):
    #    # TODO: convert key OR key items to Number objects:  if _isnumber(o): o = Number(o)
    #    if isinstance(key, int):
    #        key = Number(key)
    #    elif isinstance(key, Index):
    #        pass
    #    elif isinstance(key, tuple):
    #        k = []
    #        for t in key:
    #            if isinstance(t, int):
    #                t = Number(t)
    #            k.append(t)
    #        key = tuple(k)
    #    else:
    #        raise TypeError()

    def __getitem__(self, key):
        return Indexed(self, key)

    # Should we even have this? Not on general expressions, maybe in a Tensor/Matrix/Vector class or something.
    #def __setitem__(self, key, value):
    #    if not isinstance(key, (int, Index, tuple)):
    #        raise TypeError()
    #    raise IndexError()

    # ... Searching for an UFLObject the subexpression tree:

    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        if isinstance(item, UFLObject):
            if item is self:
                return True
            item = repr(item)
        if repr(self) == item:
            return True
        return any((item in o) for o in self.ops())
    
    # ... Support for inserting an UFLObject in dicts and sets:
    
    def __hash__(self):
        return repr(self).__hash__()
    
    def __eq__(self, other): # alternative to above functions
        return repr(self) == repr(other)



### Integral and Form definition:

class Form(UFLObject):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    def __init__(self, integrals):    
        self.integrals = integrals
    
    def ops(self):
        return tuple(self.integrals)
    
    def _integrals(self, domain_type):
        itg = []
        for i in self.integrals:
            if i.domain_type == domain_type:
                itg.append(i)
        return itg
    
    def cell_integrals(self):
        return self._integrals("cell")
    
    def exterior_facet_integrals(self):
        return self._integrals("exterior_facet")
    
    def interior_facet_integrals(self):
        return self._integrals("interior_facet")

    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        return any(item in itg for itg in self.integrals)
    
    def __add__(self, other):
        return Form(self.integrals + other.integrals)
    
    def __str__(self):
        return "Form([%s])" % ", ".join(str(i) for i in self.integrals)
    
    def __repr__(self):
        return "Form([%s])" % ", ".join(repr(i) for i in self.integrals)


class Integral(UFLObject):
    """Description of an integral over a single domain."""
    def __init__(self, domain_type, domain_id, integrand=None):
        self.domain_type = domain_type
        self.domain_id   = domain_id
        self.integrand   = integrand
    
    def ops(self):
        return (self.integrand,)
    
    def fromops(self, *ops):
        return Integral(self.domain_type, self.domain_id, *ops) # TODO: Does this make sense? And in general for objects with non-ops constructor arguments?
    
    def __rmul__(self, other):
        assert self.integrand is None
        return Form( [Integral(self.domain_type, self.domain_id, other)] )

    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        return item in self.integrand
    
    def __str__(self):
        return "Integral(%s, %d, %s)" % (repr(self.domain_type), self.domain_id, self.integrand)
    
    def __repr__(self):
        return "Integral(%s, %s, %s)" % (repr(self.domain_type), repr(self.domain_id), repr(self.integrand))



class Jacobi:
    """Represents a linearized form, the Jacobi of a given nonlinear form wrt a given function."""
    def __init__(self, form, function):
        self.form = form
        self.function = function

    def __repr__(self):
        return "Jacobi(%s, %s)" % (repr(self.form), repr(self.function))



### Basic ... stuff

class Number(UFLObject):
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return "Number(%s)" % repr(self.value)

    def ops(self):
        return tuple()

    def fromops(self, *ops):
        return self

class Identity(UFLObject):
    def __init__(self):
        pass
    
    def __repr__(self):
        return "Identity()"

    def ops(self):
        return tuple()

    def fromops(self, *ops):
        return self

class Transpose(UFLObject):
    def __init__(self, f):
        self.f = f
    
    def ops(self):
        return (self.f,)
    
    def __repr__(self):
        return "Transpose(%s)" % repr(self.f)

class Symbol(UFLObject): # TODO: needed for diff?
    def __init__(self, name):
        self.name = name
    
    def ops(self):
        return tuple()
    
    def fromops(self, *ops):
        return self
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return "Symbol(%s)" % repr(self.name)

class Variable(UFLObject):
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression
    
    def ops(self):
        return (self.expression,)
    
    def fromops(self, *ops):
        return Variable(self.name, *ops)
    
    def __repr__(self):
        return "Variable(%s, %s)" % (repr(self.name), repr(self.expression))


### Algebraic operators

class Product(UFLObject):
    def __init__(self, *ops):
        self._ops = tuple(ops)
    
    def ops(self):
        return self._ops
    
    def __repr__(self):
        return "(%s)" % " * ".join(repr(o) for o in self._ops)
    

class Sum(UFLObject):
    def __init__(self, *ops):
        self._ops = tuple(ops)
    
    def ops(self):
        return self._ops
    
    def __repr__(self):
        return "(%s)" % " + ".join(repr(o) for o in self._ops)
    

class Sub(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s - %s)" % (repr(self.a), repr(self.b))
    

class Division(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s / %s)" % (repr(self.a), repr(self.b))
    

class Power(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s ** %s)" % (repr(self.a), repr(self.b))
    

class Mod(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s %% %s)" % (repr(self.a), repr(self.b))
    

class Abs(UFLObject):
    def __init__(self, a):
        self.a = a
    
    def ops(self):
        return (self.a, )
    
    def __repr__(self):
        return "Abs(%s)" % repr(self.a)
    


### Indexing

class Index(UFLObject):
    def __init__(self, name, dim=None): # TODO: do we need dim? 
        self.name = name
    
    def __repr__(self):
        return "Index(%s)" % repr(self.name)
    
    def ops(self):
        return tuple()
    
    def fromops(self, *ops):
        return self

class Indexed(UFLObject):
    def __init__(self, expression, indices):
        self.expression = expression
        self.indices = indices
    
    def __repr__(self):
        return "Indexed(%s, %s)" % (repr(self.expression), repr(self.indices))
    
    def ops(self): # FIXME: should this return the indices at all?
        raise RuntimeError("Not sure how this should be implemented.")
        if isinstance(self.indices, tuple):
            return tuple([self.expression] + list(self.indices))
        return tuple(self.expression, self.indices) # FIXME: should this return the indices at all?
    
    def fromops(self, *ops):
        raise RuntimeError("Not sure how this should be implemented.")
        return self



### How to handle tensor, subcomponents, indexing, Einstein summation? TODO: Need experiences from FFC!


if __name__ == "__main__":
    print "No tests here."

