

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
    
    # ... Access to subtree nodes for expression traversing:
    
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

    # ... Scalar casting operators, probably won't be needed for most ufl objects.
    #     Needs to be implemented in all subclasses to work, and must
    #     fail if a single Symbol, Function or FiniteElement is hit.

    def __int__(self):
        raise NotImplementedError(self.__int__)
    
    def __long__(self):
        raise NotImplementedError(self.__long__)
    
    def __float__(self):
        raise NotImplementedError(self.__float__)

    # ... Sequence protocol for vectors, matrices, tensors.
    #     (Iteration over small objects with len and [] is fast enough,
    #      no need to bother with creating iterator objects)

    def __len__(self):
        raise NotImplementedError(self.__len__)

    def __getitem__(self, key):
        # TODO: convert key OR key items to Number objects:  if _isnumber(o): o = Number(o)
        if isinstance(key, int):
            key = Number(key)
        elif isinstance(key, Index):
            pass
        elif isinstance(key, tuple):
            k = []
            for t in key:
                if isinstance(t, int):
                    t = Number(t)
                k.append(t)
            key = tuple(k)
        else:
            raise TypeError()
        return Indexed(self, key)

    def __setitem__(self, key, value):
        if not isinstance(key, (int, Index, tuple)):
            raise TypeError()
        raise IndexError()

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




### Finite element definitions

# TODO: should rather have an inheritance hierarchy here, all *Element classes interiting a single base class:
# if isinstance(a, UFLFiniteElement)

class UFLFiniteElement:
    def __init__(self, polygon):
        self.polygon = polygon

    def __add__(self, other):
        return MixedElement(self, other)

class FiniteElement(UFLFiniteElement):
    def __init__(self, family, polygon, order):
        UFLFiniteElement.__init__(self, polygon)
        self.family  = family
        self.order   = order
    
    def __repr__(self):
        return "FiniteElement(%s, %s, %d)" % (repr(self.family), repr(self.polygon), self.order)

class VectorElement(UFLFiniteElement):
    def __init__(self, family, polygon, order, size=None):
        UFLFiniteElement.__init__(self, polygon)
        self.family  = family
        self.order   = order
        self.size    = size
    
    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.size))

class TensorElement(UFLFiniteElement):
    def __init__(self, family, polygon, order, shape=None):
        UFLFiniteElement.__init__(self, polygon)
        self.family  = family
        self.order   = order
        self.shape   = shape
    
    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.shape))

class MixedElement(UFLFiniteElement):
    def __init__(self, *elements):
        UFLFiniteElement.__init__(self, elements[0].polygon)
        self.elements = elements

class QuadratureElement(UFLFiniteElement):
    def __init__(self, polygon, domain_type="cell"):
        UFLFiniteElement.__init__(self, polygon)
        self.domain_type = domain_type # TODO: define this better



### Variants of functions derived from finite elements

class BasisFunction(UFLObject):
    def __init__(self, element):
        self.element = element
    
    def __repr__(self):
        return "BasisFunction(%s)" % repr(self.element)

    def ops(self):
        return tuple()

    def fromops(self, *ops):
        return self


def BasisFunctions(element):
    if isinstance(element, MixedElement):
        return tuple(BasisFunction(fe) for fe in element.elements)
    raise ValueError("Expecting MixedElement instance.")

class TestFunction(BasisFunction):
    def __init__(self, element):
        self.element = element

    def __repr__(self):
        return "TestFunction(%s)" % repr(self.element)

def TestFunctions(element):
    if isinstance(element, MixedElement):
        return tuple(TestFunction(fe) for fe in element.elements)
    raise ValueError("Expecting MixedElement instance.")

class TrialFunction(BasisFunction):
    def __init__(self, element):
        self.element = element

    def __repr__(self):
        return "TrialFunction(%s)" % repr(self.element)

def TrialFunctions(element):
    if isinstance(element, MixedElement):
        return tuple(TrialFunction(fe) for fe in element.elements)
    raise ValueError("Expecting MixedElement instance.")

class UFLCoefficient(UFLObject):
    _count = 0
    def __init__(self, element, name):
        self.count = UFLCoefficient._count
        self.name = name
        self.element = element
        UFLCoefficient._count += 1

    def ops(self):
        return tuple()

    def fromops(self, *ops):
        return self

class Function(UFLCoefficient):
    def __init__(self, element, name):
        UFLCoefficient.__init__(self, element, name)
    
    def __repr__(self):
        return "Function(%s, %s)" % (repr(self.element), repr(self.name))

class Constant(UFLCoefficient):
    def __init__(self, polygon, name):
        UFLCoefficient.__init__(self, FiniteElement("DiscontinuousLagrange", polygon, 0), name)
        self.polygon = polygon
    
    def __repr__(self):
        return "Constant(%s, %s)" % (repr(self.element.polygon), repr(self.name))


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
    
    def __int__(self):
        return product(int(i) for i in self._ops)
    
    def __long__(self):
        return product(long(i) for i in self._ops)
    
    def __float__(self):
        return product(float(i) for i in self._ops)

class Sum(UFLObject):
    def __init__(self, *ops):
        self._ops = tuple(ops)
    
    def ops(self):
        return self._ops
    
    def __repr__(self):
        return "(%s)" % " + ".join(repr(o) for o in self._ops)
    
    def __int__(self):
        return sum(int(i) for i in self._ops)
    
    def __long__(self):
        return sum(long(i) for i in self._ops)
    
    def __float__(self):
        return sum(float(i) for i in self._ops)

class Sub(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s - %s)" % (repr(self.a), repr(self.b))
    
    def __int__(self):
        return int(self.a) - int(self.b)
    
    def __long__(self):
        return long(self.a) - long(self.b)
    
    def __float__(self):
        return float(self.a) - float(self.b)

class Division(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s / %s)" % (repr(self.a), repr(self.b))
    
    def __int__(self):
        return int(self.a) / int(self.b)
    
    def __long__(self):
        return long(self.a) / long(self.b)
    
    def __float__(self):
        return float(self.a) / float(self.b)

class Power(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s ** %s)" % (repr(self.a), repr(self.b))
    
    def __int__(self):
        return int(self.a) ** int(self.b)
    
    def __long__(self):
        return long(self.a) ** long(self.b)
    
    def __float__(self):
        return float(self.a) ** float(self.b)

class Mod(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s %% %s)" % (repr(self.a), repr(self.b))
    
    def __int__(self):
        return int( float(self.a) % float(self.b) )
    
    def __long__(self):
        return long( float(self.a) % float(self.b) )
    
    def __float__(self):
        return float(self.a) % float(self.b)

class Abs(UFLObject):
    def __init__(self, a):
        self.a = a
    
    def ops(self):
        return (self.a, )
    
    def __repr__(self):
        return "Abs(%s)" % repr(self.a)
    
    def __int__(self):
        return int( abs(self.a) )
    
    def __long__(self):
        return long( abs(self.a) )
    
    def __float__(self):
        return float( abs(self.a) )



### Algebraic operations on tensors:
# TODO: define dot, inner, and contract clearly:
# Scalars:
#   dot(a,b)      = a*b
#   inner(a,b)    = a*b
#   contract(a,b) = a*b
# Vectors:
#   dot(u,v)      = u_i v_i
#   inner(u,v)    = u_i v_i
#   contract(u,v) = u_i v_i
# Matrices:
#   dot(A,B)      = A_{ik} B_{kj}
#   inner(A,B)    = A_{ij} B_{ij}
#   contract(A,B) = A_{ij} B_{ij}
# Combined:
#   dot(A,u)      = A_{ik} u_k
#   inner(A,u)    = A_{ik} u_k
#   contract(A,u) = A_{ik} u_k
#   dot(u,B)      = u_k B_{ki}
#   inner(u,B)    = u_k B_{ki}
#   contract(u,B) = u_k B_{ki}
#
# Maybe in general (contract is clearly a duplicate of inner above):
#   dot(x,y)   = contract(x, -1, y, 0)        # (last x dim) vs (first y dim)
#   inner(x,y) = contract(A, (0,1), B, (0,1)) # (all A dims) vs (all B dims)
#   contract(x,(xi),y,(yi)) = \sum_i x_{xi} y_{yi} # something like this, xi and yi are multiindices, TODO: need to design index stuff properly
#
#   dot(x,y): last index of x has same dimension as first index of y
#   inner(x,y): shape of x equals the shape of y
#   contract(x, xi, y, yi): len(xi) == len(yi), dimensions of indices in xi and yi match, dim(x) >= max(xi), dim(y) >= max(yi)

# objects representing the operations:

class Outer(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Outer(%s, %s)" % (repr(self.a), repr(self.b))
    
class Inner(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Inner(%s, %s)" % (repr(self.a), repr(self.b))

class Contract(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Contract(%s, %s)" % (repr(self.a), repr(self.b))

class Dot(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Dot(%s, %s)" % (self.a, self.b)

class Cross(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def ops(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Cross(%s, %s)" % (repr(self.a), repr(self.b))

class Trace(UFLObject):
    def __init__(self, A):
        self.A = A
    
    def ops(self):
        return (self.A, )
    
    def __repr__(self):
        return "Trace(%s)" % repr(self.A)

class Determinant(UFLObject):
    def __init__(self, A):
        self.A = A
    
    def ops(self):
        return (self.A, )
    
    def __repr__(self):
        return "Determinant(%s)" % repr(self.A)

class Inverse(UFLObject):
    def __init__(self, A):
        self.A = A
    
    def ops(self):
        return (self.A, )
    
    def __repr__(self):
        return "Inverse(%s)" % repr(self.A)

# functions exposed to the user:

def outer(a, b):
    return Outer(a, b)

def inner(a, b):
    return Inner(a, b)

def contract(a, b):
    return Contract(a, b)

def dot(a, b):
    return Dot(a, b)

def cross(a, b):
    return Cross(a, b)

def det(f):
    return Determinant(f)

def determinant(f):
    return Determinant(f)

def inverse(f):
    return Inverse(f)

def tr(f):
    return Trace(f)

def trace(f):
    return Trace(f)

def dev(A): # TODO:
    return Deviatoric(A)

#def cofactor(A): # TODO:
#    return det(A)*inverse(A)


### Differential operators

# objects representing the differential operations:

class DiffOperator(UFLObject):
    def __init__(self, x):
        if isinstance(x, int):
            x = p[x]
        elif not isinstance(x, Symbol):
            raise ValueError("x must be a Symbol")
        self.x = x
    
    def __mul__(self, o):
        return diff(o, self.x)

    def __repr__(self):
        return "DiffOperator(%s)" % repr(self.x)

class Diff(UFLObject):
    def __init__(self, f, x):
        self.f = f
        self.x = x
    
    def ops(self):
        return (self.f, self.x)
    
    def __repr__(self):
        return "Diff(%s, %s)" % (repr(self.f), repr(self.x))

class Grad(UFLObject):
    def __init__(self, f):
        self.f = f
    
    def ops(self):
        return (self.f, )
    
    def __repr__(self):
        return "Grad(%s)" % repr(self.f)

class Div(UFLObject):
    def __init__(self, f):
        self.f = f
    
    def ops(self):
        return (self.f, )
    
    def __repr__(self):
        return "Div(%s)" % repr(self.f)

class Curl(UFLObject):
    def __init__(self, f):
        self.f = f
    
    def ops(self):
        return (self.f, )
    
    def __repr__(self):
        return "Curl(%s)" % repr(self.f)

class Wedge(UFLObject):
    def __init__(self, f):
        self.f = f
    
    def ops(self):
        return (self.f, )
    
    def __repr__(self):
        return "Wedge(%s)" % repr(self.f)


# functions exposed to the user:

def diff(f, x):
    return Diff(f, x)

def Dx(f, i):
    return Diff(f, x[i])

def Dt(f):
    return Diff(f, t)

def grad(f):
    return Grad(f)

def div(f):
    return Div(f)

def curl(f):
    return Curl(f)

def wedge(f):
    return Wedge(f)



### Functions

class UFLFunction(UFLObject):
    def __init__(self, name, argument):
        self.name     = name
        self.argument = argument
    
    def ops(self):
        return (self.argument,)
    
    def __repr__(self):
        return "%s(%s)" % (self.name, repr(self.argument))

# functions exposed to the user:

def sqrt(f):
    return UFLFunction("sqrt", f)

def exp(f):
    return UFLFunction("exp", f)

def ln(f):
    return UFLFunction("ln", f)

def cos(f):
    return UFLFunction("cos", f)

def sin(f):
    return UFLFunction("sin", f)

def tan(f):
    return UFLFunction("tan", f)



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
    
    def ops(self):
        return tuple(self.expression, self.indices)



### Quantities computed from cell geometry

class GeometricQuantity(UFLObject):
    def __init__(self):
        pass

class FacetNormal(GeometricQuantity):
    def __init__(self):
        pass
    
    def __repr__(self):
        return "FacetNormal()"

class CellRadius(GeometricQuantity):
    def __init__(self):
        pass
    
    def __repr__(self):
        return "CellRadius()"



# Utility objects for pretty syntax in user code

I = Identity()

n = FacetNormal()
h = CellRadius()

# default indices
i, j, k, l, m, n, o, p, q, r, s = [Index(name) for name in "ijklmnopqrs"]

# default integrals
dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9 = [Integral("cell", domain_id)           for domain_id in range(10)]
ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9 = [Integral("exterior_facet", domain_id) for domain_id in range(10)]
dS0, dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8, dS9 = [Integral("interior_facet", domain_id) for domain_id in range(10)]
dx, ds, dS = dx0, ds0, dS0



### How to handle tensor, subcomponents, indexing, Einstein summation? TODO: Need experiences from FFC!


if __name__ == "__main__":
    print "No tests here."

