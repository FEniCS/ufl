
import operator
from ufl.algorithms import Transformer, expand_compounds
from physical_dimensions import PhysicalDimension

class PhysicalDimensionChecker(Transformer):
    def __init__(self, xdim, function_dims):
        super(PhysicalDimensionChecker, self).__init__()
        self.xdim = xdim
        self.function_dims = function_dims
        self.warnings = []
        self.nodim = PhysicalDimension({})

    def warn(self, o, ops):
        opdims = ", ".join(map(str, ops))
        w = "Operator %s has operands with dimensions %s." % (o._uflclass, opdims)
        self.warnings.append(w)

    def terminal(self, o):
        return self.nodim

    def coefficient(self, o):
        return self.function_dims.get(o, self.nodim)

    def argument(self, o):
        return self.function_dims.get(o, self.nodim)

    def spatial_coordinate(self, o, *ops):
        return self.xdim

    def expr(self, o, *ops):
        if any(ops):
            self.warn(o, ops)
        return self.nodim

    def indexed(self, o, *ops):
        return ops[0]

    # TODO: This and other types:
    #def list_tensor(self, o, *ops):
    #    o0 = ops[0]
    #    for o1 in ops:
    #        if o1 != o0:
    #            self.warn(o, ops)
    #            return self.nodim
    #    return o0

    def sum(self, o, *ops):
        o0 = ops[0]
        for o1 in ops:
            if o1 != o0:
                self.warn(o, ops)
                return self.nodim
        return o0

    def product(self, o, *ops):
        if any(ops):
            return reduce(operator.__mul__, ops)
        else:
            return self.nodim

    def division(self, o, *ops):
        if any(ops):
            return ops[0]/ops[1]
        else:
            return self.nodim

    def sqrt(self, o, op):
        return op**0.5

    def power(self, o, *ops):
        if ops[1]:
            self.warn(o, ops)
        elif ops[0]:
            try:
                m = o.operands()[1]
                f = float(m)
                i = int(m)
            except:
                self.warn(o, ops)
                return self.nodim
            if f == i:
                return ops[0]**i
            else:
                return ops[0]**f
        else:
            return self.nodim

    def spatial_derivative(self, o, *ops):
        return ops[0] / self.xdim
    grad = spatial_derivative
    div = spatial_derivative
    curl = spatial_derivative
    rot = spatial_derivative

def check_physical_dimensions(expr, xdim, function_dims):
    expr2 = expand_compounds(expr)
    alg = PhysicalDimensionChecker(xdim, function_dims)
    dim = alg.visit(expr)
    return dim, alg.warnings

from ufl import *
def test_check_physical_dimensions():
    V = FiniteElement("CG", triangle, 1)
    f = Coefficient(V)
    g = Coefficient(V)

    m = PhysicalDimension("m")
    s = PhysicalDimension("s")

    d, w = check_physical_dimensions(f, m, {})
    assert not w
    assert str(d) == ""

    d, w = check_physical_dimensions(f, m, {f: s})
    assert not w
    assert str(d) == "s"

    d, w = check_physical_dimensions(f*2, m, {f: s})
    assert not w
    assert str(d) == "s"

    d, w = check_physical_dimensions(f**2, m, {f: s})
    if w: print(w)
    assert not w
    assert str(d) == "s^2"

    d, w = check_physical_dimensions(g*f**2, m, {f: s, g: m**2/s})
    if w: print(w)
    assert not w
    assert str(d) == "m^2 s"

    # Test that sum gives no warning
    d, w = check_physical_dimensions(sqrt(g**2*f**2)*f + g*f**2, m, {f: s, g: m**2/s})
    if w: print(w)
    assert not w
    assert str(d) == "m^2 s"

    # Test that sum gives warning
    d, w = check_physical_dimensions(f + g, m, {f: m/s, g: m**2/s})
    #if w: print w
    assert w
    assert str(d) == ""

    # Test derivatives and x which uses xdim
    d, w = check_physical_dimensions(f.dx(0) + grad(f)[0] + div(grad(f))*triangle.x[0], m, {f: m/s})
    if w: print(w)
    assert not w
    assert str(d) == "s^-1"

if __name__ == "__main__":
    test_check_physical_dimensions()

