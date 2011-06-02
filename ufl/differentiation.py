"Differential operators."

# Copyright (C) 2008-2011 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009.
#
# First added:  2008-03-14
# Last changed: 2009-12-08

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.common import subdict, mergedicts
from ufl.expr import Expr, Operator
from ufl.terminal import Terminal, Tuple
from ufl.constantvalue import ConstantValue, Zero, ScalarValue, Identity, is_true_ufl_scalar
from ufl.indexing import Index, FixedIndex, Indexed, as_multi_index, MultiIndex
from ufl.indexutils import unique_indices
from ufl.geometry import FacetNormal
from ufl.variable import Variable
from ufl.tensors import as_tensor, ComponentTensor, ListTensor
from ufl.argument import Argument
from ufl.coefficient import Coefficient, ConstantBase
from ufl.precedence import parstr

#--- Basic differentiation objects ---

def is_spatially_constant(expression):
    "Check if a terminal object is spatially constant, such that expression.dx(i) == 0."
    if isinstance(expression, (ConstantValue, ConstantBase)):
        return True
    elif isinstance(expression, FacetNormal): # and expression.cell().degree() == 1:
        return True
    elif isinstance(expression, Coefficient):
        e = expression.element()
        if e.family() == "Discontinuous Lagrange" and e.degree() == 0: # TODO: Only e.degree() == 0?
            return True
    elif isinstance(expression, ListTensor):
        return all(is_spatially_constant(e) for e in expression.operands())
    elif isinstance(expression, (Indexed, ComponentTensor)):
        return is_spatially_constant(expression.operands()[0])

    return False

class Derivative(Operator):
    "Base class for all derivative types."
    __slots__ = ()
    def __init__(self):
        Operator.__init__(self)

class CoefficientDerivative(Derivative):
    """Derivative of the integrand of a form w.r.t. the
    degrees of freedom in a discrete Coefficient."""
    __slots__ = ("_integrand", "_coefficients", "_arguments")

    def __new__(cls, integrand, coefficients, arguments):
        ufl_assert(is_true_ufl_scalar(integrand),
            "Expecting true UFL scalar expression.")
        ufl_assert(isinstance(coefficients, Tuple), #and all(isinstance(f, (Coefficient, Indexed)) for f in coefficients),
            "Expecting Tuple instance with Coefficients.")
        ufl_assert(isinstance(arguments, Tuple), #and all(isinstance(f, Argument) for f in arguments),
            "Expecting Tuple instance with Arguments.")
        if isinstance(integrand, Zero):
            return Zero()
        return Derivative.__new__(cls)

    def __init__(self, integrand, coefficients, arguments):
        Derivative.__init__(self)
        self._integrand = integrand
        self._coefficients = coefficients
        self._arguments = arguments

    def operands(self):
        return (self._integrand, self._coefficients, self._arguments)

    def shape(self):
        return ()

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return {}

    def __str__(self):
        # TODO: Short str notation
        return "CoefficientDerivative (w.r.t. coefficient %s and using argument %s) of \n%s"\
            % (self._coefficients, self._arguments, self._integrand)

    def __repr__(self):
        return "CoefficientDerivative(%r, %r, %r)"\
            % (self._integrand, self._coefficients, self._arguments)

def split_indices(expression, idx):
    idims = dict(expression.index_dimensions())
    if isinstance(idx, Index) and idims.get(idx) is None:
        cell = expression.cell()
        ufl_assert(cell is not None and not cell.is_undefined(),
            "Need to know the spatial dimension to "\
            "compute the shape of derivatives.")
        idims[idx] = cell.geometric_dimension()
    fi = unique_indices(expression.free_indices() + (idx,))
    return fi, idims

class SpatialDerivative(Derivative):
    "Partial derivative of an expression w.r.t. spatial directions given by indices."
    __slots__ = ("_expression", "_index", "_shape", "_free_indices", "_index_dimensions", "_repr")
    def __new__(cls, expression, index):
        # Return zero if expression is trivially constant
        if is_spatially_constant(expression):
            if isinstance(index, (tuple, MultiIndex)):
                index, = index
            fi, idims = split_indices(expression, index)
            return Zero(expression.shape(), fi, idims)
        return Derivative.__new__(cls)

    def __init__(self, expression, index):
        Derivative.__init__(self)
        self._expression = expression

        # Make a MultiIndex with knowledge of the dimensions
        cell = expression.cell()
        if cell is None or cell.is_undefined():
            error("Cannot compute derivatives of expressions with undefined cell.")
        else:
            sh = (cell.geometric_dimension(),)
        self._index = as_multi_index(index, sh)

        # Make sure we have a single valid index
        ufl_assert(len(self._index) == 1, "Expecting a single index.")
        fi, idims = split_indices(expression, self._index[0])

        # Store what we need
        self._free_indices = fi
        self._index_dimensions = idims
        self._shape = expression.shape()

        self._repr = "SpatialDerivative(%r, %r)" % (self._expression, self._index)

    def operands(self):
        return (self._expression, self._index)

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def shape(self):
        return self._shape

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."

        i = self._index[0]

        if isinstance(i, FixedIndex):
            i = int(i)

        pushed = False
        if isinstance(i, Index):
            i = index_values[i]
            pushed = True
            index_values.push(self._index[0], None)

        result = self._expression.evaluate(x, mapping, component, index_values, derivatives=derivatives + (i,))

        if pushed:
            index_values.pop()

        return result

    def __str__(self):
        if isinstance(self._expression, Terminal):
            return "d%s/dx_%s" % (self._expression, self._index)
        return "d/dx_%s %s" % (self._index, parstr(self._expression, self))

    def __repr__(self):
        return self._repr

class VariableDerivative(Derivative):
    __slots__ = ("_f", "_v", "_free_indices", "_index_dimensions", "_shape", "_repr")
    def __new__(cls, f, v):
        # Return zero if expression is trivially independent of Coefficient
        if isinstance(f, Terminal):# and not isinstance(f, Variable):
            free_indices = tuple(set(f.free_indices()) ^ set(v.free_indices()))
            index_dimensions = mergedicts([f.index_dimensions(), v.index_dimensions()])
            index_dimensions = subdict(index_dimensions, free_indices)
            return Zero(f.shape() + v.shape(), free_indices, index_dimensions)
        return Derivative.__new__(cls)

    def __init__(self, f, v):
        Derivative.__init__(self)
        ufl_assert(isinstance(f, Expr), "Expecting an Expr in VariableDerivative.")
        if isinstance(v, Indexed):
            ufl_assert(isinstance(v._expression, Variable), \
                "Expecting a Variable in VariableDerivative.")
            error("diff(f, v[i]) isn't handled properly in all code.") # TODO: Should we allow this? Can do diff(f, v)[..., i], which leads to additional work but does the same.
        else:
            ufl_assert(isinstance(v, Variable), \
                "Expecting a Variable in VariableDerivative.")
        self._f = f
        self._v = v
        fi = f.free_indices()
        vi = v.free_indices()
        fid = f.index_dimensions()
        vid = v.index_dimensions()
        #print "set(fi)", set(fi)
        #print "set(vi)", set(vi)
        #print "^", (set(fi) ^ set(vi))
        ufl_assert(not (set(fi) & set(vi)), \
            "Repeated indices not allowed in VariableDerivative.") # TODO: Allow diff(f[i], v[i]) = sum_i VariableDerivative(f[i], v[i])? Can implement direct expansion in diff as a first approximation.
        self._free_indices = tuple(fi + vi)
        self._index_dimensions = dict(fid)
        self._index_dimensions.update(vid)
        self._shape = f.shape() + v.shape()
        self._repr = "VariableDerivative(%r, %r)" % (self._f, self._v)

    def operands(self):
        return (self._f, self._v)

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def shape(self):
        return self._shape

    def __str__(self):
        if isinstance(self._f, Terminal):
            return "d%s/d[%s]" % (self._f, self._v)
        return "d/d[%s] %s" % (self._v, parstr(self._f, self))

    def __repr__(self):
        return self._repr

#--- Compound differentiation objects ---

class CompoundDerivative(Derivative):
    "Base class for all compound derivative types."
    __slots__ = ()
    def __init__(self):
        Derivative.__init__(self)

class Grad(CompoundDerivative):
    __slots__ = ("_f", "_dim", "_repr")

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        cell = f.cell()
        ufl_assert(cell is not None and not cell.is_undefined(),\
                   "Can't take gradient of expression with undefined cell...")
        dim = cell.geometric_dimension()
        if is_spatially_constant(f):
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero(f.shape() + (dim,), free_indices, index_dimensions)
        if dim == 1:
            return f.dx(0)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        cell = f.cell()
        ufl_assert(cell is not None and not cell.is_undefined(),\
                   "Can't take gradient of expression with undefined cell...")
        self._dim = cell.geometric_dimension()
        ufl_assert(not f.free_indices(),\
            "Taking gradient of an expression with free indices is not supported.")
        self._repr = "Grad(%r)" % self._f

    def operands(self):
        return (self._f, )

    def free_indices(self):
        return self._f.free_indices()

    def index_dimensions(self):
        return self._f.index_dimensions()

    def shape(self):
        return  self._f.shape() + (self._dim,)

    def __str__(self):
        return "grad(%s)" % self._f

    def __repr__(self):
        return self._repr

class Div(CompoundDerivative):
    __slots__ = ("_f", "_repr")

    def __new__(cls, f):
        ufl_assert(not f.free_indices(), \
            "TODO: Taking divergence of an expression with free indices,"\
            "should this be a valid expression? Please provide examples!")

        if f.rank() == 0:
            return f.dx(0)
        #ufl_assert(f.rank() >= 1, "Can't take the divergence of a scalar.")

        # Return zero if expression is trivially constant
        if is_spatially_constant(f):
            return Zero(f.shape()[:-1]) # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        self._repr = "Div(%r)" % self._f

    def operands(self):
        return (self._f, )

    def free_indices(self):
        return self._f.free_indices()

    def index_dimensions(self):
        return self._f.index_dimensions()

    def shape(self):
        return self._f.shape()[:-1]

    def __str__(self):
        return "div(%s)" % self._f

    def __repr__(self):
        return self._repr

class Curl(CompoundDerivative):
    __slots__ = ("_f", "_dim", "_repr")

    def __new__(cls, f):
        # Validate input
        sh = f.shape()
        ufl_assert(f.shape() in ((), (2,), (3,)), "Expecting a scalar, 2D vector or 3D vector.")
        ufl_assert(not f.free_indices(), \
            "TODO: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")

        # Return zero if expression is trivially constant
        if is_spatially_constant(f):
            cell = f.cell()
            ufl_assert(cell is not None and not cell.is_undefined(),\
                       "Can't take curl of expression with undefined cell...")
            sh = { (): (2,), (2,): (), (3,): (3,) }[sh]
            #free_indices = f.free_indices()
            #index_dimensions = subdict(f.index_dimensions(), free_indices)
            #return Zero((cell.geometric_dimension(),), free_indices, index_dimensions)
            return Zero(sh)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        cell = f.cell()
        ufl_assert(cell is not None and not cell.is_undefined(),\
                   "Can't take curl of expression with undefined cell...")
        sh = { (): (2,), (2,): (), (3,): (3,) }[f.shape()]
        self._f = f
        self._shape = sh
        self._repr = "Curl(%r)" % self._f

    def operands(self):
        return (self._f, )

    def free_indices(self):
        return self._f.free_indices()

    def index_dimensions(self):
        return self._f.index_dimensions()

    def shape(self):
        return self._shape

    def __str__(self):
        return "curl(%s)" % self._f

    def __repr__(self):
        return self._repr

