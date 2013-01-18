"Differential operators."

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Last changed: 2011-11-10

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.common import subdict, mergedicts, EmptyDict
from ufl.expr import Expr
from ufl.terminal import Terminal, Data
from ufl.operatorbase import Operator, Tuple
from ufl.constantvalue import Zero
from ufl.indexing import Index, FixedIndex, MultiIndex, as_multi_index
from ufl.indexed import Indexed
from ufl.indexutils import unique_indices
from ufl.variable import Variable
from ufl.precedence import parstr

#--- Basic differentiation objects ---

class Derivative(Operator):
    "Base class for all derivative types."
    __slots__ = ()
    def __init__(self):
        Operator.__init__(self)

class CoefficientDerivative(Derivative):
    """Derivative of the integrand of a form w.r.t. the
    degrees of freedom in a discrete Coefficient."""
    __slots__ = ("_integrand", "_coefficients", "_arguments",
                 "_coefficient_derivatives")

    def __new__(cls, integrand, coefficients, arguments, coefficient_derivatives):
        ufl_assert(isinstance(coefficients, Tuple),
            "Expecting Tuple instance with Coefficients.")
        ufl_assert(isinstance(arguments, Tuple),
            "Expecting Tuple instance with Arguments.")
        ufl_assert(isinstance(coefficient_derivatives, (dict, Data)),
                   "Expecting a dict for coefficient derivatives.")

        if isinstance(integrand, Zero):
            return integrand
        return Derivative.__new__(cls)

    def __init__(self, integrand, coefficients, arguments, coefficient_derivatives):
        Derivative.__init__(self)
        self._integrand = integrand
        self._coefficients = coefficients
        self._arguments = arguments
        if isinstance(coefficient_derivatives, Data):
            self._coefficient_derivatives = coefficient_derivatives
        else:
            self._coefficient_derivatives = Data(coefficient_derivatives)

    def operands(self):
        return (self._integrand, self._coefficients, self._arguments, self._coefficient_derivatives)

    def shape(self):
        return self._integrand.shape()

    def free_indices(self):
        return self._integrand.free_indices()

    def index_dimensions(self):
        return self._integrand.index_dimensions()

    def __str__(self):
        return "d/dfj { %s }, with fh=%s, dfh/dfj = %s, and coefficient derivatives %s"\
            % (self._integrand, self._coefficients, self._arguments, self._coefficient_derivatives)

    def __repr__(self):
        return "CoefficientDerivative(%r, %r, %r, %r)"\
            % (self._integrand, self._coefficients, self._arguments, self._coefficient_derivatives)

def split_indices(expression, idx):
    idims = dict(expression.index_dimensions())
    if isinstance(idx, Index) and idims.get(idx) is None:
        idims[idx] = expression.geometric_dimension()
    fi = unique_indices(expression.free_indices() + (idx,))
    return fi, idims

class VariableDerivative(Derivative):
    __slots__ = ("_f", "_v", "_free_indices", "_index_dimensions", "_shape",)
    def __new__(cls, f, v):
        # Return zero if expression is trivially independent of variable
        if isinstance(f, Terminal):
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
        return "VariableDerivative(%r, %r)" % (self._f, self._v)

#--- Compound differentiation objects ---

class CompoundDerivative(Derivative):
    "Base class for all compound derivative types."
    __slots__ = ()
    def __init__(self):
        Derivative.__init__(self)

class Grad(CompoundDerivative):
    __slots__ = ("_f", "_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        dim = f.geometric_dimension()
        if f.is_cellwise_constant():
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero(f.shape() + (dim,), free_indices, index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        self._dim = f.geometric_dimension()

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        if op.is_cellwise_constant():
            ufl_assert(op.shape() == self._f.shape(),
                       "Operand shape mismatch in Grad reconstruct.")
            ufl_assert(self._f.free_indices() == op.free_indices(),
                       "Free index mismatch in Grad reconstruct.")
            return Zero(self.shape(), self.free_indices(),
                        self.index_dimensions())
        return self.__class__._uflclass(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        r = len(component)
        component, i = component[:-1], component[-1]
        derivatives = derivatives + (i,)
        result = self._f.evaluate(x, mapping, component, index_values,
                                  derivatives=derivatives)
        return result

    def operands(self):
        return (self._f,)

    def free_indices(self):
        return self._f.free_indices()

    def index_dimensions(self):
        return self._f.index_dimensions()

    def shape(self):
        return self._f.shape() + (self._dim,)

    def __str__(self):
        return "grad(%s)" % self._f

    def __repr__(self):
        return "Grad(%r)" % self._f

class Div(CompoundDerivative):
    __slots__ = ("_f",)

    def __new__(cls, f):
        ufl_assert(not f.free_indices(), \
            "TODO: Taking divergence of an expression with free indices,"\
            "should this be a valid expression? Please provide examples!")

        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            return Zero(f.shape()[:-1]) # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f

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
        return "Div(%r)" % self._f

class NablaGrad(CompoundDerivative):
    __slots__ = ("_f", "_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        dim = f.geometric_dimension()
        if f.is_cellwise_constant():
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero((dim,) + f.shape(), free_indices, index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        self._dim = f.geometric_dimension()

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        if op.is_cellwise_constant():
            ufl_assert(op.shape() == self._f.shape(),
                       "Operand shape mismatch in NablaGrad reconstruct.")
            ufl_assert(self._f.free_indices() == op.free_indices(),
                       "Free index mismatch in NablaGrad reconstruct.")
            return Zero(self.shape(), self.free_indices(),
                        self.index_dimensions())
        return self.__class__._uflclass(op)

    def operands(self):
        return (self._f, )

    def free_indices(self):
        return self._f.free_indices()

    def index_dimensions(self):
        return self._f.index_dimensions()

    def shape(self):
        return (self._dim,) + self._f.shape()

    def __str__(self):
        return "nabla_grad(%s)" % self._f

    def __repr__(self):
        return "NablaGrad(%r)" % self._f

class NablaDiv(CompoundDerivative):
    __slots__ = ("_f",)

    def __new__(cls, f):
        ufl_assert(not f.free_indices(), \
            "TODO: Taking divergence of an expression with free indices,"\
            "should this be a valid expression? Please provide examples!")

        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            return Zero(f.shape()[1:]) # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f

    def operands(self):
        return (self._f, )

    def free_indices(self):
        return self._f.free_indices()

    def index_dimensions(self):
        return self._f.index_dimensions()

    def shape(self):
        return self._f.shape()[1:]

    def __str__(self):
        return "nabla_div(%s)" % self._f

    def __repr__(self):
        return "NablaDiv(%r)" % self._f

class Curl(CompoundDerivative):
    __slots__ = ("_f", "_shape",)

    def __new__(cls, f):
        # Validate input
        sh = f.shape()
        ufl_assert(f.shape() in ((), (2,), (3,)), "Expecting a scalar, 2D vector or 3D vector.")
        ufl_assert(not f.free_indices(), \
            "TODO: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")

        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            sh = { (): (2,), (2,): (), (3,): (3,) }[sh]
            #free_indices = f.free_indices()
            #index_dimensions = subdict(f.index_dimensions(), free_indices)
            #return Zero((f.geometric_dimension(),), free_indices, index_dimensions)
            return Zero(sh)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        sh = { (): (2,), (2,): (), (3,): (3,) }[f.shape()]
        self._f = f
        self._shape = sh

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
        return "Curl(%r)" % self._f
