"Differential operators."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.common import subdict, mergedicts, EmptyDict
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.operatorbase import Operator
from ufl.exprcontainers import ExprList, ExprMapping
from ufl.constantvalue import Zero
from ufl.indexing import Index, FixedIndex, MultiIndex, as_multi_index
from ufl.indexed import Indexed
from ufl.indexutils import unique_indices
from ufl.variable import Variable
from ufl.precedence import parstr
from ufl.core.ufl_type import ufl_type

#--- Basic differentiation objects ---

@ufl_type(is_abstract=True)
class Derivative(Operator):
    "Base class for all derivative types."
    __slots__ = ()
    def __init__(self, operands):
        Operator.__init__(self, operands)

@ufl_type(num_ops=4, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class CoefficientDerivative(Derivative):
    """Derivative of the integrand of a form w.r.t. the
    degrees of freedom in a discrete Coefficient."""
    __slots__ = ()

    def __new__(cls, integrand, coefficients, arguments, coefficient_derivatives):
        ufl_assert(isinstance(coefficients, ExprList),
                   "Expecting ExprList instance with Coefficients.")
        ufl_assert(isinstance(arguments, ExprList),
                   "Expecting ExprList instance with Arguments.")
        ufl_assert(isinstance(coefficient_derivatives, ExprMapping),
                   "Expecting ExprMapping for coefficient derivatives.")
        if isinstance(integrand, Zero):
            return integrand
        return Derivative.__new__(cls)

    def __init__(self, integrand, coefficients, arguments, coefficient_derivatives):
        if not isinstance(coefficient_derivatives, ExprMapping):
            coefficient_derivatives = ExprMapping(coefficient_derivatives)
        Derivative.__init__(self, (integrand, coefficients, arguments, coefficient_derivatives))

    def __str__(self):
        return "d/dfj { %s }, with fh=%s, dfh/dfj = %s, and coefficient derivatives %s"\
            % (self.ufl_operands[0], self.ufl_operands[1], self.ufl_operands[2], self.ufl_operands[3])

    def __repr__(self):
        return "CoefficientDerivative(%r, %r, %r, %r)"\
            % (self.ufl_operands[0], self.ufl_operands[1], self.ufl_operands[2], self.ufl_operands[3])

def split_indices(expression, idx):
    idims = dict(expression.index_dimensions())
    if isinstance(idx, Index) and idims.get(idx) is None:
        idims[idx] = expression.geometric_dimension()
    fi = unique_indices(expression.free_indices() + (idx,))
    return fi, idims

@ufl_type(num_ops=2)
class VariableDerivative(Derivative):
    __slots__ = ("ufl_shape", "_free_indices", "_index_dimensions",)
    def __new__(cls, f, v):
        # Return zero if expression is trivially independent of variable
        if f._ufl_is_terminal_:
            free_indices = tuple(set(f.free_indices()) ^ set(v.free_indices()))
            index_dimensions = mergedicts([f.index_dimensions(), v.index_dimensions()])
            index_dimensions = subdict(index_dimensions, free_indices)
            return Zero(f.ufl_shape + v.ufl_shape, free_indices, index_dimensions)
        return Derivative.__new__(cls)

    def __init__(self, f, v):
        ufl_assert(isinstance(f, Expr), "Expecting an Expr in VariableDerivative.")
        if isinstance(v, Indexed):
            ufl_assert(isinstance(v._expression, Variable), \
                "Expecting a Variable in VariableDerivative.")
            error("diff(f, v[i]) isn't handled properly in all code.") # TODO: Should we allow this? Can do diff(f, v)[..., i], which leads to additional work but does the same.
        else:
            ufl_assert(isinstance(v, Variable), \
                "Expecting a Variable in VariableDerivative.")

        Derivative.__init__(self, (f, v))

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
        self.ufl_shape = f.ufl_shape + v.ufl_shape

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def __str__(self):
        if isinstance(self.ufl_operands[0], Terminal):
            return "d%s/d[%s]" % (self.ufl_operands[0], self.ufl_operands[1])
        return "d/d[%s] %s" % (self.ufl_operands[1], parstr(self.ufl_operands[0], self))

    def __repr__(self):
        return "VariableDerivative(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

#--- Compound differentiation objects ---

@ufl_type(is_abstract=True)
class CompoundDerivative(Derivative):
    "Base class for all compound derivative types."
    __slots__ = ()
    def __init__(self, operands):
        Derivative.__init__(self, operands)


@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class Grad(CompoundDerivative):
    __slots__ = ("_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            dim = f.geometric_dimension()
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero(f.ufl_shape + (dim,), free_indices, index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        self._dim = f.geometric_dimension()

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        if op.is_cellwise_constant():
            ufl_assert(op.ufl_shape == self.ufl_operands[0].ufl_shape,
                       "Operand shape mismatch in Grad reconstruct.")
            ufl_assert(self.ufl_operands[0].free_indices() == op.free_indices(),
                       "Free index mismatch in Grad reconstruct.")
            return Zero(self.ufl_shape, self.free_indices(),
                        self.index_dimensions())
        return self.__class__._ufl_class_(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        r = len(component)
        component, i = component[:-1], component[-1]
        derivatives = derivatives + (i,)
        result = self.ufl_operands[0].evaluate(x, mapping, component, index_values,
                                  derivatives=derivatives)
        return result

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape + (self._dim,)

    def __str__(self):
        return "grad(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Grad(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class ReferenceGrad(CompoundDerivative):
    __slots__ = ("_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            domain = f.domain()
            dim = domain.topological_dimension()
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero(f.ufl_shape + (dim,), free_indices, index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        domain = f.domain()
        dim = domain.topological_dimension()
        self._dim = dim

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        if op.is_cellwise_constant():
            ufl_assert(op.ufl_shape == self.ufl_operands[0].ufl_shape,
                       "Operand shape mismatch in ReferenceGrad reconstruct.")
            ufl_assert(self.ufl_operands[0].free_indices() == op.free_indices(),
                       "Free index mismatch in ReferenceGrad reconstruct.")
            return Zero(self.ufl_shape, self.free_indices(),
                        self.index_dimensions())
        return self.__class__._ufl_class_(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        r = len(component)
        component, i = component[:-1], component[-1]
        derivatives = derivatives + (i,)
        result = self.ufl_operands[0].evaluate(x, mapping, component, index_values,
                                  derivatives=derivatives)
        return result

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape + (self._dim,)

    def __str__(self):
        return "reference_grad(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "ReferenceGrad(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class Div(CompoundDerivative):
    __slots__ = ()

    def __new__(cls, f):
        ufl_assert(not f.free_indices(), \
            "TODO: Taking divergence of an expression with free indices,"\
            "should this be a valid expression? Please provide examples!")

        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            return Zero(f.ufl_shape[:-1]) # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape[:-1]

    def __str__(self):
        return "div(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Div(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class NablaGrad(CompoundDerivative):
    __slots__ = ("_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            dim = f.geometric_dimension()
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero((dim,) + f.ufl_shape, free_indices, index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        self._dim = f.geometric_dimension()

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        if op.is_cellwise_constant():
            ufl_assert(op.ufl_shape == self.ufl_operands[0].ufl_shape,
                       "Operand shape mismatch in NablaGrad reconstruct.")
            ufl_assert(self.ufl_operands[0].free_indices() == op.free_indices(),
                       "Free index mismatch in NablaGrad reconstruct.")
            return Zero(self.ufl_shape, self.free_indices(),
                        self.index_dimensions())
        return self.__class__._ufl_class_(op)

    @property
    def ufl_shape(self):
        return (self._dim,) + self.ufl_operands[0].ufl_shape

    def __str__(self):
        return "nabla_grad(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "NablaGrad(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class NablaDiv(CompoundDerivative):
    __slots__ = ()

    def __new__(cls, f):
        ufl_assert(not f.free_indices(), \
            "TODO: Taking divergence of an expression with free indices,"\
            "should this be a valid expression? Please provide examples!")

        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            return Zero(f.ufl_shape[1:]) # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape[1:]

    def __str__(self):
        return "nabla_div(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "NablaDiv(%r)" % self.ufl_operands[0]

_curl_shapes = { (): (2,), (2,): (), (3,): (3,) }
@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class Curl(CompoundDerivative):
    __slots__ = ("ufl_shape",)

    def __new__(cls, f):
        # Validate input
        sh = f.ufl_shape
        ufl_assert(f.ufl_shape in ((), (2,), (3,)), "Expecting a scalar, 2D vector or 3D vector.")
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
        global _curl_shapes
        CompoundDerivative.__init__(self, (f,))
        self.ufl_shape = _curl_shapes[f.ufl_shape]

    def __str__(self):
        return "curl(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Curl(%r)" % self.ufl_operands[0]
