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
from ufl.core.expr import Expr
from ufl.core.terminal import Terminal
from ufl.core.operator import Operator
from ufl.exprcontainers import ExprList, ExprMapping
from ufl.constantvalue import Zero
from ufl.coefficient import Coefficient
from ufl.core.multiindex import Index, FixedIndex, MultiIndex
from ufl.indexed import Indexed
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

@ufl_type(num_ops=2)
class VariableDerivative(Derivative):
    __slots__ = ("ufl_shape", "ufl_free_indices", "ufl_index_dimensions",)
    def __new__(cls, f, v):
        # Checks
        ufl_assert(isinstance(f, Expr), "Expecting an Expr in VariableDerivative.")
        ufl_assert(isinstance(v, (Variable, Coefficient)), "Expecting a Variable in VariableDerivative.")
        ufl_assert(not v.ufl_free_indices, "Differentiation variable cannot have free indices.")

        # Simplification
        # Return zero if expression is trivially independent of variable
        if f._ufl_is_terminal_:
            return Zero(f.ufl_shape + v.ufl_shape, f.ufl_free_indices, f.ufl_index_dimensions)

        # Construction
        return Derivative.__new__(cls)

    def __init__(self, f, v):
        Derivative.__init__(self, (f, v))
        self.ufl_free_indices = f.ufl_free_indices
        self.ufl_index_dimensions = f.ufl_index_dimensions
        self.ufl_shape = f.ufl_shape + v.ufl_shape

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
            return Zero(f.ufl_shape + (dim,), f.ufl_free_indices, f.ufl_index_dimensions)

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        self._dim = f.geometric_dimension()

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        if op.is_cellwise_constant():
            ufl_assert(op.ufl_shape == self.ufl_operands[0].ufl_shape,
                       "Operand shape mismatch in Grad reconstruct.")
            ufl_assert(self.ufl_operands[0].ufl_free_indices == op.ufl_free_indices,
                       "Free index mismatch in Grad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices, self.ufl_index_dimensions)
        return self._ufl_class_(op)

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
            dim = f.domain().topological_dimension()
            return Zero(f.ufl_shape + (dim,), f.ufl_free_indices, f.ufl_index_dimensions)
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
            ufl_assert(self.ufl_operands[0].ufl_free_indices == op.ufl_free_indices,
                       "Free index mismatch in ReferenceGrad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices, self.ufl_index_dimensions)
        return self._ufl_class_(op)

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
        ufl_assert(not f.ufl_free_indices,
            "Free indices in the divergence argument is not allowed.")

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
            return Zero((dim,) + f.ufl_shape, f.ufl_free_indices, f.ufl_index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        self._dim = f.geometric_dimension()

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        if op.is_cellwise_constant():
            ufl_assert(op.ufl_shape == self.ufl_operands[0].ufl_shape,
                       "Operand shape mismatch in NablaGrad reconstruct.")
            ufl_assert(self.ufl_operands[0].ufl_free_indices == op.ufl_free_indices,
                       "Free index mismatch in NablaGrad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices,
                        self.ufl_index_dimensions)
        return self._ufl_class_(op)

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
        ufl_assert(not f.ufl_free_indices,
            "Free indices in the divergence argument is not allowed.")

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
        ufl_assert(not f.ufl_free_indices,
            "Free indices in the curl argument is not allowed.")

        # Return zero if expression is trivially constant
        if f.is_cellwise_constant():
            sh = { (): (2,), (2,): (), (3,): (3,) }[sh]
            return Zero(sh) # No free indices asserted above
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        global _curl_shapes
        CompoundDerivative.__init__(self, (f,))
        self.ufl_shape = _curl_shapes[f.ufl_shape]

    def __str__(self):
        return "curl(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Curl(%r)" % self.ufl_operands[0]
