"""Algorithms for estimating polynomial degrees of expressions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010
# Modified by Jan Blechta, 2012
# Modified by Pablo Brubeck, Jørgen S. Dokken 2026

import warnings
from functools import singledispatchmethod

import ufl.classes
from ufl.argument import Argument
from ufl.checks import is_cellwise_constant
from ufl.coefficient import Coefficient
from ufl.constantvalue import IntValue
from ufl.core.multiindex import FixedIndex
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import extract_domains, extract_unique_domain
from ufl.form import Form
from ufl.integral import Integral
from ufl.utils.indexflattening import flatten_multiindex, shape_to_strides


class SumDegreeEstimator(DAGTraverser):
    """Sum degree estimator.

    This algorithm is exact for a few operators and heuristic for many.
    """

    def __init__(self, default_degree, element_replace_map):
        """Initialise."""
        super().__init__()
        self.default_degree = default_degree
        self.element_replace_map = element_replace_map

    # --- Helper functions shared by several rules

    def _reduce_degree(self, v, f):
        """Reduce the estimated degree by one.

        This is used when derivatives are taken. It does not reduce the degree when
        TensorProduct elements or quadrilateral elements are involved.
        """
        # Can have multiple domains e.g. in mixed cell type problems.
        cells = set(d.ufl_cell() for d in extract_domains(v))
        if isinstance(f, int) and all(cell.is_simplex for cell in cells):
            return max(f - 1, 0)
        else:
            return f

    def _add_degrees(self, v, *ops):
        """Apply to _add_degrees."""
        if any(isinstance(o, tuple) for o in ops):
            # we can add a slight hack here to handle things
            # like adding 0 to (3, 3) [by expanding
            # 0 to (0, 0) when making tempops]
            tempops = [foo if isinstance(foo, tuple) else (foo, foo) for foo in ops]
            return tuple(map(sum, zip(*tempops)))
        else:
            return sum(ops)

    def _max_degrees(self, v, *ops):
        """Apply to _max_degrees."""
        if any(isinstance(o, tuple) for o in ops):
            tempops = [foo if isinstance(foo, tuple) else (foo, foo) for foo in ops]
            return tuple(map(max, zip(*tempops)))
        else:
            return max(ops + (0,))

    def _not_handled(self, v, *args):
        """Apply to _not_handled."""
        raise ValueError(f"Missing degree handler for type {v._ufl_class_.__name__}")

    # --- Default rule for operators without a specific handler

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr):
        """Process ``o``."""
        return super().process(o)

    @process.register(ufl.classes.Expr)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        """Apply to expr.

        For most operators we take the max degree of its operands.
        """
        warnings.warn(f"Missing degree estimation handler for type {o._ufl_class_.__name__}")
        return self._add_degrees(o, *ops)

    # --- Terminals; these are cutoff rules, i.e. they have no operands to
    # --- process and therefore are not decorated with ``postorder``

    # Constant values and constants are constant.
    @process.register(ufl.classes.ConstantValue)
    @process.register(ufl.classes.Constant)
    def _(self, o):
        return 0

    @process.register(ufl.classes.GeometricQuantity)
    def _(self, o):
        """Apply to geometric_quantity.

        Some geometric quantities are cellwise constant. Others are
        nonpolynomial and thus hard to estimate.
        """
        if is_cellwise_constant(o):
            return 0
        else:
            # As a heuristic, just returning domain degree to bump up degree somewhat
            return extract_unique_domain(o).ufl_coordinate_element().embedded_superdegree

    @process.register(ufl.classes.SpatialCoordinate)
    def _(self, o):
        """Apply to spatial_coordinate.

        A coordinate provides additional degrees depending on coordinate field of domain.
        """
        return extract_unique_domain(o).ufl_coordinate_element().embedded_superdegree

    @process.register(ufl.classes.CellCoordinate)
    def _(self, o):
        """Apply to cell_coordinate.

        A coordinate provides one additional degree.
        """
        return 1

    @process.register(Argument)
    def _(self, o):
        """Apply to argument.

        A form argument provides a degree depending on the element,
        or the default degree if the element has no degree.
        """
        # For a mixed element this is the max degree over all sub-elements;
        # the Indexed rule refines this to the accessed sub-element's own
        # degree when the specific component is known.
        return o.ufl_element().embedded_superdegree

    @process.register(Coefficient)
    def _(self, o):
        """Apply to coefficient.

        A form argument provides a degree depending on the element,
        or the default degree if the element has no degree.
        """
        e = o.ufl_element()
        e = self.element_replace_map.get(e, e)
        # See the comment in the Argument rule above.
        d = e.embedded_superdegree
        if d is None:
            d = self.default_degree
        return d

    # Utility types with no degree concept
    @process.register(ufl.classes.MultiIndex)
    @process.register(ufl.classes.Label)
    def _(self, o):
        return None

    # --- Operators

    @process.register(ufl.classes.Interpolate)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        """Apply to interpolate.

        An interpolated field has the polynomial degree of its target element.
        """
        e = o.ufl_element()
        e = self.element_replace_map.get(e, e)
        d = e.embedded_superdegree
        if d is None:
            d = self.default_degree
        return d

    # Fall-through types: the degree of the single operand is unchanged.
    @process.register(ufl.classes.ReferenceValue)
    @process.register(ufl.classes.Transposed)
    @process.register(ufl.classes.PositiveRestricted)
    @process.register(ufl.classes.NegativeRestricted)
    @process.register(ufl.classes.Conj)
    @process.register(ufl.classes.Real)
    @process.register(ufl.classes.Imag)
    @DAGTraverser.postorder
    def _(self, o, a):
        return a

    # Indexing and similar types: the degree of the first operand is
    # unchanged, the index carries no degree.
    @process.register(ufl.classes.Variable)
    @process.register(ufl.classes.IndexSum)
    @process.register(ufl.classes.ComponentTensor)
    @DAGTraverser.postorder
    def _(self, o, a, ii):
        return a

    @process.register(ufl.classes.Indexed)
    @DAGTraverser.postorder
    def _(self, o, A, ii):
        """Apply to indexed.

        A fixed-index component of a mixed-element Argument or
        Coefficient may belong to a lower-degree sub-element than the
        whole mixed element, so look up that sub-element's own degree
        instead of falling back to A, the whole element's degree.
        """
        op = o.ufl_operands[0]
        multiindex = o.ufl_operands[1]
        if isinstance(op, (Argument, Coefficient)) and all(
            isinstance(idx, FixedIndex) for idx in multiindex
        ):
            element = op.ufl_element()
            if isinstance(op, Coefficient):
                element = self.element_replace_map.get(element, element)
            sub_elements = element.sub_elements
            if sub_elements and len(multiindex) == len(op.ufl_shape):
                component = flatten_multiindex(
                    [int(idx) for idx in multiindex], shape_to_strides(op.ufl_shape)
                )
                # Walk the sub-elements in order to find which one covers
                # this flattened component.
                offset = 0
                for sub_element in sub_elements:
                    sub_size = sub_element.reference_value_size
                    if component < offset + sub_size:
                        d = sub_element.embedded_superdegree
                        return self.default_degree if d is None else d
                    offset += sub_size
        return A

    # A sum takes the max degree of its operands, and so does a list tensor:
    @process.register(ufl.classes.Sum)
    @process.register(ufl.classes.ListTensor)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        return self._max_degrees(o, *ops)

    # A product accumulates the degrees of its operands:
    @process.register(ufl.classes.Product)
    # Handling these types although they should not occur... please
    # apply preprocessing before using this algorithm:
    @process.register(ufl.classes.Inner)
    @process.register(ufl.classes.Dot)
    @process.register(ufl.classes.Outer)
    @process.register(ufl.classes.Cross)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        return self._add_degrees(o, *ops)

    # TODO: Need a new algorithm which considers direction of
    # derivatives of form arguments A spatial derivative reduces the
    # degree with one
    @process.register(ufl.classes.Grad)
    @process.register(ufl.classes.ReferenceGrad)
    # Handling these types although they should not occur... please
    # apply preprocessing before using this algorithm:
    @process.register(ufl.classes.NablaGrad)
    @process.register(ufl.classes.Div)
    @process.register(ufl.classes.ReferenceDiv)
    @process.register(ufl.classes.NablaDiv)
    @process.register(ufl.classes.Curl)
    @process.register(ufl.classes.ReferenceCurl)
    @DAGTraverser.postorder
    def _(self, o, f):
        return self._reduce_degree(o, f)

    # Explicitly not handling these types, please apply preprocessing
    # before using this algorithm. The base types cover the compounds
    # (Trace, Determinant, Cofactor, Inverse, Deviatoric, Skew, Sym) and
    # the derivatives (VariableDerivative and the CompoundDerivative
    # types not handled above).
    @process.register(ufl.classes.CompoundTensorOperator)
    @process.register(ufl.classes.Derivative)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        return self._not_handled(o, *ops)

    @process.register(ufl.classes.CellAvg)
    @process.register(ufl.classes.FacetAvg)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Apply to cell_avg and facet_avg.

        Cell and facet averages of a function are always cellwise constant.
        """
        return 0

    @process.register(ufl.classes.Abs)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Apply to abs.

        This is a heuristic, correct if there is no.
        """
        if a == 0:
            return a
        else:
            return a

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        """Apply to division.

        Using the sum here is a heuristic. Consider e.g. (x+1)/(x-1).
        """
        return self._add_degrees(o, *ops)

    @process.register(ufl.classes.Power)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        """Apply to power.

        If b is a positive integer: degree(a**b) == degree(a)*b
        otherwise use the heuristic: degree(a**b) == degree(a) + 2.
        """
        _f, g = o.ufl_operands

        if isinstance(g, IntValue):
            gi = g.value()
            if gi >= 0:
                if isinstance(a, int):
                    return a * gi
                else:
                    return tuple(foo * gi for foo in a)

        # Something to a non-(positive integer) power, e.g. float,
        # negative integer, Coefficient, etc.
        return self._add_degrees(o, a, 2)

    @process.register(ufl.classes.Atan2)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        """Apply to atan2.

        Using the heuristic:
        degree(atan2(const,const)) == 0
        degree(atan2(a,b)) == max(degree(a),degree(b))+2
        which can be wildly inaccurate but at least gives a somewhat
        high integration degree.
        """
        if a or b:
            return self._add_degrees(o, self._max_degrees(o, a, b), 2)
        else:
            return self._max_degrees(o, a, b)

    @process.register(ufl.classes.MathFunction)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Apply to math_function.

        Using the heuristic:
        degree(sin(const)) == 0
        degree(sin(a)) == degree(a)+2
        which can be wildly inaccurate but at least gives a somewhat
        high integration degree.
        """
        if a:
            return self._add_degrees(o, a, 2)
        else:
            return a

    @process.register(ufl.classes.BesselFunction)
    @DAGTraverser.postorder
    def _(self, o, nu, x):
        """Apply to bessel_function.

        Using the heuristic
        degree(bessel_*(const)) == 0
        degree(bessel_*(x)) == degree(x)+2
        which can be wildly inaccurate but at least gives a somewhat
        high integration degree.
        """
        if x:
            return self._add_degrees(o, x, 2)
        else:
            return x

    @process.register(ufl.classes.Condition)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        return None

    @process.register(ufl.classes.Conditional)
    @DAGTraverser.postorder
    def _(self, o, c, t, f):
        """Apply to conditional.

        Degree of condition does not influence degree of values which conditional takes. So
        heuristicaly taking max of true degree and false degree. This will be exact in cells
        where condition takes single value. For improving accuracy of quadrature near
        condition transition surface quadrature order must be adjusted manually.
        """
        return self._max_degrees(o, t, f)

    @process.register(ufl.classes.MinValue)
    @process.register(ufl.classes.MaxValue)
    @DAGTraverser.postorder
    def _(self, o, a, r):
        """Apply to min_value and max_value.

        Same as conditional.
        """
        return self._max_degrees(o, a, r)

    @process.register(ufl.classes.CoordinateDerivative)
    @DAGTraverser.postorder
    def _(self, o, integrand_degree, b, direction_degree, d):
        """Apply to coordinate_derivative.

        We use the heuristic that a shape derivative in direction V
        introduces terms V and grad(V) into the integrand. Hence we add the
        degree of the deformation to the estimate.
        """
        return self._add_degrees(o, integrand_degree, direction_degree)

    @process.register(ufl.classes.ExprList)
    @process.register(ufl.classes.ExprMapping)
    @DAGTraverser.postorder
    def _(self, o, *ops):
        return self._max_degrees(o, *ops)


def estimate_total_polynomial_degree(e, default_degree=1, element_replace_map={}):
    """Estimate total polynomial degree of integrand.

    NB: Although some compound types are supported here,
    some derivatives and compounds must be preprocessed
    prior to degree estimation. In generic code, this algorithm
    should only be applied after preprocessing.

    For coefficients defined on an element with unspecified degree
    (None), the degree is set to the given default degree.
    """
    de = SumDegreeEstimator(default_degree, element_replace_map)
    if isinstance(e, Form):
        if not e.integrals():
            raise ValueError("Form has no integrals.")
        degrees = [de(it.integrand()) for it in e.integrals()]
    elif isinstance(e, Integral):
        degrees = [de(e.integrand())]
    else:
        degrees = [de(e)]
    degree = max(degrees) if degrees else default_degree
    return degree
