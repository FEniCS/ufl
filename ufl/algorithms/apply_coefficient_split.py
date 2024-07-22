"""Apply coefficient split.

This module contains classes and functions to split coefficients defined on mixed function spaces.
"""

import functools
from collections import defaultdict
import numpy
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import Restricted
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_unique_domain, MixedMesh
from ufl.measure import integral_type_to_measure_name
from ufl.sobolevspace import H1
from ufl.geometry import GeometricQuantity
from ufl.classes import Coefficient, Form, ReferenceGrad, ReferenceValue, Restricted, Indexed, MultiIndex, Index, FixedIndex, ComponentTensor, ListTensor, Zero, NegativeRestricted, PositiveRestricted, SingleValueRestricted, ToBeRestricted
from ufl import indices
from ufl.checks import is_cellwise_constant
from ufl.tensors import as_tensor


class CoefficientSplitter(MultiFunction):

    def __init__(self, coefficient_split):
        MultiFunction.__init__(self)
        self._coefficient_split = coefficient_split

    expr = MultiFunction.reuse_if_untouched

    def modified_terminal(self, o):
        restriction = None
        local_derivatives = 0
        reference_value = False
        t = o
        while not t._ufl_is_terminal_:
            assert t._ufl_is_terminal_modifier_, f"Got {repr(t)}"
            if isinstance(t, ReferenceValue):
                assert not reference_value, "Got twice pulled back terminal!"
                reference_value = True
                t, = t.ufl_operands
            elif isinstance(t, ReferenceGrad):
                local_derivatives += 1
                t, = t.ufl_operands
            elif isinstance(t, Restricted):
                assert restriction is None, "Got twice restricted terminal!"
                restriction = t._side
                t, = t.ufl_operands
            elif t._ufl_terminal_modifiers_:
                raise ValueError("Missing handler for terminal modifier type %s, object is %s." % (type(t), repr(t)))
            else:
                raise ValueError("Unexpected type %s object %s." % (type(t), repr(t)))
        if not isinstance(t, Coefficient):
            # Only split coefficients
            return o
        if t not in self._coefficient_split:
            # Only split mixed coefficients
            return o
        # Reference value expected
        assert reference_value
        # Derivative indices
        beta = indices(local_derivatives)
        components = []
        for subcoeff in self._coefficient_split[t]:
            c = subcoeff
            # Apply terminal modifiers onto the subcoefficient
            if reference_value:
                c = ReferenceValue(c)
            for n in range(local_derivatives):
                # Return zero if expression is trivially constant. This has to
                # happen here because ReferenceGrad has no access to the
                # topological dimension of a literal zero.
                if is_cellwise_constant(c):
                    dim = extract_unique_domain(subcoeff).topological_dimension()
                    c = Zero(c.ufl_shape + (dim,), c.ufl_free_indices, c.ufl_index_dimensions)
                else:
                    c = ReferenceGrad(c)
            if restriction == '+':
                c = PositiveRestricted(c)
            elif restriction == '-':
                c = NegativeRestricted(c)
            elif restriction == '|':
                c = SingleValueRestricted(c)
            elif restriction == '?':
                c = ToBeRestricted(c)
            elif restriction is not None:
                raise RuntimeError(f"Got unknown restriction: {restriction}")
            # Collect components of the subcoefficient
            for alpha in numpy.ndindex(subcoeff.ufl_element().reference_value_shape):
                # New modified terminal: component[alpha + beta]
                components.append(c[alpha + beta])
        # Repack derivative indices to shape
        c, = indices(1)
        return ComponentTensor(as_tensor(components)[c], MultiIndex((c,) + beta))

    positive_restricted = modified_terminal
    negative_restricted = modified_terminal
    single_value_restricted = modified_terminal
    to_be_restricted = modified_terminal
    reference_grad = modified_terminal
    reference_value = modified_terminal
    terminal = modified_terminal


def apply_coefficient_split(expr, coefficient_split):
    """Split mixed coefficients, so mixed elements need not be
    implemented.

    :arg split: A :py:class:`dict` mapping each mixed coefficient to a
                sequence of subcoefficients.  If None, calling this
                function is a no-op.
    """
    if coefficient_split is None:
        return expr
    splitter = CoefficientSplitter(coefficient_split)
    return map_expr_dag(splitter, expr)


class FixedIndexRemover(MultiFunction):

    def __init__(self, fimap):
        MultiFunction.__init__(self)
        self.fimap = fimap
        self._object_cache = {}

    expr = MultiFunction.reuse_if_untouched

    @staticmethod
    def _cached(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            o, = args
            if o in self._object_cache:
                return self._object_cache[o]
            else:
                return self._object_cache.setdefault(o, f(self, o))
        return wrapper

    @_cached
    def zero(self, o):
        free_indices = []
        index_dimensions = []
        for i, d in zip(o.ufl_free_indices, o.ufl_index_dimensions):
            if Index(i) in self.fimap:
                ind_j = self.fimap[Index(i)]
                if not isinstance(ind_j, FixedIndex):
                    free_indices.append(ind_j.count())
                    index_dimensions.append(d)
            else:
                free_indices.append(i)
                index_dimensions.append(d)
        return Zero(shape=o.ufl_shape, free_indices=tuple(free_indices), index_dimensions=tuple(index_dimensions))

    @_cached
    def list_tensor(self, o):
        cc = []
        for o1 in o.ufl_operands:
            comp = map_expr_dag(self, o1)
            cc.append(comp)
        return ListTensor(*cc)

    @_cached
    def multi_index(self, o):
        return MultiIndex(tuple(self.fimap.get(i, i) for i in o.indices()))


class IndexRemover(MultiFunction):

    def __init__(self):
        MultiFunction.__init__(self)
        self._object_cache = {}

    expr = MultiFunction.reuse_if_untouched

    @staticmethod
    def _cached(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            o, = args
            if o in self._object_cache:
                return self._object_cache[o]
            else:
                return self._object_cache.setdefault(o, f(self, o))
        return wrapper

    @_cached
    def _zero_simplify(self, o):
        operand, = o.ufl_operands
        operand = map_expr_dag(self, operand)
        if isinstance(operand, Zero):
            return Zero(shape=o.ufl_shape, free_indices=o.ufl_free_indices, index_dimensions=o.ufl_index_dimensions)
        else:
            return o._ufl_expr_reconstruct_(operand)

    @_cached
    def indexed(self, o):
        o1, i1 = o.ufl_operands
        if isinstance(o1, ComponentTensor):
            o2, i2 = o1.ufl_operands
            fimap = dict(zip(i2.indices(), i1.indices(), strict=True))
            rule = FixedIndexRemover(fimap)
            v = map_expr_dag(rule, o2)
            return map_expr_dag(self, v)
        elif isinstance(o1, ListTensor):
            if isinstance(i1[0], FixedIndex):
                o1 = o1.ufl_operands[i1[0]._value]
                if len(i1) > 1:
                    i1 = MultiIndex(i1[1:])
                    return map_expr_dag(self, Indexed(o1, i1))
                else:
                    return map_expr_dag(self, o1)
        o1 = map_expr_dag(self, o1)
        return Indexed(o1, i1)

    # Do something nicer
    positive_restricted = _zero_simplify
    negative_restricted = _zero_simplify
    single_value_restricted = _zero_simplify
    to_be_restricted = _zero_simplify
    reference_grad = _zero_simplify
    reference_value = _zero_simplify


def remove_component_and_list_tensors(o):
    if isinstance(o, Form):
        integrals = []
        for integral in o.integrals():
            integrand = remove_component_and_list_tensors(integral.integrand())
            if not isinstance(integrand, Zero):
                integrals.append(integral.reconstruct(integrand=integrand))
        return o._ufl_expr_reconstruct_(integrals)
    else:
        rule = IndexRemover()
        return map_expr_dag(rule, o)
