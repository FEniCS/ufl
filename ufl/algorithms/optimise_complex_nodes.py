# -*- coding: utf-8 -*-
"""Algorithm for removing unnecessary complex node combinations."""

from ufl.corealg.multifunction import MultiFunction
from ufl.constantvalue import Zero
from ufl.algebra import Abs, Conj, Real, Imag
from ufl.algorithms.map_integrands import map_integrand_dags


class ComplexNodeOptimisation(MultiFunction):
    """Removes unecessary complex node combinations."""
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def conj(self, o, a):
        if isinstance(a, (Abs, Real, Imag, Zero)):
            return a
        elif isinstance(a, (Conj)):
            return a.ufl_operands[0]
        else:
            return o

    def abs(self, o, a):
        if isinstance(a, (Abs, Zero)):
            return a
        elif isinstance(a, (Conj)):
            return Abs(a.ufl_operands[0])
        else:
            return o

    def real(self, o, a):
        # Can't mess with Real too much due to its role in allowing valid comparisons
        if isinstance(a, (Real)):
            return a
        elif isinstance(a, (Conj)):
            return Real(a.ufl_operands[0])
        else:
            return o

    def imag(self, o, a):
        if isinstance(a, (Real, Imag, Abs)):
            return Zero(a.ufl_shape, a.ufl_free_indices, a.ufl_index_dimensions)
        else:
            return o


def optimise_complex_nodes(expr):
    """Removes unnecessary complex node combinations."""
    return map_integrand_dags(ComplexNodeOptimisation(), expr)
