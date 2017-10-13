# -*- coding: utf-8 -*-
"""Algorithm for removing conj, real, and imag nodes
from a form for when the user is in 'real mode'"""

from ufl.corealg.multifunction import MultiFunction
from ufl.constantvalue import ComplexValue, FloatValue
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.log import error


class ComplexNodeRemoval(MultiFunction):
    """Replaces complex operator nodes with their children"""
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def conj(self, o, a):
        return a

    def real(self, o, a):
        return a

    def imag(self, o, a):
        return a

    def terminal(self, t, *ops):
        if isinstance(t, ComplexValue):
            error('Cannot cast ComplexValues to FloatValues!')
        else:
            return t


def remove_complex_nodes(expr):
    """Replaces complex operator nodes with their children"""
    return map_integrand_dags(ComplexNodeRemoval(), expr)
