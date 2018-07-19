# -*- coding: utf-8 -*-
"""Algorithm for removing conj, real, and imag nodes
from a form for when the user is in 'real mode'"""

from ufl.corealg.multifunction import MultiFunction
from ufl.constantvalue import ComplexValue
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.log import error


class ComplexNodeRemoval(MultiFunction):
    """Replaces complex operator nodes with their children"""
    expr = MultiFunction.reuse_if_untouched

    def conj(self, o, a):
        return a

    def real(self, o, a):
        return a

    def imag(self, o, a):
        error("Unexpected imag in real expression.")

    def terminal(self, t, *ops):
        if isinstance(t, ComplexValue):
            error('Unexpected complex value in real expression.')
        else:
            return t


def remove_complex_nodes(expr):
    """Replaces complex operator nodes with their children. This is called
    during compute_form_data if the compiler wishes to compile
    real-valued forms. In essence this strips all trace of complex
    support from the preprocessed form.
    """
    return map_integrand_dags(ComplexNodeRemoval(), expr)
