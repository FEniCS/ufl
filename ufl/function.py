"""This module defines the Function class and a number 
of related classes (functions), including Constant."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-03-05"

# Modified by Anders Logg, 2008

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.common import Counted, product
from ufl.terminal import FormArgument
from ufl.finiteelement import FiniteElementBase, FiniteElement, VectorElement, TensorElement
from ufl.split_functions import split
from ufl.geometry import as_cell

# --- The Function class represents a coefficient function argument to a form ---

class Function(FormArgument, Counted):
    __slots__ = ("_element", "_repr", "_gradient", "_derivatives")
    _globalcount = 0

    def __init__(self, element, count=None, gradient=None, derivatives=None):
        FormArgument.__init__(self)
        Counted.__init__(self, count, Function)
        ufl_assert(isinstance(element, FiniteElementBase),
            "Expecting a FiniteElementBase instance.")
        self._element = element
        self._repr = None
        self._gradient = gradient
        self._derivatives = {} if derivatives is None else dict(derivatives)
        if gradient or derivatives:
            # TODO: Use gradient and derivatives in AD
            # TODO: Check shapes of gradient and derivatives
            warning("Specifying the gradient or derivatives of a Function is not yet implemented anywhere.")
    
    def reconstruct(self, count=None):
        if count is None or count == self._count:
            return self
        return Function(self._element, count)
    
    def gradient(self):
        "Hook for experimental feature, do not use!"
        return self._gradient
    
    def derivative(self, f):
        "Hook for experimental feature, do not use!"
        return self._derivatives.get(f)
    
    def element(self):
        return self._element
    
    def shape(self):
        return self._element.value_shape()
    
    def cell(self):
        return self._element.cell()
    
    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "w_%s" % count
        else:
            return "w_{%s}" % count
    
    def __repr__(self):
        if self._repr is None:
            self._repr = "Function(%r, %r)" % (self._element, self._count)
        return self._repr

    def __eq__(self, other):
        return isinstance(other, Function) and self._element == other._element and self._count == other._count

# --- Subclasses for defining constant functions without specifying element ---

# TODO: Handle actual global constants?

class Constant(Function):
    __slots__ = ()

    def __init__(self, cell, count=None):
        e = FiniteElement("DG", cell, 0)
        Function.__init__(self, e, count)
        self._repr = "Constant(%r, %r)" % (e.cell(), self._count)
    
    def reconstruct(self, count=None):
        if count is None or count == self._count:
            return self
        return Constant(self._element.cell(), count)
    
    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "c_%s" % count
        else:
            return "c_{%s}" % count

class VectorConstant(Function):
    __slots__ = ()
    
    def __init__(self, cell, dim=None, count=None):
        e = VectorElement("DG", cell, 0, dim)
        Function.__init__(self, e, count)
        self._repr = "VectorConstant(%r, %r, %r)" % (e.cell(), e.value_shape()[0], self._count)
    
    def reconstruct(self, count=None):
        if count is None or count == self._count:
            return self
        e = self._element
        return VectorConstant(e.cell(), e.value_shape()[0], count)
    
    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "C_%s" % count
        else:
            return "C_{%s}" % count

class TensorConstant(Function):
    __slots__ = ()
    def __init__(self, cell, shape=None, symmetry=None, count=None):
        e = TensorElement("DG", cell, 0, shape=shape, symmetry=symmetry)
        Function.__init__(self, e, count)
        self._repr = "TensorConstant(%r, %r, %r, %r)" % (e.cell(), e.value_shape(), e._symmetry, self._count)
    
    def reconstruct(self, count=None):
        if count is None or count == self._count:
            return self
        e = self._element
        return TensorConstant(e.cell(), e.value_shape(), e._symmetry, count)
    
    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "C_%s" % count
        else:
            return "C_{%s}" % count

# --- Helper functions for subfunctions on mixed elements ---

def Functions(element):
    return split(Function(element))
