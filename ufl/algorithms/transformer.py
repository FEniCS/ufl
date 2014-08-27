"""This module defines the Transformer base class and some
basic specializations to further base other algorithms upon,
as well as some utilities for easier application of such
algorithms."""

# Copyright (C) 2008-2014 Martin Sandve Alnes and Anders Logg
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
# Modified by Anders Logg, 2009-2010

from inspect import getargspec
from ufl.log import error, debug
from ufl.assertions import ufl_assert

from ufl.classes import Expr, Terminal, Variable, Zero, all_ufl_classes
from ufl.algorithms.map_integrands import map_integrands

from ufl.integral import Integral
from ufl.form import Form


def is_post_handler(function):
    "Is this a handler that expects transformed children as input?"
    insp = getargspec(function)
    num_args = len(insp[0]) + int(insp[1] is not None)
    visit_children_first = num_args > 2
    return visit_children_first

class Transformer(object):
    """Base class for a visitor-like algorithm design pattern used to
    transform expression trees from one representation to another."""
    _handlers_cache = {}
    def __init__(self, variable_cache=None):
        if variable_cache is None:
            variable_cache = {}
        self._variable_cache = variable_cache

        # Analyse class properties and cache handler data the
        # first time this is run for a particular class
        cache_data = Transformer._handlers_cache.get(type(self))
        if not cache_data:
            cache_data = [None]*len(all_ufl_classes)
            # For all UFL classes
            for classobject in all_ufl_classes:
                # Iterate over the inheritance chain
                # (NB! This assumes that all UFL classes inherits a single
                # Expr subclass and that this is the first superclass!)
                for c in classobject.mro():
                    # Register classobject with handler for the first encountered superclass
                    name = c._ufl_handler_name_
                    function = getattr(self, name, None)
                    if function:
                        cache_data[classobject._ufl_typecode_] = name, is_post_handler(function)
                        break
            Transformer._handlers_cache[type(self)] = cache_data

        # Build handler list for this particular class (get functions bound to self)
        self._handlers = [(getattr(self, name), post) for (name, post) in cache_data]

        # Keep a stack of objects visit is called on, to ease backtracking
        self._visit_stack = []

    def print_visit_stack(self):
        print("/"*80)
        print("Visit stack in Transformer:")
        def sstr(s):
            ss = str(type(s)) + " ; "
            n = 160 - len(ss)
            return ss + str(s)[:n]
        print("\n".join(sstr(s) for s in self._visit_stack))
        print("\\"*80)

    def visit(self, o):
        #debug("Visiting object of type %s." % type(o).__name__)
        # Update stack
        self._visit_stack.append(o)

        # Get handler for the UFL class of o (type(o) may be an external subclass of the actual UFL class)
        h, visit_children_first = self._handlers[o._ufl_typecode_]

        #if not h:
        #    # Failed to find a handler! Should never happen, but will happen if a non-Expr object is visited.
        #    error("Can't handle objects of type %s" % str(type(o)))

        # Is this a handler that expects transformed children as input?
        if visit_children_first:
            # Yes, visit all children first and then call h.
            r = h(o, *[self.visit(op) for op in o.ufl_operands])
        else:
            # No, this is a handler that handles its own children
            # (arguments self and o, where self is already bound)
            r = h(o)

        # Update stack and return
        self._visit_stack.pop()
        return r

    def undefined(self, o):
        "Trigger error."
        error("No handler defined for %s." % o._ufl_class_.__name__)

    def reuse(self, o):
        "Always reuse Expr (ignore children)"
        return o

    def reuse_if_possible(self, o, *operands):
        "Reuse Expr if possible, otherwise reconstruct from given operands."

        ufl_assert(len(operands) == len(o.ufl_operands), "Expecting number of operands to match")

        # TODO: Try using hashes of operands instead for a faster probability based version? One benchmark showed == to be faster.
        #if all(op0 is op1 for op0, op1 in zip(operands, o.ufl_operands)):
        #    return o
        #if all(op0 is op1 or hash(op0) == hash(op1) for op0, op1 in zip(operands, o.ufl_operands)):
        #    return o
        #if all(hash(op0) == hash(op1) for op0, op1 in zip(operands, o.ufl_operands)):
        #    return o
        if operands == o.ufl_operands:
            return o
        return o.reconstruct(*operands)

    def always_reconstruct(self, o, *operands):
        "Always reconstruct expr."
        return o.reconstruct(*operands)

    # Set default behaviour for any Expr
    expr = undefined

    # Set default behaviour for any Terminal
    terminal = reuse

    def reuse_variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.ufl_operands
        v = self._variable_cache.get(l)
        if v is not None:
            return v

        # Visit the expression our variable represents
        e2 = self.visit(e)

        # If the expression is the same, reuse Variable object
        if e == e2:
            v = o
        else:
            # Recreate Variable (with same label)
            v = Variable(e2, l)

        # Cache variable
        self._variable_cache[l] = v
        return v

    def reconstruct_variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.ufl_operands
        v = self._variable_cache.get(l)
        if v is not None:
            return v

        # Visit the expression our variable represents
        e2 = self.visit(e)

        # Always reconstruct Variable (with same label)
        v = Variable(e2, l)
        self._variable_cache[l] = v
        return v


class ReuseTransformer(Transformer):
    def __init__(self, variable_cache=None):
        Transformer.__init__(self, variable_cache)

    # Set default behaviour for any Expr
    expr = Transformer.reuse_if_possible

    # Set default behaviour for any Terminal
    terminal = Transformer.reuse

    # Set default behaviour for Variable
    variable = Transformer.reuse_variable


class CopyTransformer(Transformer):
    def __init__(self, variable_cache=None):
        Transformer.__init__(self, variable_cache)

    # Set default behaviour for any Expr
    expr = Transformer.always_reconstruct

    # Set default behaviour for any Terminal
    terminal = Transformer.reuse

    # Set default behaviour for Variable
    variable = Transformer.reconstruct_variable


class VariableStripper(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)

    def variable(self, o):
        return self.visit(o.ufl_operands[0])


def apply_transformer(e, transformer, integral_type=None):
    """Apply transformer.visit(expression) to each integrand
    expression in form, or to form if it is an Expr."""
    return map_integrands(lambda expr: transformer.visit(expr), e, integral_type)

def ufl2ufl(e):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    return apply_transformer(e, ReuseTransformer())

def ufl2uflcopy(e):
    """Convert an UFL expression to a new UFL expression.
    All nonterminal object instances are replaced with identical
    copies, while terminal objects are kept. This is used for
    testing that objects in the expression behave as expected."""
    return apply_transformer(e, CopyTransformer())

def strip_variables(e):
    "Replace all Variable instances with the expression they represent."
    return apply_transformer(e, VariableStripper())
