"""This module defines the Terminal class, the superclass
for all types that are terminal nodes in the expression trees."""

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
# Modified by Anders Logg, 2008
#
# First added:  2008-03-14
# Last changed: 2011-08-08

from ufl.expr import Expr
from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import EmptyDict
from ufl.common import counted_init

#--- Base class for terminal objects ---

class Terminal(Expr):
    "A terminal node in the UFL expression tree."
    __slots__ = ()

    def __init__(self):
        Expr.__init__(self)

    def reconstruct(self, *operands):
        "Return self."
        operands and error("Got call to reconstruct in a terminal with non-empty operands.")
        return self

    def operands(self):
        "A Terminal object never has operands."
        return ()

    def free_indices(self):
        "A Terminal object never has free indices."
        return ()

    def index_dimensions(self):
        "A Terminal object never has free indices."
        return EmptyDict

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get self from mapping and return the component asked for."
        f = mapping.get(self)
        # No mapping, trying to evaluate self as a constant
        if f is None:
            try:
                f = float(self)
                if derivatives:
                    f = 0.0
                return f
            except:
                pass
            # If it has an ufl_evaluate function, call it
            if hasattr(self, 'ufl_evaluate'):
                return self.ufl_evaluate(x, component, derivatives)
            # Take component if any
            warning("Couldn't map '%s' to a float, returning ufl object without evaluation." % str(self))
            f = self
            if component:
                f = f[component]
            return f

        # Found a callable in the mapping
        if callable(f):
            if derivatives:
                f = f(x, derivatives)
            else:
                f = f(x)
        else:
            if derivatives:
                return 0.0

        # Take component if any (expecting nested tuple)
        for c in component:
            f = f[c]
        return f

    def signature_data(self):
        return repr(self)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        "Default comparison of terminals just compare repr strings."
        return repr(self) == repr(other)

    #def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
    #    "Used for pickle and copy operations."
    #    raise NotImplementedError, "Must reimplement in each Terminal, or?"

#--- Subgroups of terminals ---

class FormArgument(Terminal):
    __slots__ = ()
    def __init__(self, count=None, countedclass=None):
        Terminal.__init__(self)
        counted_init(self, count, countedclass)

    def count(self):
        return self._count

class UtilityType(Terminal):
    __slots__ = ()
    def __init__(self):
        Terminal.__init__(self)

    def shape(self):
        error("Calling shape on a utility type is an error.")

    def free_indices(self):
        error("Calling free_indices on a utility type is an error.")

    def index_dimensions(self):
        error("Calling index_dimensions on a utility type is an error.")

    def is_cellwise_constant(self):
        error("Calling is_cellwise_constant on a utility type is an error.")

#--- Non-tensor types ---

class Data(UtilityType):
    "For internal use, never to be created by users."
    __slots__ = ("_data",)
    def __init__(self, data):
        UtilityType.__init__(self)
        self._data = data

    def __str__(self):
        return "Data(%s)" % str(self._data)

    def __repr__(self):
        return "Data(%r)" % (self._data,)

