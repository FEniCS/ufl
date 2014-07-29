"""This module defines the Terminal class, the superclass
for all types that are terminal nodes in the expression trees."""

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
# Modified by Anders Logg, 2008

from ufl.expr import Expr
from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import EmptyDict
from ufl.common import counted_init
from ufl.core.ufl_type import ufl_type

#--- Base class for terminal objects ---

@ufl_type(is_abstract=True, is_terminal=True)
class Terminal(Expr):
    "A terminal node in the UFL expression tree."
    __slots__ = ("_hash",)

    def __init__(self):
        Expr.__init__(self)
        self._hash = None

    def reconstruct(self, *operands):
        "Return self."
        operands and error("Got call to reconstruct in a terminal with non-empty operands.")
        return self

    ufl_operands = ()
    ufl_free_indices = ()
    ufl_index_dimensions = ()

    def operands(self):
        "A Terminal object never has operands."
        return ()

    def free_indices(self):
        "A Terminal object never has free indices."
        return ()

    def index_dimensions(self):
        "A Terminal object never has free indices."
        return EmptyDict

    def domains(self):
        "Return tuple of domains related to this terminal object."
        raise NotImplementedError("Missing implementation of domains().")

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

    def signature_data(self, renumbering):
        "Default signature data for of terminals just return the repr string."
        return repr(self)

    def __hash__(self):
        "Default hash of terminals just hash the repr string."
        if self._hash is None:
            self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, other):
        "Default comparison of terminals just compare repr strings."
        return repr(self) == repr(other)

    #def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
    #    "Used for pickle and copy operations."
    #    raise NotImplementedError, "Must reimplement in each Terminal, or?"

#--- Subgroups of terminals ---

@ufl_type(is_abstract=True)
class FormArgument(Terminal):
    __slots__ = ()
    def __init__(self):
        Terminal.__init__(self)

# TODO: This type breaks the data model, can we make it fit in better?
@ufl_type(is_abstract=True)
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

    def domains(self):
        "Return tuple of domains related to this terminal object."
        return ()
