"""Base class for multifunctions with UFL Expr type dispatch."""

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

from inspect import getargspec

from ufl.log import error
from ufl.core.expr import Expr

def get_num_args(function):
    insp = getargspec(function)
    return len(insp[0]) + int(insp[1] is not None)

class MultiFunction(object):
    """Base class for collections of nonrecursive expression node handlers.

    Subclass this (remember to call the __init__ method of this class),
    and implement handler functions for each Expr type, using the lower case
    handler name of the type (exprtype._ufl_handler_name_).

    This class is optimized for efficient type based dispatch in the __call__
    operator via typecode based lookup of the handler function bound to the
    algorithm object. Of course function call overhead of Python still applies.
    """

    _handlers_cache = {}

    def __init__(self):
        # Analyse class properties and cache handler data the
        # first time this is run for a particular class
        # (cached for each algorithm for performance)
        algorithm_class = type(self).__name__
        cache_data = MultiFunction._handlers_cache.get(algorithm_class)
        if not cache_data:
            cache_data = [None]*len(Expr._ufl_all_classes_)

            # Iterate over the inheritance chain for each Expr subclass
            # (NB! This assumes that all UFL classes inherits from
            # a single Expr subclass and that the first superclass
            # is always from the UFL Expr hierarchy!)
            for classobject in Expr._ufl_all_classes_:
                for c in classobject.mro():
                    # Register classobject with handler for the first encountered superclass
                    name = c._ufl_handler_name_
                    if hasattr(self, name):
                        cache_data[classobject._ufl_typecode_] = name
                        break
            MultiFunction._handlers_cache[algorithm_class] = cache_data

        # Build handler list for this particular class
        # (get functions bound to self, these cannot be cached)
        self._handlers = [getattr(self, name) for name in cache_data]
        self._is_cutoff_type = [get_num_args(h) == 2 for h in self._handlers]

    def __call__(self, o, *args):
        "Delegate to handler function based on typecode of first argument."
        return self._handlers[o._ufl_typecode_](o, *args)

    def undefined(self, o, *args):
        "Trigger error for types with missing handlers."
        error("No handler defined for %s." % o._ufl_class_.__name__)

    def reuse_if_untouched(self, o, *ops):
        """Reuse object if operands are the same objects.

        Use in your own subclass by setting e.g.

            expr = MultiFunction.reuse_if_untouched

        as a default rule.
        """
        if all(a is b for a, b in zip(o.ufl_operands, ops)):
            return o
        else:
            return o.reconstruct(*ops)

    # Set default behaviour for any Expr as undefined
    expr = undefined
