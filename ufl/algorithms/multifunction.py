"""Multifunctions."""

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
# First added:  2008-05-07
# Last changed: 2012-04-10

from ufl.log import error
from ufl.classes import all_ufl_classes

class MultiFunction(object):
    """Base class for collections of nonrecursive expression node handlers."""
    _handlers_cache = {}
    def __init__(self):
        # Analyse class properties and cache handler data the
        # first time this is run for a particular class
        cache_data = MultiFunction._handlers_cache.get(type(self))
        if not cache_data:
            cache_data = [None]*len(all_ufl_classes)
            # For all UFL classes
            for classobject in all_ufl_classes:
                # Iterate over the inheritance chain
                # (NB! This assumes that all UFL classes inherits from
                # a single Expr subclass and that the first superclass
                # is always from the UFL Expr hierarchy!)
                for c in classobject.mro():
                    # Register classobject with handler for the first encountered superclass
                    name = c._handlername
                    if getattr(self, name, None):
                        cache_data[classobject._classid] = name
                        break
            MultiFunction._handlers_cache[type(self)] = cache_data
        # Build handler list for this particular class (get functions bound to self)
        self._handlers = [getattr(self, name) for name in cache_data]

    def __call__(self, o, *args, **kwargs):
        h = self._handlers[o._classid]
        return h(o, *args, **kwargs)

    def undefined(self, o):
        "Trigger error."
        error("No handler defined for %s." % o._uflclass.__name__)

    # Set default behaviour for any Expr
    expr = undefined
