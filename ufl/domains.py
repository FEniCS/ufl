"Types for specification of domains and subdomain relations."

# Copyright (C) 2013 Martin Sandve Alnes
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
# First added:  2013-01-10
# Last changed: 2013-01-10

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.common import istr
from ufl.terminal import Terminal

from ufl.geometry import as_cell, Cell

class DomainDescription(object):
    __slots__ = ('_cell',
                 '_name',
                 '_gdim',
                 '_tdim',
                 )
    def __init__(self, cell, name, gdim, tdim):
        ufl_assert(isinstance(cell, Cell), "Expecting a Cell.")

        self._cell = cell
        self._name = name or "%s_%s" % (cell.cellname(), "multiverse")
        self._gdim = gdim or cell.geometric_dimension()
        self._tdim = tdim or cell.topological_dimension()

        ufl_assert(isinstance(self._name, str), "Expecting a string.")
        ufl_assert(isinstance(self._gdim, int), "Expecting an integer.")
        ufl_assert(isinstance(self._tdim, int), "Expecting an integer.")

    def geometric_dimension(self):
        return self._gdim

    def topological_dimension(self):
        return self._tdim

    def cell(self):
        return self._cell

    def name(self):
        return self._name

    def top_domain(self):
        raise NotImplementedException("Missing implementation of top_domain.")

    def __eq__(self, other):
        return (isinstance(other, DomainDescription)
                and self._cell == other._cell
                and self._name == other._name
                and self._gdim == other._gdim
                and self._tdim == other._tdim)

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return repr(self) < repr(other) # FIXME: Sort in a more predictable way

class Domain(DomainDescription):
    __slots__ = ('_regions',)
    def __init__(self, cell, name=None, gdim=None, tdim=None,
                 regions=None):
        DomainDescription.__init__(self, cell, name, gdim, tdim)
        self._regions = {} if regions is None else regions

    def __repr__(self):
        return "Domain(%r, %r, %r, %r, %r)" % (self._cell, self._name,
                                               self._gdim, self._tdim, self._regions)

    def __eq__(self, other):
        return (isinstance(other, Domain)
                and DomainDescription.__eq__(self, other)
                and self._regions == other._regions)

    def top_domain(self):
        return self

    def region_names(self):
        return sorted(self._regions.keys())

    def regions(self):
        return [self[r] for r in self.region_names()]

    # TODO: Can we make it possible to get all subdomain ids for a Domain?
    #def subdomain_ids(self):
    #    return self._subdomain_ids

    def __getitem__(self, name):
        if isinstance(name, int):
            return Region(self, (name,), "%s_%d" % (self._name, name))
        elif isinstance(name, str):
            dom = self._regions.get(name)
            if dom is None:
                error("No record of subdomain with label %r" % name)
            return dom
        else:
            error("Invalid subdomain label %r, expecting string or integer." % name)

    # TODO: Does this make sense?
    #def __call__(self, name):
    #    return Boundary(self, name)

class Region(DomainDescription):
    __slots__ = ('_parent', '_subdomain_ids')
    def __init__(self, parent, subdomain_ids, name):
        DomainDescription.__init__(self, parent.cell(), name,
                                   parent._gdim, parent._tdim)
        self._parent = parent
        self._subdomain_ids = tuple(sorted(set(subdomain_ids)))
        parent._regions[name] = self

    def top_domain(self):
        return self._parent

    def subdomain_ids(self):
        return self._subdomain_ids

    def __repr__(self):
        return "Region(%r, %r, %r)" % (self._parent, self._subdomain_ids, self._name)

    def __eq__(self, other):
        return (isinstance(other, Region)
                and self._parent == other._parent
                and self._subdomain_ids == other._subdomain_ids
                and DomainDescription.__eq__(self, other))

# Map cells to a default domain for compatibility and cache this:
_default_domains = {}
def as_domain(domain):
    if isinstance(domain, DomainDescription):
        return domain
    else:
        cell = as_cell(domain)
        if isinstance(cell, Cell):
            if cell not in _default_domains:
                _default_domains[cell] = Domain(cell)
            return _default_domains[cell]
        else:
            error("Invalid domain %s." % str(domain))
