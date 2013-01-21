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
    __slots__ = ()
    def __init__(self, cell, name=None, gdim=None, tdim=None):
        DomainDescription.__init__(self, cell, name, gdim, tdim)

    def __repr__(self):
        return "Domain(%r, %r, %r, %r)" % (self._cell, self._name,
                                           self._gdim, self._tdim)

    def __eq__(self, other):
        return (isinstance(other, Domain)
                and DomainDescription.__eq__(self, other))

    def top_domain(self):
        return self

    def subdomain_ids(self):
        return None

    def __getitem__(self, subdomain_id):
        if isinstance(subdomain_id, int):
            return Region(self, (subdomain_id,), "%s_%d" % (self._name, subdomain_id))
        else:
            error("Invalid subdomain label %r, expecting integer." % (subdomain_id,))

class Region(DomainDescription):
    __slots__ = ('_parent', '_subdomain_ids')
    def __init__(self, parent, subdomain_ids, name):
        DomainDescription.__init__(self, parent.cell(), name,
                                   parent._gdim, parent._tdim)
        self._parent = parent
        self._subdomain_ids = tuple(sorted(set(subdomain_ids)))
        ufl_assert(name != parent.name(), "Cannot assign same name to top level domain and region.")

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

# TODO: Move somewhere else
def extract_top_domains(integrand):
    from ufl.terminal import FormArgument
    from ufl.algorithms.traversal import traverse_terminals
    top_domains = set()
    for t in traverse_terminals(integrand):
        if isinstance(t, FormArgument):
            domain = t.element().domain()
            if domain is not None:
                top_domains.add(domain.top_domain())
        # FIXME: Check geometry here too when that becomes necessary
    return sorted(top_domains)

def extract_domains(integrand):
    from ufl.terminal import FormArgument
    from ufl.algorithms.traversal import traverse_terminals
    regions = set()
    for t in traverse_terminals(integrand): # FIXME: Need to analyse which components of functions are actually referenced
        if isinstance(t, FormArgument):
            reg = t.element().regions() # FIXME: Implement
            regions.update(reg)
            regions.update(r.top_domain() for r in reg)
        # FIXME: Check geometry here too when that becomes necessary
    return sorted(regions)
