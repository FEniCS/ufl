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

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.common import istr
from ufl.terminal import Terminal

from ufl.geometry import Cell

EVERYWHERE = "everywhere"
class Domain(object):
    __slots__ = ('_cell', '_name',
                 '_gdim', '_tdim',
                 '_numbered_subdomains',
                 '_named_subdomains',)
    def __init__(self, cell, name=EVERYWHERE):
        self._cell = cell
        self._name = name
        self._numbered_subdomains = {}
        self._named_subdomains = {}

        # At the moment we just get dimensions from the cell, but
        # later we can make it primarily a domain property
        self._gdim = cell.geometric_dimension()
        self._tdim = cell.topological_dimension()

    def geometric_dimension(self):
        return self._gdim

    def topological_dimension(self):
        return self._tdim

    def cell(self):
        return self._cell

    def name(self):
        return self._name

    def parent_domain(self):
        return None

    def top_domain(self):
        return self

    def is_top_domain(self):
        return self is self.top_domain()

    def disjoint_subdomain_ids(self):
        return sorted(self._numbered_subdomains.keys())

    def disjoint_subdomains(self):
        return sorted(self._numbered_subdomains.values())

    def subdomain_group_names(self):
        return sorted(self._named_subdomains.keys())

    def subdomain_groups(self):
        return sorted(self._named_subdomains.values())

    def __getitem__(self, name):
        if isinstance(name, int):
            dom = self._numbered_subdomains.get(name)
            if dom is None:
                dom = DisjointSubDomain(self, name)
                self._numbered_subdomains[name] = dom
            return dom

        if isinstance(name, str):
            dom = self._named_subdomains.get(name)
            if dom is None:
                error("No record of subdomain with label %r" % name)
            return dom

        error("Invalid subdomain label %r" % name)

    def __call__(self, name=EVERYWHERE):
        return Boundary(self, name)

    def __repr__(self):
        return "Domain(%r, %r)" % (self._cell, self._name)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (isinstance(other, Domain)
                and self._cell == other._cell
                and self._name == other._name)

    def __lt__(self, other):
        return repr(self) < repr(other) # FIXME: Sort in a more predictable way

class DisjointSubDomain(Domain):
    __slots__ = ('_parent', '_number')
    def __init__(self, parent, number):
        Domain.__init__(self, parent.cell(), name="%s[%d]" % (parent.name(), number))
        self._parent = parent
        self._number = number
        parent._numbered_subdomains[number] = self

    def top_domain(self):
        return self._parent.top_domain()

    def __repr__(self):
        return "DisjointSubDomain(%r, %r)" % (self._parent, self._name)

    def __eq__(self, other):
        return (isinstance(other, DisjointSubDomain)
                and self._parent == other._parent
                and self._name == other._name)

class DomainGroup(Domain):
    __slots__ = ('_parent', '_subdomains')
    def __init__(self, parent, subdomains, name):
        Domain.__init__(self, parent.cell(), name)
        self._parent = parent
        self._subdomains = subdomains
        parent._named_subdomains[name] = self

    def top_domain(self):
        return self._parent.top_domain()

    def __repr__(self):
        return "DomainGroup(%r, %r, %r)" % (self._parent, self._subdomains, self._name)

    def __eq__(self, other):
        return (isinstance(other, DomainGroup)
                and self._parent == other._parent
                and self._subdomains == other._subdomains
                and self._name == other._name)

# Map cells to a default domain for compatibility and cache this:
_default_domains = {}
def as_domain(domain):
    if isinstance(domain, Domain):
        return domain
    elif isinstance(domain, Cell):
        if domain not in _default_domains:
            _default_domains[domain] = Domain(domain, name=EVERYWHERE)
        return _default_domains[domain]
    else:
        error("Invalid domain %s." % str(domain))
