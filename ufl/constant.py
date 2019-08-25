# -*- coding: utf-8 -*-
"""This module defines classes representing non-literal values
which are constant with respect to a domain."""

# Copyright (C) 2019 Michal Habera
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

from ufl.utils.str import as_native_str
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal
from ufl.domain import as_domain
from ufl.utils.counted import counted_init


@ufl_type()
class Constant(Terminal):
    _ufl_noslots_ = True
    _globalcount = 0

    def __init__(self, domain, shape=(), count=None):
        Terminal.__init__(self)
        counted_init(self, count=count, countedclass=Constant)

        self._ufl_domain = as_domain(domain)
        self._ufl_shape = shape

        # Repr string is build in such way, that reconstruction
        # with eval() is possible
        self._repr = as_native_str("Constant(%s, %s, %s)" % (
            repr(self._ufl_domain), repr(self._ufl_shape), repr(self._count)))

    def count(self):
        return self._count

    @property
    def ufl_shape(self):
        return self._ufl_shape

    def ufl_domain(self):
        return self._ufl_domain

    def ufl_domains(self):
        return (self.ufl_domain(), )

    def is_cellwise_constant(self):
        return True

    def __str__(self):
        count = str(self._count)
        return "c_{%s}" % count

    def __repr__(self):
        return self._repr


def VectorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), ), count=count)


def TensorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), domain.geometric_dimension()), count=count)
