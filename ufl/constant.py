# -*- coding: utf-8 -*-
"""This module defines classes representing non-literal values
which are constant with respect to a domain."""

# Copyright (C) 2019 Michal Habera
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
        self._repr = "Constant({}, {}, {})".format(
            repr(self._ufl_domain), repr(self._ufl_shape), repr(self._count))

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
        if len(count) == 1:
            return "c_%s" % count
        else:
            return "c_{%s}" % count

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False
        if self is other:
            return True
        return (self._count == other._count and
                self._ufl_domain == other._ufl_domain and
                self._ufl_shape == self._ufl_shape)

    def _ufl_signature_data_(self, renumbering):
        "Signature data for constant depends on renumbering"
        return "Constant({}, {}, {})".format(
            repr(self._ufl_domain), repr(self._ufl_shape), repr(renumbering[self]))


def VectorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), ), count=count)


def TensorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), domain.geometric_dimension()), count=count)
