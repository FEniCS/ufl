# -*- coding: utf-8 -*-
"""This module defines classes representing non-literal values
which are constant with respect to a domain."""

# Copyright (C) 2019 Michal Habera
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.terminal import Terminal
from ufl.core.ufl_type import ufl_type
from ufl.domain import as_domain
from ufl.utils.counted import Counted


@ufl_type()
class Constant(Terminal, Counted):
    _ufl_noslots_ = True

    def __init__(self, domain, shape=(), count=None):
        Terminal.__init__(self)
        Counted.__init__(self, count, Constant)

        self._ufl_domain = as_domain(domain)
        self._ufl_shape = shape

        # Repr string is build in such way, that reconstruction
        # with eval() is possible
        self._repr = "Constant({}, {}, {})".format(
            repr(self._ufl_domain), repr(self._ufl_shape), repr(self._count))

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
        return f"c_{self._count}"

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False
        if self is other:
            return True
        return (self._count == other._count and self._ufl_domain == other._ufl_domain and   # noqa: W504
                self._ufl_shape == self._ufl_shape)

    def _ufl_signature_data_(self, renumbering):
        "Signature data for constant depends on renumbering"
        return "Constant({}, {}, {})".format(
            self._ufl_domain._ufl_signature_data_(renumbering), repr(self._ufl_shape),
            repr(renumbering[self]))


def VectorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), ), count=count)


def TensorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), domain.geometric_dimension()), count=count)
