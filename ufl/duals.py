# -*- coding: utf-8 -*-
"""Predicates for recognising duals"""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#


def is_primal(object):
    """Determine if the object belongs to a primal space

    This is not simply the negation of :func:`is_dual`,
    because a mixed function space containing both primal
    and dual components is neither primal nor dual."""
    return hasattr(object, '_primal') and object._primal


def is_dual(object):
    """Determine if the object belongs to a dual space

    This is not simply the negation of :func:`is_primal`,
    because a mixed function space containing both primal
    and dual components is neither primal nor dual."""
    return hasattr(object, '_dual') and object._dual
