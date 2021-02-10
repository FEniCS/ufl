# -*- coding: utf-8 -*-
"Predicates for recognising duals"

# Copyright (C) 2015-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by India Marsden, 2021

# from ufl.log import error
# from ufl.functionspace import FunctionSpace, DualSpace, MixedFunctionSpace


def is_primal(object):
    """ determines if the object belongs to a primal space"""
    return hasattr(object, '_primal') and object._primal


def is_dual(object):
    """ determines if the object belongs to a dual space"""
    return hasattr(object, '_dual') and object._dual
