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
from ufl.functionspace import FunctionSpace, DualSpace, MixedFunctionSpace


def is_primal(object):
    if isinstance(object, FunctionSpace):
        return True
    elif isinstance(object, MixedFunctionSpace):
        return all([is_primal(subspace) for subspace in object.ufl_sub_spaces()])
    return False


def is_dual(object):
    if isinstance(object, DualSpace):
        return True
    elif isinstance(object, MixedFunctionSpace):
        return all([is_dual(subspace) for subspace in object.ufl_sub_spaces()])
    return False
