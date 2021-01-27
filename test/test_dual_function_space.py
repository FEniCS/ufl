#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

from ufl import *

__authors__ = "India Marsden"
__date__ = "2020-12-28 -- 2020-12-28"

import pytest

from ufl import *
from ufl.domain import default_domain
from ufl.duals import is_primal,is_dual



def test_mixed_functionspace(self):
    # Domains
    domain_3d = default_domain(tetrahedron)
    domain_2d = default_domain(triangle)
    domain_1d = default_domain(interval)
    # Finite elements
    f_1d = FiniteElement("CG", interval, 1)
    f_2d = FiniteElement("CG", triangle, 1)
    f_3d = FiniteElement("CG", tetrahedron, 1)
    # Function spaces
    V_3d = FunctionSpace(domain_3d, f_3d)
    V_2d = FunctionSpace(domain_2d, f_2d)
    V_1d = FunctionSpace(domain_1d, f_1d)

    # MixedFunctionSpace = V_3d x V_2d x V_1d
    V = MixedFunctionSpace(V_3d, V_2d, V_1d)
    # Check sub spaces
    assert(is_primal(V_3d))
    assert(is_primal(V_2d))
    assert(is_primal(V_1d))
    assert(is_primal(V))

     # Function spaces
    V_dual = FunctionSpace(domain_3d, f_3d)
    

    # MixedFunctionSpace = V_dual x V_2d x V_1d
    V = MixedFunctionSpace(V_dual, V_2d, V_1d)

    assert(is_dual(V_dual))
    assert(is_dual(V))
