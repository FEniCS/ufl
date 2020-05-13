#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

"""
Test use of transformed.
"""

import pytest
from ufl import *
from ufl.algorithms import compute_form_data


def test_transformed_one_form():
    cell = triangle
    element = VectorElement("Lagrange", cell, 1)

    # Mesh with some underlying coordinates.
    domain = Mesh(cell)
    # Regular function space on the fixed mesh.
    V = FunctionSpace(domain, element)

    # Topological mesh with no underlying coordinates.
    t_domain = TopologicalMesh(cell)
    # Topological function space on the topological mesh.
    t_V = TopologicalFunctionSpace(t_domain, element)

    c = Coefficient(V)
    # TestFunction defined on the full space V.
    v = TestFunction(V)
    # TestFunction restricted onto subspace V0 of V,
    # which, for instance, represents V modified to
    # apply Dirichlet boundary conditions strongly.
    # Given representation of V on which v is defined,
    # this restriction can be encoded in a topological
    # coefficient defined on the associated topological
    # function space t_V.
    transform_op = TopologicalCoefficient(t_V)
    v0 = Transformed(v, transform_op)

    # Some form for testing.
    form = inner(c, grad(v0[1])) * dx

    fd = compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=True,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(),
        do_apply_restrictions=True,
        do_apply_transforms=True,
        do_estimate_degrees=True,
        complex_mode=True
    )

    assert fd.num_topological_coefficients == 1
    assert fd.reduced_topological_coefficients[0] is transform_op
    assert fd.original_topological_coefficient_positions == [0, ]
    assert fd.integral_data[0].integral_topological_coefficients == set((transform_op, ))
    assert fd.integral_data[0].enabled_topological_coefficients == [True, ]
