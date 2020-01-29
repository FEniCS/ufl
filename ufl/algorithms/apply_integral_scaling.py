# -*- coding: utf-8 -*-
"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2013-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.log import error
from ufl.classes import JacobianDeterminant, FacetJacobianDeterminant, QuadratureWeight, Form, Integral
from ufl.measure import custom_integral_types, point_integral_types
from ufl.differentiation import CoordinateDerivative
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree


def compute_integrand_scaling_factor(integral):
    """Change integrand geometry to the right representations."""

    domain = integral.ufl_domain()
    integral_type = integral.integral_type()
    # co = CellOrientation(domain)
    weight = QuadratureWeight(domain)
    tdim = domain.topological_dimension()
    # gdim = domain.geometric_dimension()

    # Polynomial degree of integrand scaling
    degree = 0
    if integral_type == "cell":
        detJ = JacobianDeterminant(domain)
        degree = estimate_total_polynomial_degree(apply_geometry_lowering(detJ))
        # Despite the abs, |detJ| is polynomial except for
        # self-intersecting cells, where we have other problems.
        scale = abs(detJ) * weight

    elif integral_type.startswith("exterior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant and
            # quadrature weight
            detFJ = FacetJacobianDeterminant(domain)
            degree = estimate_total_polynomial_degree(apply_geometry_lowering(detFJ))
            scale = detFJ * weight
        else:
            # No need to scale 'integral' over a vertex
            scale = 1

    elif integral_type.startswith("interior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant from one
            # side and quadrature weight
            detFJ = FacetJacobianDeterminant(domain)
            degree = estimate_total_polynomial_degree(apply_geometry_lowering(detFJ))
            scale = detFJ('+') * weight
        else:
            # No need to scale 'integral' over a vertex
            scale = 1

    elif integral_type in custom_integral_types:
        # Scaling with custom weight, which includes eventual volume
        # scaling
        scale = weight

    elif integral_type in point_integral_types:
        # No need to scale 'integral' over a point
        scale = 1

    else:
        error("Unknown integral type {}, don't know how to scale.".format(integral_type))

    return scale, degree


def apply_integral_scaling(form):
    "Multiply integrands by a factor to scale the integral to reference frame."
    # TODO: Consider adding an in_reference_frame property to Integral
    #       and checking it here and setting it in the returned form
    if isinstance(form, Form):
        newintegrals = [apply_integral_scaling(integral)
                        for integral in form.integrals()]
        return Form(newintegrals)

    elif isinstance(form, Integral):
        integral = form
        integrand = integral.integrand()
        # Compute and apply integration scaling factor since we want to compute
        # coordinate derivatives at the end, the scaling factor has to be moved
        # inside those
        scale, degree = compute_integrand_scaling_factor(integral)
        md = {}
        md.update(integral.metadata())
        new_degree = degree
        cur_degree = md.get("estimated_polynomial_degree")
        if cur_degree is not None:
            if isinstance(cur_degree, tuple) and isinstance(degree, tuple):
                new_degree = tuple(d[0] + d[1] for d in zip(cur_degree, degree))
            elif isinstance(cur_degree, tuple):
                new_degree = tuple(d + degree for d in cur_degree)
            elif isinstance(degree, tuple):
                new_degree = tuple(cur_degree + d for d in degree)
            else:
                new_degree = cur_degree + degree
        md["estimated_polynomial_degree"] = new_degree

        def scale_coordinate_derivative(o, scale):
            o_ = o.ufl_operands
            if isinstance(o, CoordinateDerivative):
                return CoordinateDerivative(scale_coordinate_derivative(o_[0], scale), o_[1], o_[2], o_[3])
            else:
                return scale * o
        newintegrand = scale_coordinate_derivative(integrand, scale)
        return integral.reconstruct(integrand=newintegrand, metadata=md)

    else:
        error("Invalid type %s" % (form.__class__.__name__,))
