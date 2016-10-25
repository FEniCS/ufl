# -*- coding: utf-8 -*-
"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2013-2016 Martin Sandve Alnæs
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

from ufl.log import error
from ufl.classes import JacobianDeterminant, FacetJacobianDeterminant, QuadratureWeight, Form, Integral
from ufl.measure import custom_integral_types, point_integral_types


def compute_integrand_scaling_factor(integral):
    """Change integrand geometry to the right representations."""

    domain = integral.ufl_domain()
    integral_type = integral.integral_type()
    # co = CellOrientation(domain)
    weight = QuadratureWeight(domain)
    tdim = domain.topological_dimension()
    # gdim = domain.geometric_dimension()

    if integral_type == "cell":
        scale = abs(JacobianDeterminant(domain)) * weight

    elif integral_type.startswith("exterior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant and
            # quadrature weight
            scale = FacetJacobianDeterminant(domain) * weight
        else:
            # No need to scale 'integral' over a vertex
            scale = 1

    elif integral_type.startswith("interior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant from one
            # side and quadrature weight
            scale = FacetJacobianDeterminant(domain)('+') * weight
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

    return scale


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
        # Compute and apply integration scaling factor
        scale = compute_integrand_scaling_factor(integral)
        newintegrand = integral.integrand() * scale
        return integral.reconstruct(integrand=newintegrand)

    else:
        error("Invalid type %s" % (form.__class__.__name__,))
